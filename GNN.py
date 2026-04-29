"""
GNN Criticality Predictor — Danish Transmission Grid

Architecture : 2-layer GCN → per-node criticality score
               Edge criticality = concat(endpoint embeddings) → MLP head
Training data: scenario_results.xlsx produced by simulation_all_single_removals()
               One row per (scenario, removed node/edge) → total_unmet_demand_MW label

Usage (from Master.ipynb or standalone):
    from GNN import train_gnn, predict_criticality
    model, t_mean, t_std = train_gnn(scenario_graphs, results_df)
    preds = predict_criticality(model, G_new, t_mean, t_std)
"""

import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import pandas as pd
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


# ── Source encoding ────────────────────────────────────────────────────────────
# Add new source types here if the grid data changes
SOURCE_TYPES = sorted(["", "gas", "hydro", "international", "other",
                        "solar", "unknown", "wind_offshore", "wind_onshore"])
N_SOURCES = len(SOURCE_TYPES)

def _encode_source(source_str):
    vec = [0.0] * N_SOURCES
    for s in str(source_str).split(", "):
        s = s.strip()
        if s in SOURCE_TYPES:
            vec[SOURCE_TYPES.index(s)] = 1.0
    return vec


# ── Feature extraction ─────────────────────────────────────────────────────────
# Numeric features per node (edit this list if you add attributes later):
#   supply, demand, p_min, p_max, degree, betweenness, is_generator, is_load
#   + voltage_kv (TODO: store this in aggregate_graph node attrs to enable)
N_NUMERIC = 8

def _node_features(G):
    """Returns (node_list, np.ndarray shape [N, N_NUMERIC + N_SOURCES])."""
    node_list = sorted(G.nodes())
    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, normalized=True)

    rows = []
    for n in node_list:
        d = G.nodes[n]
        supply  = float(d.get("supply",  0.0))
        demand  = float(d.get("demand",  0.0))
        p_min   = float(d.get("p_min",   0.0))
        p_max   = float(d.get("p_max",   0.0))
        deg     = float(degree[n])
        btwn    = float(betweenness[n])
        is_gen  = 1.0 if p_max  > 0 else 0.0
        is_load = 1.0 if demand > 0 else 0.0
        # TODO: add voltage_kv once stored on aggregated graph nodes
        rows.append([supply, demand, p_min, p_max, deg, btwn, is_gen, is_load]
                    + _encode_source(d.get("source", "")))

    return node_list, np.array(rows, dtype=np.float32)


def _graph_to_pyg(G):
    """Convert aggregated NetworkX graph to a PyG Data object."""
    node_list, feat_mat = _node_features(G)
    node_idx = {n: i for i, n in enumerate(node_list)}

    x = torch.from_numpy(feat_mat)

    srcs, dsts, caps = [], [], []
    for u, v, edata in G.edges(data=True):
        if u not in node_idx or v not in node_idx:
            continue
        i, j = node_idx[u], node_idx[v]
        cap = float(edata.get("capacity", 0.0))
        # log-scale capacity; transformers have cap=99999 — treat separately
        cap_log = np.log1p(min(cap, 10_000.0))
        srcs += [i, j];  dsts += [j, i];  caps += [cap_log, cap_log]

    edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
    edge_attr  = torch.tensor(caps, dtype=torch.float).unsqueeze(1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), node_idx


def _parse_removal(val):
    """Handle int node IDs and '(u, v)' edge strings that come back from Excel."""
    if isinstance(val, tuple):
        return val
    s = str(val).strip()
    if s.startswith("("):
        parts = s.strip("()").split(",")
        return tuple(int(p.strip()) for p in parts)
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return val


def _build_targets(results_df, node_idx, scenario_id):
    """
    Returns:
      node_y     : FloatTensor (N,)  — NaN where node absent from results
      edge_pairs : list of (i, j) int tuples
      edge_y     : FloatTensor (len(edge_pairs),)
    """
    df = results_df[results_df["scenario_id"] == scenario_id]
    N  = len(node_idx)

    node_y = torch.full((N,), float("nan"))
    edge_pairs, edge_vals = [], []

    for _, row in df.iterrows():
        removal = _parse_removal(row["removals"])
        target  = float(row["total_unmet_demand_MW"])

        if row["attack"] == "node":
            if removal in node_idx:
                node_y[node_idx[removal]] = target
        else:
            if isinstance(removal, tuple) and len(removal) == 2:
                u, v = removal
                if u in node_idx and v in node_idx:
                    edge_pairs.append((node_idx[u], node_idx[v]))
                    edge_vals.append(target)

    edge_y = torch.tensor(edge_vals, dtype=torch.float) if edge_vals else torch.empty(0)
    return node_y, edge_pairs, edge_y


# ── Model ──────────────────────────────────────────────────────────────────────
class GridGCN(nn.Module):
    """
    2-layer GCN that produces a criticality score per node (and per edge).

    Node score  : how much unmet demand results from removing that node.
    Edge score  : how much unmet demand results from removing that edge
                  (predicted from the concatenation of its endpoint embeddings).
    """

    def __init__(self, in_feats, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden, 32)

        self.node_head = nn.Linear(32, 1)

        self.edge_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _embed(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x,      data.edge_index))
        return x  # (N, 32)

    def forward_nodes(self, data):
        return self.node_head(self._embed(data)).squeeze(-1)  # (N,)

    def forward_edges(self, data, edge_pairs):
        """edge_pairs : list of (i, j) index tuples."""
        emb = self._embed(data)
        src = torch.tensor([p[0] for p in edge_pairs], dtype=torch.long)
        dst = torch.tensor([p[1] for p in edge_pairs], dtype=torch.long)
        return self.edge_head(torch.cat([emb[src], emb[dst]], dim=-1)).squeeze(-1)


# ── Target normalisation ───────────────────────────────────────────────────────
def _fit_scaler(results_df):
    vals = results_df["total_unmet_demand_MW"].dropna().astype(float).values
    return float(vals.mean()), float(vals.std()) + 1e-8

def _norm(y, mean, std):   return (y - mean) / std
def _denorm(y, mean, std): return y * std + mean


# ── Training ───────────────────────────────────────────────────────────────────
def train_gnn(scenario_graphs, results_df,
              epochs=300, lr=1e-3, hidden=64, val_scenario_ids=None):
    """
    Parameters
    ----------
    scenario_graphs   : dict  scenario_id → aggregated nx.Graph
    results_df        : DataFrame from simulation_all_single_removals (all scenarios stacked)
    val_scenario_ids  : list of scenario IDs to hold out for validation reporting

    Returns
    -------
    model, t_mean, t_std
    """
    t_mean, t_std = _fit_scaler(results_df)
    val_ids   = val_scenario_ids or []
    train_ids = [s for s in scenario_graphs if s not in val_ids]

    # Pre-build PyG data objects (betweenness is slow — do it once)
    print("Building graph features...")
    pyg = {}
    for sid, G in scenario_graphs.items():
        data, node_idx = _graph_to_pyg(G)
        node_y, edge_pairs, edge_y = _build_targets(results_df, node_idx, sid)
        pyg[sid] = (data, node_idx, node_y, edge_pairs, edge_y)
        print(f"  {sid}: {len(node_idx)} nodes, "
              f"{node_y.isnan().logical_not().sum().item()} node labels, "
              f"{len(edge_pairs)} edge labels")

    in_feats = next(iter(pyg.values()))[0].x.shape[1]
    model    = GridGCN(in_feats=in_feats, hidden=hidden)
    opt      = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched    = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)

    print(f"\nTraining on {len(train_ids)} scenarios | val: {val_ids or 'none'}")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for sid in train_ids:
            data, _, node_y, edge_pairs, edge_y = pyg[sid]
            opt.zero_grad()
            loss_terms = []

            mask = ~node_y.isnan()
            if mask.any():
                pred_n = model.forward_nodes(data)
                loss_terms.append(F.mse_loss(pred_n[mask],
                                             _norm(node_y[mask], t_mean, t_std)))

            if len(edge_pairs) > 0:
                pred_e = model.forward_edges(data, edge_pairs)
                loss_terms.append(F.mse_loss(pred_e,
                                             _norm(edge_y, t_mean, t_std)))

            if loss_terms:
                loss = sum(loss_terms)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

        sched.step()

        if epoch % 50 == 0:
            val_mae = _eval_mae(model, pyg, val_ids, t_mean, t_std)
            print(f"Epoch {epoch:4d} | loss {epoch_loss/max(len(train_ids),1):.4f}"
                  + (f" | val MAE {val_mae:.1f} MW" if val_ids else ""))

    return model, t_mean, t_std


def _eval_mae(model, pyg, scenario_ids, t_mean, t_std):
    model.eval()
    errs = []
    with torch.no_grad():
        for sid in scenario_ids:
            data, _, node_y, edge_pairs, edge_y = pyg[sid]
            mask = ~node_y.isnan()
            if mask.any():
                pred = _denorm(model.forward_nodes(data), t_mean, t_std)
                errs.extend((pred[mask] - node_y[mask]).abs().tolist())
            if len(edge_pairs) > 0:
                pred_e = _denorm(model.forward_edges(data, edge_pairs), t_mean, t_std)
                errs.extend((pred_e - edge_y).abs().tolist())
    return float(np.mean(errs)) if errs else float("nan")


# ── Inference ──────────────────────────────────────────────────────────────────
def predict_criticality(model, G, t_mean, t_std):
    """
    Rank every node and edge in G by predicted unmet demand if removed.
    Returns a DataFrame sorted descending by predicted_unmet_MW.
    """
    model.eval()
    data, node_idx = _graph_to_pyg(G)
    idx_to_node    = {i: n for n, i in node_idx.items()}

    with torch.no_grad():
        node_scores = _denorm(model.forward_nodes(data), t_mean, t_std).tolist()

        edge_pairs = [(node_idx[u], node_idx[v])
                      for u, v in G.edges() if u in node_idx and v in node_idx]
        edge_scores = (_denorm(model.forward_edges(data, edge_pairs), t_mean, t_std).tolist()
                       if edge_pairs else [])

    rows = []
    for i, score in enumerate(node_scores):
        n = idx_to_node[i]
        rows.append({"attack": "node", "removal": n,
                     "name":   G.nodes[n].get("name",   ""),
                     "source": G.nodes[n].get("source", ""),
                     "predicted_unmet_MW": round(score, 2)})

    graph_edges = [(u, v) for u, v in G.edges() if u in node_idx and v in node_idx]
    for (u, v), score in zip(graph_edges, edge_scores):
        rows.append({"attack": "edge", "removal": (u, v),
                     "name":   G.edges[u, v].get("name", ""),
                     "source": "N/A",
                     "predicted_unmet_MW": round(score, 2)})

    return (pd.DataFrame(rows)
            .sort_values("predicted_unmet_MW", ascending=False)
            .reset_index(drop=True))


# ── Standalone entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import cleaning
    from scenarios import SCENARIOS, apply_scenario_mean

    file_path = "publicdataexportv131450706334_with_lon_lat.xlsx"
    sheet_names = ["Bus", "Line", "Generator", "Load", "HVDC", "Transformer2", "Transformer3"]
    raw = {s: pd.read_excel(file_path, sheet_name=s, header=3) for s in sheet_names}
    for df in raw.values():
        df.columns = [str(c).strip() for c in df.columns]

    g_disagg = cleaning.main_clean(
        raw["Bus"], raw["Line"], raw["Generator"], raw["Load"],
        raw["HVDC"], raw["Transformer2"], raw["Transformer3"]
    )
    edges_df = pd.read_excel("danish_grid_graph_ready.xlsx", sheet_name="edges")

    # Build one aggregated scenario graph per scenario
    results_df     = pd.read_excel("scenario_results.xlsx")
    available_ids  = set(results_df["scenario_id"].unique())

    scenario_graphs = {}
    for sid, scenario in SCENARIOS.items():
        if sid not in available_ids:
            continue
        G_sc = apply_scenario_mean(g_disagg, scenario)
        scenario_graphs[sid] = cleaning.aggregate_graph(G_sc, edges_df)

    # Train / val split by scenario (last 2 held out)
    all_ids   = list(scenario_graphs.keys())
    val_ids   = all_ids[-2:] if len(all_ids) > 2 else []

    model, t_mean, t_std = train_gnn(
        scenario_graphs, results_df,
        epochs=300, lr=1e-3, hidden=64,
        val_scenario_ids=val_ids,
    )

    torch.save({"model_state": model.state_dict(),
                "in_feats":    N_NUMERIC + N_SOURCES,
                "t_mean":      t_mean,
                "t_std":       t_std}, "gnn_model.pt")
    print("\nModel saved → gnn_model.pt")

    # Demo: predict criticality on first val scenario
    if val_ids:
        preds = predict_criticality(model, scenario_graphs[val_ids[0]], t_mean, t_std)
        print(f"\nTop 15 critical elements in {val_ids[0]}:")
        print(preds.head(15).to_string(index=False))
        preds.to_excel("gnn_predictions.xlsx", index=False)
        print("Predictions saved → gnn_predictions.xlsx")
