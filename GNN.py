### Claude Code has been used for the making of this file
"""
GNN Criticality Predictor — Danish Transmission Grid (k-removal edition)

Architecture : 2-layer GCN → graph-level mean+add pool → MLP → scalar unmet demand
               Input graph is the grid AFTER k elements have been removed.
Training data: CSV with columns
                 scenario_id, k, removed_nodes, removed_edges,
                 total_unmet_demand_MW, total_oversupply_MW, num_groups
               One row per (scenario, removal-set); k ≥ 1.

Usage:
    from GNN import train_gnn, predict_criticality, predict_criticality_k
    model, t_mean, t_std = train_gnn(scenario_graphs, train_df, val_df=val_df)
    preds  = predict_criticality(model, G, t_mean, t_std)           # k=1 ranking
    scores = predict_criticality_k(model, G, t_mean, t_std,
                                   [(removed_nodes, removed_edges)]) # arbitrary k
"""

import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import pandas as pd
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Data

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader


# ── Source encoding ────────────────────────────────────────────────────────────
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
# [0] supply  [1] demand  [2] p_addable  [3] p_removable
# [4] degree  [5] betweenness  [6] is_generator  [7] is_load
N_NUMERIC     = 8
CONTINUOUS_IDX = [0, 1, 2, 3, 4, 5]

def _node_features(G):
    """Returns (node_list, np.ndarray shape [N, N_NUMERIC + N_SOURCES])."""
    node_list  = sorted(G.nodes())
    degree     = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, normalized=True)

    rows = []
    for n in node_list:
        d = G.nodes[n]
        supply      = float(d.get("supply",      0.0))
        demand      = float(d.get("demand",      0.0))
        p_addable   = float(d.get("p_addable",   0.0))
        p_removable = float(d.get("p_removable", 0.0))
        deg         = float(degree[n])
        btwn        = float(betweenness[n])
        is_gen      = 1.0 if supply > 0 else 0.0
        is_load     = 1.0 if demand > 0 else 0.0
        rows.append([supply, demand, p_addable, p_removable, deg, btwn, is_gen, is_load]
                    + _encode_source(d.get("source", "")))

    return node_list, np.array(rows, dtype=np.float32)


def _graph_to_pyg(G):
    """Convert NetworkX graph to a PyG Data object."""
    node_list, feat_mat = _node_features(G)
    node_idx = {n: i for i, n in enumerate(node_list)}
    x        = torch.from_numpy(feat_mat)

    srcs, dsts, caps = [], [], []
    for u, v, edata in G.edges(data=True):
        if u not in node_idx or v not in node_idx:
            continue
        i, j   = node_idx[u], node_idx[v]
        cap_log = np.log1p(min(float(edata.get("capacity", 0.0)), 10_000.0))
        srcs += [i, j];  dsts += [j, i];  caps += [cap_log, cap_log]

    edge_index  = (torch.tensor([srcs, dsts], dtype=torch.long)
                   if srcs else torch.zeros((2, 0), dtype=torch.long))
    edge_weight = (torch.tensor(caps, dtype=torch.float)
                   if caps else torch.zeros(0))

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight), node_idx


# ── Schema parsing ─────────────────────────────────────────────────────────────
def _parse_list(val):
    """Parse a list-like string ('[]', '[1,2]', '[(1,2),(3,4)]') to a Python list."""
    s = str(val).strip()
    if s in ("", "nan", "[]"):
        return []
    try:
        return list(ast.literal_eval(s))
    except (ValueError, SyntaxError):
        return []


def _parse_removal_set(removed_nodes_val, removed_edges_val):
    """Return (list of int node IDs, list of (u, v) int tuples)."""
    nodes = [int(n) for n in _parse_list(removed_nodes_val)]
    edges = [tuple(int(x) for x in e) for e in _parse_list(removed_edges_val)]
    return nodes, edges


# ── Perturbed graph builder ────────────────────────────────────────────────────
def _build_perturbed_graph(G_base, removed_nodes, removed_edges):
    """
    Apply a removal-set to the base scenario graph.
    Edges incident to removed nodes are deleted automatically by NetworkX.
    Returns None if the resulting graph is empty.
    """
    G = G_base.copy()
    for u, v in removed_edges:
        if   G.has_edge(u, v): G.remove_edge(u, v)
        elif G.has_edge(v, u): G.remove_edge(v, u)
    for n in removed_nodes:
        if n in G:
            G.remove_node(n)
    if len(G.nodes()) == 0:
        return None
    return _graph_to_pyg(G)[0]


# ── Dataset builder ────────────────────────────────────────────────────────────
def _build_sample_dataset(results_df, scenario_graphs, scenario_ids=None):
    """
    For each row in results_df, build the perturbed graph and attach the label.
    Returns a list of PyG Data objects with a .y tensor (total_unmet_demand_MW).

    Deduplicates canonically (sorted frozensets) so list-order variants of the
    same removal set never produce duplicate Data objects.
    """
    ids = set(scenario_ids) if scenario_ids is not None else set(scenario_graphs.keys())
    df  = results_df[results_df["scenario_id"].isin(ids)]

    seen     = set()
    samples, skipped = [], 0
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 10_000 == 0 and i > 0:
            print(f"    {i}/{total} rows scanned, {len(samples)} unique graphs built...")
        sid = row["scenario_id"]
        if sid not in scenario_graphs:
            skipped += 1
            continue
        nodes, edges = _parse_removal_set(row["removed_nodes"], row["removed_edges"])

        # Canonical key: order-independent
        canon = (sid,
                 frozenset(nodes),
                 frozenset(tuple(sorted(e)) for e in edges))
        if canon in seen:
            continue
        seen.add(canon)

        data = _build_perturbed_graph(scenario_graphs[sid], nodes, edges)
        if data is None:
            skipped += 1
            continue
        data.y = torch.tensor([float(row["total_unmet_demand_MW"])], dtype=torch.float)
        samples.append(data)

    if skipped:
        print(f"    [warn] skipped {skipped} rows (missing scenario or empty graph)")
    return samples


# ── Model ──────────────────────────────────────────────────────────────────────
class GridGCN(nn.Module):
    """
    2-layer GCN on the post-removal grid.
    Node embeddings are pooled (mean + sum) to a single graph vector → MLP → unmet demand.
    """

    def __init__(self, in_feats, hidden=64):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv2 = GCNConv(hidden, 32)

        # mean-pool (32) + add-pool (32) concatenated → 64
        self.readout = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.register_buffer("feat_mean", torch.zeros(in_feats))
        self.register_buffer("feat_std",  torch.ones(in_feats))

    def set_feature_stats(self, mean, std):
        with torch.no_grad():
            self.feat_mean.copy_(mean)
            self.feat_std.copy_(std)

    def _normalize(self, x):
        return (x - self.feat_mean) / self.feat_std

    def _embed(self, data):
        x  = self._normalize(data.x)
        ew = getattr(data, "edge_weight", None)
        x  = F.relu(self.conv1(x, data.edge_index, ew))
        x  = F.relu(self.conv2(x, data.edge_index, ew))
        return x  # (N, 32)

    def forward(self, data):
        h     = self._embed(data)
        batch = getattr(data, "batch",
                        torch.zeros(h.size(0), dtype=torch.long, device=h.device))
        g = torch.cat([global_mean_pool(h, batch),
                       global_add_pool(h, batch)], dim=-1)  # (B, 64)
        return self.readout(g).squeeze(-1)                   # (B,) or scalar


# ── Target normalisation ───────────────────────────────────────────────────────
def _fit_scaler(results_df):
    vals = results_df["total_unmet_demand_MW"].dropna().astype(float).values
    return float(vals.mean()), float(vals.std()) + 1e-8

def _norm(y, mean, std):   return (y - mean) / std
def _denorm(y, mean, std): return y * std + mean


# ── Feature normalisation ──────────────────────────────────────────────────────
def _fit_feature_stats(feature_matrices, in_feats):
    stacked = np.concatenate(feature_matrices, axis=0)
    mean    = np.zeros(in_feats, dtype=np.float32)
    std     = np.ones(in_feats,  dtype=np.float32)
    for idx in CONTINUOUS_IDX:
        col      = stacked[:, idx]
        mean[idx] = float(col.mean())
        std[idx]  = float(col.std()) + 1e-8
    return torch.from_numpy(mean), torch.from_numpy(std)


# ── Training ───────────────────────────────────────────────────────────────────
def train_gnn(scenario_graphs, train_df, val_df=None,
              epochs=300, lr=1e-3, hidden=64,
              eval_every=1, patience=10, batch_size=32, pos_weight=3.0):
    """
    Parameters
    ----------
    scenario_graphs  : dict  scenario_id → base nx.Graph (pre-removal)
    train_df         : DataFrame of training rows (pre-split, pre-deduped)
    val_df           : DataFrame of validation rows (pre-split, pre-deduped); None to skip
    eval_every       : evaluate val MAE every N epochs
    patience         : early-stopping patience in evals; None to disable
    batch_size       : mini-batch size for DataLoader
    pos_weight       : loss weight for nonzero-label samples (compensates for ~72% zero labels)

    Returns
    -------
    model, t_mean, t_std
    """
    # Scaler fitted on training rows only — no leakage from val or test
    t_mean, t_std = _fit_scaler(train_df)

    print("Building perturbed graph dataset (betweenness computed once per unique removal)...")
    train_samples = _build_sample_dataset(train_df, scenario_graphs)
    val_samples   = _build_sample_dataset(val_df, scenario_graphs) if val_df is not None else []
    print(f"  train: {len(train_samples):,} unique graphs | val: {len(val_samples):,} unique graphs")

    if not train_samples:
        raise ValueError("No training samples built — check scenario_id overlap between "
                         "train_df and scenario_graphs.")

    in_feats  = train_samples[0].x.shape[1]
    feat_mats = [s.x.numpy() for s in train_samples]
    feat_mean, feat_std = _fit_feature_stats(feat_mats, in_feats)

    model = GridGCN(in_feats=in_feats, hidden=hidden)
    model.set_feature_stats(feat_mean, feat_std)

    loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                         patience=3, min_lr=1e-5)

    nz_frac = sum(1 for s in train_samples if s.y.item() > 0) / len(train_samples)
    n_scen  = len(train_df["scenario_id"].unique())
    print(f"\nTraining on {len(train_samples):,} samples ({n_scen} scenarios)"
          f" | val: {len(val_samples):,} samples"
          f" | nonzero labels: {nz_frac*100:.1f}%"
          f" | pos_weight: {pos_weight}")

    best_val_mae = float("inf")
    best_state   = None
    no_improve   = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            opt.zero_grad()
            pred   = model(batch)
            y_true = batch.y.view(-1)
            y_norm = _norm(y_true, t_mean, t_std)

            if pos_weight != 1.0:
                w    = torch.where(y_true > 0,
                                   torch.full_like(y_true, pos_weight),
                                   torch.ones_like(y_true))
                loss = (F.mse_loss(pred, y_norm, reduction="none") * w).mean()
            else:
                loss = F.mse_loss(pred, y_norm)

            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        if val_samples and epoch % eval_every == 0:
            val_mae  = _eval_mae(model, val_samples, t_mean, t_std)
            sched.step(val_mae)
            improved = val_mae < best_val_mae
            if improved:
                best_val_mae = val_mae
                best_state   = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve  += 1

            tag = " ★" if improved else ""
            print(f"Epoch {epoch:4d} | loss {epoch_loss/len(loader):.4f}"
                  f" | val MAE {val_mae:7.1f} MW (best {best_val_mae:7.1f}){tag}")

            if patience is not None and no_improve >= patience:
                print(f"\nEarly stop @ epoch {epoch} — no val improvement for "
                      f"{patience} epochs (best {best_val_mae:.1f} MW)")
                break

        else:
            sched.step(epoch_loss / len(loader))
            if epoch % 10 == 0:
                print(f"Epoch {epoch:4d} | loss {epoch_loss/len(loader):.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights (val MAE {best_val_mae:.1f} MW)")

    return model, t_mean, t_std


def _eval_mae(model, val_samples, t_mean, t_std):
    model.eval()
    errs   = []
    loader = DataLoader(val_samples, batch_size=64, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            pred = _denorm(model(batch), t_mean, t_std)
            true = batch.y.view(-1)
            errs.extend((pred - true).abs().tolist())
    return float(np.mean(errs)) if errs else float("nan")


def evaluate_test(model, test_samples, t_mean, t_std):
    """
    Full held-out test evaluation.
    Prints MAE / RMSE overall and broken down by actual removal count.
    Returns a DataFrame with per-sample predictions vs ground truth.
    """
    model.eval()
    preds_all, trues_all = [], []
    loader = DataLoader(test_samples, batch_size=64, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            preds_all.extend(_denorm(model(batch), t_mean, t_std).tolist())
            trues_all.extend(batch.y.view(-1).tolist())

    preds = np.array(preds_all)
    trues = np.array(trues_all)
    mae   = float(np.mean(np.abs(preds - trues)))
    rmse  = float(np.sqrt(np.mean((preds - trues) ** 2)))

    print(f"\n{'─'*50}")
    print(f"Test set  ({len(trues):,} samples)")
    print(f"  Overall  MAE : {mae:.1f} MW")
    print(f"  Overall  RMSE: {rmse:.1f} MW")

    nz = trues > 0
    if nz.sum() > 0:
        mae_nz  = float(np.mean(np.abs(preds[nz] - trues[nz])))
        rmse_nz = float(np.sqrt(np.mean((preds[nz] - trues[nz]) ** 2)))
        print(f"  Nonzero  MAE : {mae_nz:.1f} MW  (n={nz.sum():,})")
        print(f"  Nonzero  RMSE: {rmse_nz:.1f} MW")
    print(f"{'─'*50}")

    return pd.DataFrame({"predicted_unmet_MW": np.round(preds, 2),
                         "true_unmet_MW":      trues})


# ── Inference ──────────────────────────────────────────────────────────────────
def predict_criticality(model, G, t_mean, t_std):
    """
    k=1 ranking: score every node and edge by removing it individually.
    Returns DataFrame sorted by predicted_unmet_MW descending.
    (~N+E forward passes; a few seconds on CPU for 150-node graphs.)
    """
    model.eval()
    rows = []

    with torch.no_grad():
        for n in G.nodes():
            data = _build_perturbed_graph(G, removed_nodes=[n], removed_edges=[])
            if data is None:
                continue
            score = _denorm(model(data), t_mean, t_std).item()
            rows.append({"attack": "node", "removal": n,
                         "name":   G.nodes[n].get("name",   ""),
                         "source": G.nodes[n].get("source", ""),
                         "predicted_unmet_MW": round(score, 2)})

        for u, v in G.edges():
            data = _build_perturbed_graph(G, removed_nodes=[], removed_edges=[(u, v)])
            if data is None:
                continue
            score = _denorm(model(data), t_mean, t_std).item()
            rows.append({"attack": "edge", "removal": (u, v),
                         "name":   G.edges[u, v].get("name", ""),
                         "source": "N/A",
                         "predicted_unmet_MW": round(score, 2)})

    return (pd.DataFrame(rows)
            .sort_values("predicted_unmet_MW", ascending=False)
            .reset_index(drop=True))


def predict_criticality_k(model, G, t_mean, t_std, candidate_sets):
    """
    Score arbitrary k-removal sets in one pass each.

    Parameters
    ----------
    candidate_sets : list of (removed_nodes, removed_edges) tuples
                     e.g. [([12, 45], []), ([], [(3, 9), (5, 8)])]

    Returns DataFrame with columns:
        removed_nodes, removed_edges, k, predicted_unmet_MW
    sorted descending by predicted_unmet_MW.
    """
    model.eval()
    rows = []

    with torch.no_grad():
        for removed_nodes, removed_edges in candidate_sets:
            data  = _build_perturbed_graph(G, removed_nodes, removed_edges)
            k     = len(removed_nodes) + len(removed_edges)
            score = (_denorm(model(data), t_mean, t_std).item()
                     if data is not None else float("nan"))
            rows.append({"removed_nodes":       removed_nodes,
                         "removed_edges":       removed_edges,
                         "k":                   k,
                         "predicted_unmet_MW":  round(score, 2) if data else float("nan")})

    return (pd.DataFrame(rows)
            .sort_values("predicted_unmet_MW", ascending=False)
            .reset_index(drop=True))


# ── Standalone entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import cleaning
    from scenarios import SCENARIOS, apply_scenario_mean

    # ── Grid data ──────────────────────────────────────────────────────────────
    file_path   = "publicdataexportv131450706334_with_lon_lat.xlsx"
    sheet_names = ["Bus", "Line", "Generator", "Load", "HVDC", "Transformer2", "Transformer3"]
    raw = {s: pd.read_excel(file_path, sheet_name=s, header=3) for s in sheet_names}
    for df in raw.values():
        df.columns = [str(c).strip() for c in df.columns]

    g_disagg = cleaning.main_clean(
        raw["Bus"], raw["Line"], raw["Generator"], raw["Load"],
        raw["HVDC"], raw["Transformer2"], raw["Transformer3"]
    )
    edges_df = pd.read_excel("danish_grid_graph_ready.xlsx", sheet_name="edges")

    # ── Simulation results — no header row ────────────────────────────────────
    CSV_PATH  = "results/simulation_20260502_000845.csv"
    COL_NAMES = ["scenario_id", "k", "removed_nodes", "removed_edges",
                 "total_unmet_demand_MW", "total_oversupply_MW", "num_groups"]
    results_df = pd.read_csv(CSV_PATH, header=None, names=COL_NAMES)
    print(f"Loaded {len(results_df):,} rows covering "
          f"{results_df['scenario_id'].nunique()} scenarios.")

    # ── Canonical deduplication (must happen BEFORE splitting) ─────────────────
    # Same removal set in different list orders (e.g. [102,103] vs [103,102])
    # builds the same perturbed graph → keep only first occurrence.
    # Deduping before split prevents the same removal set appearing in both
    # train and test, which would be data leakage.
    def _canon_key(rn, re):
        nodes = sorted(ast.literal_eval(str(rn)))
        edges = sorted(tuple(sorted(e)) for e in ast.literal_eval(str(re)))
        return repr(nodes) + "|" + repr(edges)

    results_df["_key"] = (results_df["scenario_id"] + "||" +
                          results_df.apply(lambda r: _canon_key(r["removed_nodes"],
                                                                 r["removed_edges"]), axis=1))
    results_df = (results_df
                  .drop_duplicates(subset="_key")
                  .drop(columns="_key")
                  .reset_index(drop=True))
    print(f"After canonical dedup: {len(results_df):,} unique samples.")

    # ── Build aggregated scenario graphs ───────────────────────────────────────
    available_ids   = set(results_df["scenario_id"].unique())
    scenario_graphs = {}
    for sid, scenario in SCENARIOS.items():
        if sid not in available_ids:
            continue
        G_sc = apply_scenario_mean(g_disagg, scenario)
        scenario_graphs[sid] = cleaning.aggregate_graph(G_sc, edges_df)

    # ── Stratified 70 / 20 / 10 row split ─────────────────────────────────────
    # Stratify by scenario so every scenario is proportionally represented
    # in all three splits.  Seed=42 for reproducibility.
    rng = np.random.default_rng(42)
    train_rows, val_rows, test_rows = [], [], []
    for _, grp in results_df.groupby("scenario_id"):
        idx     = rng.permutation(len(grp))
        n_train = int(len(grp) * 0.70)
        n_val   = int(len(grp) * 0.20)
        train_rows.append(grp.iloc[idx[:n_train]])
        val_rows.append(grp.iloc[idx[n_train:n_train + n_val]])
        test_rows.append(grp.iloc[idx[n_train + n_val:]])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    val_df   = pd.concat(val_rows).reset_index(drop=True)
    test_df  = pd.concat(test_rows).reset_index(drop=True)
    print(f"\nSplit (stratified by scenario, seed=42):")
    print(f"  train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

    # ── Train ──────────────────────────────────────────────────────────────────
    model, t_mean, t_std = train_gnn(
        scenario_graphs, train_df, val_df=val_df,
        epochs=300, lr=1e-3, hidden=64,
        eval_every=1, patience=10,
        batch_size=32, pos_weight=3.0,
    )

    # ── Test evaluation ────────────────────────────────────────────────────────
    test_samples = _build_sample_dataset(test_df, scenario_graphs)
    test_results = evaluate_test(model, test_samples, t_mean, t_std)
    test_results.to_csv("gnn_test_results.csv", index=False)
    print("Test predictions saved → gnn_test_results.csv")

    # ── Save model ─────────────────────────────────────────────────────────────
    torch.save({"model_state": model.state_dict(),
                "in_feats":    N_NUMERIC + N_SOURCES,
                "t_mean":      t_mean,
                "t_std":       t_std}, "gnn_model.pt")
    print("Model saved → gnn_model.pt")

    # ── k=1 criticality ranking on first available scenario ───────────────────
    first_sid = sorted(scenario_graphs.keys())[0]
    preds     = predict_criticality(model, scenario_graphs[first_sid], t_mean, t_std)
    print(f"\nTop 15 critical elements (k=1) in {first_sid}:")
    print(preds.head(15).to_string(index=False))
    preds.to_excel("gnn_predictions.xlsx", index=False)
    print("k=1 predictions saved → gnn_predictions.xlsx")
