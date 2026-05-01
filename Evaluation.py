import networkx as nx
from math_mod import mathematical_model
import pandas as pd
import random
import math

# Thisfunction will run our math model and save the results
def get_optimization_results(G: nx.Graph):
    results = {}
    model, res = mathematical_model(G)

    # annotate results
    results['slack'] = res['total_imbalance_MW']  # TODO Placeholder for actual slack results
    results['total_unmet_demand_MW'] = res['total_unmet_demand_MW']
    results['total_oversupply_MW'] = res['total_oversupply_MW']
    results['num_groups'] = res['num_groups']
    results['node_results'] = res['node_results']
    results['arc_results'] = res['arc_results']
    return results


# will return the results of the math model given which nodes and edges are removed
def simulate_load_shedding(G: nx.Graph, rem_nodes=[], rem_edges=[]):
    G = G.copy()
    for node in rem_nodes:
        G.remove_node(node)
    for edge in rem_edges:
        G.remove_edge(*edge)
    
    # get results from the graph after modifications
    results = get_optimization_results(G)

    return results

# Receives a grah (base scenario) and simulates load shedding for all possible single node and edge removals, returns dict with results for each removal scenario
def simulation_all_single_removals(G: nx.Graph):
    G = G.copy()  # one copy, reused for all trials
    if not isinstance(G, nx.DiGraph):
        dg = nx.DiGraph()
        dg.add_nodes_from(G.nodes(data=True))
        dg.add_edges_from(G.edges(data=True))
        G = dg


    rows = []

    for node in list(G.nodes()):
        n_attrs   = dict(G.nodes[node])
        if isinstance(G, nx.DiGraph):
            adj_edges = ([(u, v, dict(ea)) for u, v, ea in G.out_edges(node, data=True)] +
                         [(u, v, dict(ea)) for u, v, ea in G.in_edges(node, data=True)])
        else:
            adj_edges = [(u, v, dict(ea)) for u, v, ea in G.edges(node, data=True)]
        G.remove_node(node)

        results = get_optimization_results(G)
        rows.append({
            'scenario': G.name,
            'attack': 'node',
            'removals': node,
            'name': n_attrs.get('name', 'unknown'),
            'source': n_attrs.get('source', 'unknown'),
            'total_unmet_demand_MW': results['total_unmet_demand_MW'],
            'total_oversupply_MW': results['total_oversupply_MW'],
            'num_groups': results['num_groups']
        })

        G.add_node(node, **n_attrs)
        for u, v, ea in adj_edges:
            G.add_edge(u, v, **ea)

    for u, v in list(G.edges()):
        e_attrs = dict(G.edges[u, v])
        u_name  = G.nodes[u].get('name', 'unknown')
        v_name  = G.nodes[v].get('name', 'unknown')
        G.remove_edge(u, v)

        results = get_optimization_results(G)
        rows.append({
            'scenario': G.name,
            'attack': 'edge',
            'removals': (u, v),
            'name': u_name + ' - ' + v_name,
            'source': 'N/A',
            'total_unmet_demand_MW': results['total_unmet_demand_MW'],
            'total_oversupply_MW': results['total_oversupply_MW'],
            'num_groups': results['num_groups']
        })

        G.add_edge(u, v, **e_attrs)

    return pd.DataFrame(rows)


def get_k_removals(G, k=3, n_graphs=100, closeness_factor=1.0):
    def _pos(node):
        d = G.nodes[node]
        return (float(d.get('lat', 0.0) or 0.0),
                float(d.get('lon', 0.0) or 0.0))

    def _edge_pos(u, v):
        pu, pv = _pos(u), _pos(v)
        return ((pu[0] + pv[0]) / 2, (pu[1] + pv[1]) / 2)

    def _dist(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # Build candidate pool once: (kind, id, position)
    candidates = (
        [('node', n,      _pos(n))        for n in G.nodes()]
      + [('edge', (u, v), _edge_pos(u, v)) for u, v in G.edges()]
    )

    graphs = []
    for _ in range(n_graphs):
        seed      = random.choice(candidates)
        selected  = [seed]
        sel_pos   = [seed[2]]
        remaining = [c for c in candidates if c is not seed]

        while len(selected) < k and remaining:
            if closeness_factor == 0:
                chosen = random.choice(remaining)
            else:
                weights = [
                    math.exp(-closeness_factor * min(_dist(c[2], sp) for sp in sel_pos))
                    for c in remaining
                ]
                total = sum(weights)
                r, cum = random.uniform(0, total), 0.0
                chosen = remaining[-1]
                for c, w in zip(remaining, weights):
                    cum += w
                    if r <= cum:
                        chosen = c
                        break

            selected.append(chosen)
            sel_pos.append(chosen[2])
            remaining = [c for c in remaining if c is not chosen]

        G_mod = G.copy()
        removed_nodes, removed_edges = [], []

        for kind, cid, _ in selected:
            if kind == 'node':
                if cid in G_mod:
                    removed_nodes.append(cid)
                    G_mod.remove_node(cid)
            else:
                u, v = cid
                if G_mod.has_edge(u, v):
                    removed_edges.append((u, v))
                    G_mod.remove_edge(u, v)

        graphs.append({
            'graph':         G_mod,
            'removed_nodes': removed_nodes,
            'removed_edges': removed_edges,
        })

    return graphs


def simulate_k_removals(G, k=3, n_graphs=500, closeness_factor=1000.0):
    graphs = get_k_removals(G, k, n_graphs, closeness_factor)
    rows = []

    for graph in graphs:
        results = get_optimization_results(graph['graph'])
        rows.append({
            'scenario': G.name,
            'n_attacks': k,
            'removed nodes': graph['removed_nodes'],
            'removed edges': graph['removed_edges'],
            'total_unmet_demand_MW': results['total_unmet_demand_MW'],
            'total_oversupply_MW': results['total_oversupply_MW'],
            'num_groups': results['num_groups']
        })
    
    return pd.DataFrame(rows)



def greedy_sequential_attack(G, n_attacks=6):
    G = G.copy()
    history = []

    for i in range(n_attacks):
        res = simulation_all_single_removals(G)
        res = res.sort_values('total_unmet_demand_MW', ascending=False).reset_index(drop=True)
        history.append(res)

        best = res.iloc[0]
        if best['attack'] == 'node':
            G.remove_node(best['removals'])
        else:
            G.remove_edge(*best['removals'])

        print(f"[Round {i + 1}/{n_attacks}] Removed {best['attack']} "
              f"'{best['name']}' — unmet demand: {best['total_unmet_demand_MW']:.1f} MW")

    return history