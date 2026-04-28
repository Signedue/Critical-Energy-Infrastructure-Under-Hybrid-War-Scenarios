import networkx as nx
from math_mod import mathematical_model
import pandas as pd

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
    # get list of all edges and nodes in the graph
    all_nodes = list(G.nodes())
    all_edges = list(G.edges())
    rows = []
    for node in all_nodes:
        rem_nodes = [node]
        rem_edges = []
        results = simulate_load_shedding(G, rem_nodes, rem_edges)
        rows.append({
            'scenario': G.name,
            'attack': 'node',
            'removals': node,
            'name': G.nodes[node].get('name', 'unknown'),
            'source': G.nodes[node].get('source', 'unknown'),
            'total_unmet_demand_MW': results['total_unmet_demand_MW'],
            'total_oversupply_MW': results['total_oversupply_MW'],
            'num_groups': results['num_groups']
        })
    for edge in all_edges:
        rem_nodes = []
        rem_edges = [edge]
        results = simulate_load_shedding(G, rem_nodes, rem_edges)
        rows.append({
            'scenario': G.name,
            'attack': 'edge',
            'removals': edge,
            'name': G.nodes[edge[0]].get('name', 'unknown') + ' - ' + G.nodes[edge[1]].get('name', 'unknown'),
            'source': 'N/A',
            'total_unmet_demand_MW': results['total_unmet_demand_MW'],
            'total_oversupply_MW': results['total_oversupply_MW'],
            'num_groups': results['num_groups']
        })

        
    return pd.DataFrame(rows)
