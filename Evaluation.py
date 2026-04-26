import networkx as nx
from math_mod import mathematical_model
import pandas as pd

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


# will return the results for load shedding given which nodes and edges are removed
def simulate_load_shedding(G: nx.Graph, rem_nodes=[], rem_edges=[]):
    G = G.copy()
    for node in rem_nodes:
        G.remove_node(node)
    for edge in rem_edges:
        G.remove_edge(*edge)
    
    # get results from the graph after modifications
    results = get_optimization_results(G)

    return results

# Receives a grah and simulates load shedding for all possible single node and edge removals, returns dict with results for each scenario
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





def get_toy_graph() -> nx.Graph:
    G = nx.Graph()

    nodes = [
        ("North_Wind",      {"name": "North_Wind",      "P_max": 300, "P_min":  0, "supply": 200, "demand":  20, "source": "windON"}),
        ("Offshore_Wind",   {"name": "Offshore_Wind",   "P_max": 500, "P_min":  0, "supply": 280, "demand":  10, "source": "windOFF"}),
        ("Solar_South",     {"name": "Solar_South",     "P_max": 200, "P_min":  0, "supply": 100, "demand":  30, "source": "solar"}),
        ("Gas_Backup",      {"name": "Gas_Backup",      "P_max": 400, "P_min": 50, "supply": 150, "demand":  40, "source": "gas"}),
        ("Hub_North",       {"name": "Hub_North",       "P_max":  50, "P_min":  0, "supply":  50, "demand":  80, "source": "windON"}),
        ("City_A",          {"name": "City_A",          "P_max":  30, "P_min":  0, "supply":  30, "demand": 350, "source": "solar"}),
        ("City_B",          {"name": "City_B",          "P_max":  20, "P_min":  0, "supply":  20, "demand": 200, "source": "solar"}),
        ("Industrial_Zone", {"name": "Industrial_Zone", "P_max":  10, "P_min":  0, "supply":  10, "demand": 110, "source": "gas"}),
    ]
    G.add_nodes_from(nodes)

    edges = [
        ("North_Wind",    "Hub_North",       {"capacity": 250}),
        ("Offshore_Wind", "Hub_North",       {"capacity": 400}),
        ("Hub_North",     "Gas_Backup",      {"capacity": 300}),
        ("Hub_North",     "City_A",          {"capacity": 350}),
        ("Hub_North",     "City_B",          {"capacity": 200}),
        ("Solar_South",   "City_B",          {"capacity": 150}),
        ("Gas_Backup",    "City_A",          {"capacity": 300}),
        ("City_A",        "City_B",          {"capacity": 200}),
        ("City_A",        "Industrial_Zone", {"capacity": 180}),
        ("City_B",        "Industrial_Zone", {"capacity": 120}),
    ]
    G.add_edges_from(edges)

    return G

