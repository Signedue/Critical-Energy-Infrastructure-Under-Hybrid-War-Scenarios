import pandas as pd
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

DIR = Path(__file__).parent 

def to_num(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )


def get_all_nodes(bus):
    bus["Bus Index"] = pd.to_numeric(bus["Bus Index"], errors="coerce")
    bus["lon"] = pd.to_numeric(bus["lon"], errors="coerce")
    bus["lat"] = pd.to_numeric(bus["lat"], errors="coerce")
    bus["Voltage base[kV]"] = to_num(bus["Voltage base[kV]"])

    nodes = bus[[
        "Bus Index",
        "Bus Name",
        "Station Full Name",
        "Location Name",
        "Voltage base[kV]",
        "lon",
        "lat"
    ]].copy()


    nodes = nodes.rename(columns={
        "Bus Index": "bus_index",
        "Bus Name": "bus_name",
        "Station Full Name": "station_full_name",
        "Location Name": "area_code",
        "Voltage base[kV]": "voltage_kv"
    })

    nodes = nodes.dropna(subset=["bus_index"]).drop_duplicates(subset=["bus_index"])
    nodes["bus_index"] = nodes["bus_index"].astype(int)

    # name = station_full_name hvis den findes, ellers bus_name
    nodes["name"] = nodes["station_full_name"].fillna("").astype(str).str.strip()
    nodes.loc[nodes["name"] == "", "name"] = nodes["bus_name"]
    nodes.head()

    # delet these nodes manually, as the are unconnected
    nodes = nodes[~nodes["bus_index"].isin([212, 213])]

    return nodes

# returns file with cleaned generation data and sources
def get_clean_generation(gen, load, hvdc):
    # change datatypes
    gen["Bus Index"] = pd.to_numeric(gen["Bus Index"], errors="coerce")
    gen["Pmin[MW]"] = to_num(gen["Pmin[MW]"])
    gen["Pmax[MW]"] = to_num(gen["Pmax[MW]"])
    gen["Act.P[MW]"] = to_num(gen["Act.P[MW]"])

    # --- classify source type from Generator Name ---
    def classify_generator_source(name):
        if pd.isna(name):
            return "unknown"
        name = str(name).lower()

        if "windoff" in name:
            return "wind_offshore"
        elif "windon" in name:
            return "wind_onshore"
        elif "solar" in name:
            return "solar"
        elif "gas" in name:
            return "gas"
        elif "hydro" in name:
            return "hydro"
        elif "other" in name:
            return "other"
        else:
            return "unknown"

    gen["gen_source"] = gen["Generator Name"].apply(classify_generator_source)
    gen_clean = gen.rename(columns={"Bus Index": "bus_index", "Pmin[MW]": "p_min", "Pmax[MW]": "p_max", "Act.P[MW]": "supply"})
    gen_clean = gen_clean[["bus_index", "p_min", "p_max", "supply", "gen_source"]].copy()
    
    border_flows = get_clean_border_flows(load, hvdc)
    border_flows['p_min'] = -border_flows['p_max']
    border_flows['gen_source'] = 'international'
    gen_clean = pd.concat([gen_clean, border_flows[['bus_index', 'p_min', 'p_max', 'supply', 'gen_source']]], ignore_index=True)

    return gen_clean


def get_clean_load(load, generation_MW):
    # Assumption: Load Index corresponds to Bus Index

    load["Load Index"] = pd.to_numeric(load["Load Index"], errors="coerce")
    load["Act.P[MW]"] = to_num(load["Act.P[MW]"])

    load_agg = (
        load.groupby("Load Index", dropna=False)
        .agg(demand=("Act.P[MW]", "sum"))
        .reset_index()
        .rename(columns={"Load Index": "bus_index"})
    )

    load_agg["bus_index"] = pd.to_numeric(load_agg["bus_index"], errors="coerce")
    load_agg = load_agg.dropna(subset=["bus_index"])
    load_agg["bus_index"] = load_agg["bus_index"].astype(int)

    # delete nodes with bus_index 0-3, as the are international connections, taken account in the next function
    load_agg = load_agg[load_agg["bus_index"] > 3]


    #Adjust demand, so it equals our supply, to simplify our models. IN reality  this is the loss
    demand_adjustment = generation_MW/load_agg["demand"].sum()
    load_agg["demand"] = load_agg["demand"] * demand_adjustment

    return load_agg



def get_clean_border_flows(load, hvdc):

    # First we get the international connections form the load data
    load["Load Index"] = pd.to_numeric(load["Load Index"], errors="coerce")
    load["Act.P[MW]"] = to_num(load["Act.P[MW]"])

    load_agg = (
        load.groupby("Load Index", dropna=False)
        .agg(demand=("Act.P[MW]", "sum"))
        .reset_index()
        .rename(columns={"Load Index": "bus_index"})
    )

    load_agg["bus_index"] = pd.to_numeric(load_agg["bus_index"], errors="coerce")
    load_agg = load_agg.dropna(subset=["bus_index"])
    load_agg["bus_index"] = load_agg["bus_index"].astype(int)

    # for demands taht are negative, and add it to gen as supply, since this is likely a border flow with negative convention, delet row after
    border_flows = load_agg[load_agg["demand"] < 0].copy()
    border_flows["supply"] = -border_flows["demand"]
    border_flows["p_max"] = border_flows["supply"]


    # Now we get the remoaining ones from the hvdc data
    new_rows = []
    for _, row in hvdc.iterrows():
        bus_index = row["Bus Index"]
        flow = row["Act.P[MW]"]
        max = abs(row["Pmax[MW]"])

        if flow < 0:
            new_rows.append({
                "bus_index": bus_index,
                "supply": -flow,
                "p_max": max
            })
        else:
            new_rows.append({
                "bus_index": bus_index,
                "supply": -flow,
                "p_max": max
            })

    border_flows = pd.concat([border_flows, pd.DataFrame(new_rows)], ignore_index=True)
    border_flows.head(26)
    return border_flows



def get_all_edges(line, transformer2, transformer3):
    # Cleaning transformer2
    transformer2["High.V Bus Index"] = pd.to_numeric(transformer2["High.V Bus Index"], errors="coerce")
    transformer2["Low.V Bus Index"] = pd.to_numeric(transformer2["Low.V Bus Index"], errors="coerce")

    t2_edges = transformer2.rename(columns={
        "High.V Bus Index": "node1",
        "Low.V Bus Index": "node2"
    }).copy()

    t2_edges = t2_edges.dropna(subset=["node1", "node2"])
    t2_edges["node1"] = t2_edges["node1"].astype(int)
    t2_edges["node2"] = t2_edges["node2"].astype(int)

    t2_edges["name"] = "transformer2"
    t2_edges["capacity"] = 99999


    # Cleaning transformer3
    transformer3["High.V Bus Index"] = pd.to_numeric(transformer3["High.V Bus Index"], errors="coerce")
    transformer3["Mid.V Bus Index"] = pd.to_numeric(transformer3["Mid.V Bus Index"], errors="coerce")
    transformer3["Low.V Bus Index"] = pd.to_numeric(transformer3["Low.V Bus Index"], errors="coerce")

    t3_list = []

    for _, row in transformer3.iterrows():
        h = row["High.V Bus Index"]
        m = row["Mid.V Bus Index"]
        l = row["Low.V Bus Index"]

        if pd.notna(h) and pd.notna(m):
            t3_list.append((int(h), int(m)))
        if pd.notna(m) and pd.notna(l):
            t3_list.append((int(m), int(l)))

    t3_edges = pd.DataFrame(t3_list, columns=["node1", "node2"])

    t3_edges["name"] = "transformer3"
    t3_edges["capacity"] = 99999

    line["Node 1"] = pd.to_numeric(line["Node 1"], errors="coerce")
    line["Node 2"] = pd.to_numeric(line["Node 2"], errors="coerce")
    line["Nominal Current[kA]"] = to_num(line["Nominal Current[kA]"])
    line["Nominal Voltage[kV]"] = to_num(line["Nominal Voltage[kV]"])

    edges = line.rename(columns={
        "Node 1": "node1",
        "Node 2": "node2",
        "Line name": "name"
    }).copy()

    edges = edges.dropna(subset=["node1", "node2"]).copy()
    edges["node1"] = edges["node1"].astype(int)
    edges["node2"] = edges["node2"].astype(int)

    # Approximate edge capacity in MW from line rating
    # capacity ≈ sqrt(3) * V[kV] * I[kA]
    edges["capacity"] = np.sqrt(3) * edges["Nominal Voltage[kV]"] * edges["Nominal Current[kA]"]

    edges_final = edges[[
        "node1",
        "node2",
        "name",
        "capacity",
        "Area Name",
        "Line type",
        "Nominal Voltage[kV]",
        "Nominal Current[kA]",
        "R1[Ohm]",
        "X1[Ohm]",
        "Length[km]"
    ]].copy()

    # Renname columns to more consistent names
    edges_final = edges_final.rename(columns={
        "Area Name": "area_code",
        "Line type": "line_type",
        "Nominal Voltage[kV]": "voltage_kv",
        "Nominal Current[kA]": "nominal_current_ka",
        "R1[Ohm]": "r_ohm",
        "X1[Ohm]": "x_ohm",
        "Length[km]": "length_km"
    })

    edges_final["edge_type"] = "line"


    # Transformer2
    t2_edges_clean = t2_edges.copy()

    t2_edges_clean["area_code"] = None
    t2_edges_clean["line_type"] = "transformer"
    t2_edges_clean["voltage_kv"] = None
    t2_edges_clean["nominal_current_ka"] = None
    t2_edges_clean["r_ohm"] = None
    t2_edges_clean["x_ohm"] = None
    t2_edges_clean["length_km"] = None
    t2_edges_clean["edge_type"] = "transformer"


    # Transformer3
    t3_edges_clean = t3_edges.copy()

    t3_edges_clean["area_code"] = None
    t3_edges_clean["line_type"] = "transformer"
    t3_edges_clean["voltage_kv"] = None
    t3_edges_clean["nominal_current_ka"] = None
    t3_edges_clean["r_ohm"] = None
    t3_edges_clean["x_ohm"] = None
    t3_edges_clean["length_km"] = None
    t3_edges_clean["edge_type"] = "transformer"


    # Combine all edges
    edges_final = pd.concat([
        edges_final,
        t2_edges_clean,
        t3_edges_clean
    ], ignore_index=True)

    # add mnual entruy for storebaelt to connect DK1 aqnd DK2, since it is missing from the data, but we know it exists and is important for connectivity
    storebelt = {
        "node1": 15,      # HKS_400_S1 (ENDKE, Zealand side)
        "node2": 16,         # FGD_400_S1 (ENDKW, Funen side)
        "name": "Storebelt",
        "capacity": 600,
        "line_type": "Cable",
        "edge_type": "line"
    }

    edges_final = pd.concat([edges_final, pd.DataFrame([storebelt])], ignore_index=True)
    
    return edges_final[['node1', 'node2', 'name', 'capacity', 'area_code', 'line_type', 'voltage_kv', 'length_km', 'edge_type']]



def populate_nodes(nodes, clean_gen, clean_load):
    # Aggregate gen to one row per (bus, source)
    gen_agg = (
        clean_gen.groupby(["bus_index", "gen_source"])
        .agg(supply=("supply", "sum"), p_min=("p_min", "sum"), p_max=("p_max", "sum"))
        .reset_index()
        .rename(columns={"gen_source": "source"})
    )

    # Expand nodes: one row per (node, source); nodes with no gen get one row with NaN
    nodes_expanded = nodes.merge(gen_agg, on="bus_index", how="left")

    # Add demand — split equally across the source rows for each node
    nodes_expanded = nodes_expanded.merge(clean_load[["bus_index", "demand"]], on="bus_index", how="left")
    nodes_expanded["demand"] = nodes_expanded["demand"].fillna(0.0)
    rows_per_node = nodes_expanded.groupby("bus_index")["bus_index"].transform("size")
    nodes_expanded["demand"] = nodes_expanded["demand"] / rows_per_node

    for col in ["supply", "p_min", "p_max"]:
        nodes_expanded[col] = nodes_expanded[col].fillna(0.0)
    nodes_expanded["source"] = nodes_expanded["source"].fillna("")

    return nodes_expanded


def make_disaggregated_graph(nodes_final, edges_final):
    G = nx.Graph()

    # Add one node per (bus, source) row with ID "bus-counter"
    bus_counters = {}
    for _, row in nodes_final.iterrows():
        bus = int(row["bus_index"])
        count = bus_counters.get(bus, 0)
        bus_counters[bus] = count + 1
        G.add_node(
            f"{bus}-{count}",
            bus_index=bus,
            name=row["name"],
            supply=float(row["supply"]),
            demand=float(row["demand"]),
            p_min=float(row["p_min"]),
            p_max=float(row["p_max"]),
            source=row["source"],
            lat=float(row["lat"]) if pd.notna(row.get("lat")) else None,
            lon=float(row["lon"]) if pd.notna(row.get("lon")) else None,
        )

    # Build lookup: bus_index -> list of sub-node IDs
    bus_subnodes = {}
    for node_id, data in G.nodes(data=True):
        bus_subnodes.setdefault(data["bus_index"], []).append(node_id)

    # Internal bus-bar edges: connect all sub-nodes of the same bus with infinite capacity
    for subnodes in bus_subnodes.values():
        for i in range(len(subnodes) - 1):
            G.add_edge(subnodes[i], subnodes[i + 1], name="busbar", capacity=float("inf"))

    # External edges: one edge per physical line, between the -0 nodes of each bus
    for _, row in edges_final.iterrows():
        b1, b2 = int(row["node1"]), int(row["node2"])
        n1, n2 = f"{b1}-0", f"{b2}-0"
        if G.has_node(n1) and G.has_node(n2):
            G.add_edge(
                n1, n2,
                name=row["name"],
                capacity=float(row["capacity"]) if pd.notna(row["capacity"]) else 0.0
            )

    return G


def aggregate_graph(G_disagg, edges_final, dynamic_iternational=False):
    """Collapse a disaggregated graph back to one node per bus for the math model."""
    G = nx.Graph()

    # Group sub-nodes by bus and combine their attributes
    bus_attrs = {}
    for _, data in G_disagg.nodes(data=True):
        bus = data["bus_index"]
        if bus not in bus_attrs:
            bus_attrs[bus] = {"name": data["name"], "supply": 0.0, "demand": 0.0,
                              "p_min": 0.0, "p_max": 0.0, "p_removable": 0.0, 'p_addable':0.0, "sources": [],
                              "lat": None, "lon": None}
        bus_attrs[bus]["supply"] += data["supply"]
        bus_attrs[bus]["demand"] += data["demand"]
        bus_attrs[bus]["p_min"] += data["p_min"]
        bus_attrs[bus]["p_max"] += data["p_max"]
        
        # Make some sources dynamic. Wind can be truned off
        if data['source'] == "wind_offshore" or data['source'] == "wind_onshore":
            bus_attrs[bus]["p_removable"] += data["supply"] - data['p_min']
        
        if data['source'] == "gas":
            bus_attrs[bus]["p_addable"] += data['p_max'] - data['supply']
            bus_attrs[bus]["p_removable"] += data["supply"] - data['p_min']
        
        # hydro can be reduced by "0 %
        if data['source'] == "hydro":
            bus_attrs[bus]["p_addable"] += data['supply'] * 0.2
            bus_attrs[bus]["p_removable"] += data["supply"] * 0.2
        
        # if we make international lines dynamic, we can also add and remove here.
        if data['source'] == "international" and dynamic_iternational:
            bus_attrs[bus]["p_addable"] += data['p_max'] - data['supply']
            bus_attrs[bus]["p_removable"] += data["supply"] - data['p_min']

        if data["source"]:
            bus_attrs[bus]["sources"].append(data["source"])
        if bus_attrs[bus]["lat"] is None and pd.notna(data.get("lat")):
            bus_attrs[bus]["lat"] = data["lat"]
            bus_attrs[bus]["lon"] = data["lon"]
        if bus_attrs[bus]["p_addable"] < 0.0:
            bus_attrs[bus]["p_addable"] = 0.0
        if bus_attrs[bus]["p_removable"] < 0.0:
            bus_attrs[bus]["p_removable"] = 0.0
        
        

    for bus, attrs in bus_attrs.items():
        G.add_node(
            bus,
            name=attrs["name"],
            supply=attrs["supply"],
            demand=attrs["demand"],
            p_min=attrs["p_min"],
            p_max=attrs["p_max"],
            source=", ".join(sorted(set(attrs["sources"]))),
            lat=attrs["lat"],
            lon=attrs["lon"],
            p_addable=attrs["p_addable"],
            p_removable=attrs["p_removable"]
        )

    # Only add edges where both buses still exist in the aggregated graph
    for _, row in edges_final.iterrows():
        b1, b2 = int(row["node1"]), int(row["node2"])
        if G.has_node(b1) and G.has_node(b2):
            G.add_edge(
                b1, b2,
                name=row["name"],
                capacity=float(row["capacity"]) if pd.notna(row["capacity"]) else 0.0
            )

    return G

def print_graph(G, highlighted_nodes=[], highlighted_edges=[]):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    h_nodes = set(highlighted_nodes)
    h_edges = set(tuple(e) for e in highlighted_edges)

    node_colors = ["red" if n in h_nodes else "steelblue" for n in G.nodes()]
    node_sizes  = [80    if n in h_nodes else 10          for n in G.nodes()]
    edge_colors = ["red" if (u, v) in h_edges or (v, u) in h_edges else "gray" for u, v in G.edges()]
    edge_widths = [3.0   if (u, v) in h_edges or (v, u) in h_edges else 0.8    for u, v in G.edges()]

    nx.draw(G, pos, node_color=node_colors, node_size=node_sizes,
            edge_color=edge_colors, width=edge_widths, with_labels=False)
    plt.title("Danish Transmission Network (Topology)")
    plt.show()


def print_graph_coordinates(G, highlighted_nodes=[], highlighted_edges=[]):
    plt.figure(figsize=(12, 10))

    G_plot = G.copy()
    for node, data in list(G_plot.nodes(data=True)):
        if not (pd.notna(data.get("lat")) and pd.notna(data.get("lon"))):
            G_plot.remove_node(node)

    pos = {n: (float(d["lon"]), float(d["lat"])) for n, d in G_plot.nodes(data=True)}
    h_nodes = set(highlighted_nodes)
    h_edges = set(tuple(e) for e in highlighted_edges)

    node_colors, node_sizes = [], []
    for n in G_plot.nodes():
        if n in h_nodes:
            node_colors.append("red");    node_sizes.append(80)
        elif "international" in str(G_plot.nodes[n].get("source", "")):
            node_colors.append("orange"); node_sizes.append(15)
        else:
            node_colors.append("steelblue"); node_sizes.append(15)

    edge_colors = ["red" if (u, v) in h_edges or (v, u) in h_edges else "gray" for u, v in G_plot.edges()]
    edge_widths = [3.0   if (u, v) in h_edges or (v, u) in h_edges else 0.5    for u, v in G_plot.edges()]

    nx.draw(G_plot, pos, node_color=node_colors, node_size=node_sizes,
            edge_color=edge_colors, width=edge_widths, with_labels=False)
    plt.title("Danish Transmission Network (Topology)")
    plt.tight_layout()
    plt.show()


# returns the baseline disaggregated graph
def main_clean(bus, line, gen, load, hvdc, transformer2, transformer3):
    clean_gen = get_clean_generation(gen, load, hvdc)
    generation_MW = clean_gen["supply"].sum()
    clean_load = get_clean_load(load, generation_MW)

    nodes = get_all_nodes(bus)
    nodes_final = populate_nodes(nodes, clean_gen, clean_load)
    edges_final = get_all_edges(line, transformer2, transformer3)

    out_file = "danish_grid_graph_ready.xlsx"
    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        nodes_final.to_excel(writer, sheet_name="nodes", index=False)
        edges_final.to_excel(writer, sheet_name="edges", index=False)

    G = make_disaggregated_graph(nodes_final, edges_final)
    G.name = "Baseline"

    return G