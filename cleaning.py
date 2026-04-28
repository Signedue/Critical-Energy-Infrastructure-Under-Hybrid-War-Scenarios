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


def get_clean_load(load):
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

    # for demands taht are negative, and add it to gen as supply, since this is likely a border flow with negative convention, delet row after
    load_agg = load_agg[load_agg["demand"] >= 0]

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



def get_clean_edges(line, transformer2, transformer3):
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

