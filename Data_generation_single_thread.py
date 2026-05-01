import os
import Evaluation
import cleaning
import pandas as pd
import cleaning




file_path = "publicdataexportv131450706334_with_lon_lat.xlsx"

# Loading sheets
bus = pd.read_excel(file_path, sheet_name="Bus", header=3)
line = pd.read_excel(file_path, sheet_name="Line", header=3)
gen = pd.read_excel(file_path, sheet_name="Generator", header=3)
load = pd.read_excel(file_path, sheet_name="Load", header=3)
hvdc = pd.read_excel(file_path, sheet_name="HVDC", header=3)
transformer2 = pd.read_excel(file_path, sheet_name="Transformer2", header=3)
transformer3 = pd.read_excel(file_path, sheet_name="Transformer3", header=3)

for df in [bus, line, gen, load, hvdc, transformer2, transformer3]:
    df.columns = [str(c).strip() for c in df.columns]


# This is our main cleaning function that turns our raw data into an unaggregated graph, that is used for the scenario generation
g_unaggregated = cleaning.main_clean(bus, line, gen, load, hvdc, transformer2, transformer3)
edges = pd.read_excel('danish_grid_graph_ready.xlsx', sheet_name='edges')
nodes = pd.read_excel('danish_grid_graph_ready.xlsx', sheet_name='nodes')
base_scenario = g_unaggregated


from scenarios import SCENARIOS, apply_scenario_mean

# ── Check balance for every scenario ─────────────────────────────────────────
BALANCE_TOL_PCT = 1.0   # allow up to 1 % residual gap

balance_rows = []
for sid, scenario in SCENARIOS.items():
    G_sc = apply_scenario_mean(base_scenario, scenario)
    total_supply = sum(d.get('supply', 0.0) for _, d in G_sc.nodes(data=True))
    total_demand = sum(d.get('demand', 0.0) for _, d in G_sc.nodes(data=True))
    gap          = abs(total_supply - total_demand)
    gap_pct      = gap / max(total_demand, 1.0) * 100
    balance_rows.append({
        'scenario_id'     : sid,
        'label'           : scenario['label'],
        'total_supply_MW' : round(total_supply, 1),
        'total_demand_MW' : round(total_demand, 1),
        'gap_MW'          : round(gap, 1),
        'gap_pct'         : round(gap_pct, 3),
        'balanced'        : gap_pct <= BALANCE_TOL_PCT,
    })

balance_df = pd.DataFrame(balance_rows)


balanced_ids = balance_df.loc[balance_df['balanced'], 'scenario_id'].tolist()
skipped_ids  = balance_df.loc[~balance_df['balanced'], 'scenario_id'].tolist()

print(f"Balanced ({len(balanced_ids)}): {balanced_ids}")
if skipped_ids:
    print(f"Skipped  ({len(skipped_ids)}): {skipped_ids}")

# ── Run single-removal attack simulation for each balanced scenario ──────────
scenarios_graph_list = []

for sid in balanced_ids:
    scenario = SCENARIOS[sid]
    G_sc = apply_scenario_mean(base_scenario, scenario)
    G_agg = cleaning.aggregate_graph(G_sc, edges)
    G_agg.name = sid
    scenarios_graph_list.append(G_agg)

# Evaluation.simulate_k_removals(G, k=3, n_graphs=500, closeness_factor=1000.0):

# =============================================================================
# PARALLEL DATA GENERATION
# k=1..6, calls per k follow 1:2:4:8:16:32 ratio, each call n_graphs=2000
# Every call is individually exception-handled; results checkpointed to CSV.
# =============================================================================


from datetime import datetime
import random

# ── Configuration ──────────────────────────────────────────────────────────────
N_GRAPHS_PER_CALL = 1
CALLS_PER_K       = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32}
CLOSENESS_FACTOR  = 1000.0
MAX_WORKERS       = 10       # tune to available CPU cores / Gurobi license seats
SAVE_INTERVAL     = 20       # flush buffer to CSV after every N completed tasks
OUTPUT_DIR        = "results"
_RUN_TS  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_CSV  = os.path.join(OUTPUT_DIR, f"simulation_{_RUN_TS}.csv")
LOG_FILE = os.path.join(OUTPUT_DIR, f"simulation_{_RUN_TS}.log")

keys = list(CALLS_PER_K.keys())
weights = list(CALLS_PER_K.values())
n_errors = 0
n_completed = 0

while True:
    draw_k = random.choices(keys, weights=weights, k=1)[0]
    draw_scenario = random.choice(scenarios_graph_list)
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] #{n_completed+1} | scenario={draw_scenario.name}  k={draw_k}  errors={n_errors}")
        res = Evaluation.simulate_k_removals(draw_scenario, k=draw_k, n_graphs=N_GRAPHS_PER_CALL, closeness_factor=CLOSENESS_FACTOR)
        n_errors = 0
        n_completed += 1
        res.to_csv(OUT_CSV, mode='a', header=False, index=False)
    except Exception:
        n_errors +=1
        if n_errors == 1000:
            break
        else:
            continue