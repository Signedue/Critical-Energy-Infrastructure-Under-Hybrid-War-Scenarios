import os
import Evaluation
import cleaning
import pandas as pd
import cleaning
from scenarios import SCENARIOS, DEFAULT_SIGMAS, sample_scenario, apply_scenario_to_grid
from datetime import datetime
import random



#### Pre steps - Data Loading
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
#########


#  Configuration of our data generation
N_SCENARIOS = 10 # This is how many scenarios the Monte-Carlo should generate
N_GRAPHS_PER_CALL = 100 # This is how often we simulate_k_removals for one scenario generatio
CALLS_PER_K       = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32}
CLOSENESS_FACTOR  = 20.0
OUTPUT_DIR        = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
_RUN_TS  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_CSV  = os.path.join(OUTPUT_DIR, f"simulation_{_RUN_TS}.csv")


keys = list(CALLS_PER_K.keys())
weights = list(CALLS_PER_K.values())
n_errors = 0
n_completed = 0


# This loop will run forecver until stopped manually using CTRL + C
while True:
    draw_k = random.choices(keys, weights=weights, k=1)[0]
    draw_scenario_name = random.choice(list(SCENARIOS))
    scenario_draws = sample_scenario(draw_scenario_name, SCENARIOS, DEFAULT_SIGMAS, n=N_GRAPHS_PER_CALL, seed=None)
    
    # This for loop will generate N_SCENARIOS * N_GRAPHS_PER_CALL rows each time
    for draw in scenario_draws:
        try:
            draw_scenario = apply_scenario_to_grid(base_scenario, draw)
            res = Evaluation.simulate_k_removals(draw_scenario, k=draw_k, n_graphs=N_GRAPHS_PER_CALL, closeness_factor=CLOSENESS_FACTOR)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] #{n_completed+1} | scenario={draw_scenario.name}  k={draw_k}  errors={n_errors}")
            n_errors = 0
            n_completed += 1
            res.to_csv(OUT_CSV, mode='a', header=False, index=False)
        except Exception:
            n_errors +=1
            if n_errors == 1000:
                break
            else:
                continue