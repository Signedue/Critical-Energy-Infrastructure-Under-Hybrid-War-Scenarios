import os
import random
import multiprocessing as mp
from datetime import datetime

import pandas as pd

import Evaluation
import cleaning
from scenarios import SCENARIOS, apply_scenario_mean


# ── Worker (runs in a subprocess) ─────────────────────────────────────────────
def _worker_process(scenarios_graph_list, keys, weights, n_graphs, closeness_factor,
                    result_queue, stop_event):
    while not stop_event.is_set():
        draw_k        = random.choices(keys, weights=weights, k=1)[0]
        draw_scenario = random.choice(scenarios_graph_list)
        try:
            res = Evaluation.simulate_k_removals(
                draw_scenario, k=draw_k,
                n_graphs=n_graphs, closeness_factor=closeness_factor,
            )
            result_queue.put(('ok', draw_scenario.name, draw_k, res))
        except Exception:
            result_queue.put(('err', draw_scenario.name, draw_k, None))


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    file_path = "publicdataexportv131450706334_with_lon_lat.xlsx"

    bus          = pd.read_excel(file_path, sheet_name="Bus",          header=3)
    line         = pd.read_excel(file_path, sheet_name="Line",         header=3)
    gen          = pd.read_excel(file_path, sheet_name="Generator",    header=3)
    load         = pd.read_excel(file_path, sheet_name="Load",         header=3)
    hvdc         = pd.read_excel(file_path, sheet_name="HVDC",         header=3)
    transformer2 = pd.read_excel(file_path, sheet_name="Transformer2", header=3)
    transformer3 = pd.read_excel(file_path, sheet_name="Transformer3", header=3)

    for df in [bus, line, gen, load, hvdc, transformer2, transformer3]:
        df.columns = [str(c).strip() for c in df.columns]

    g_unaggregated = cleaning.main_clean(bus, line, gen, load, hvdc, transformer2, transformer3)
    edges          = pd.read_excel('danish_grid_graph_ready.xlsx', sheet_name='edges')
    base_scenario  = g_unaggregated

    # ── Balance check ──────────────────────────────────────────────────────────
    BALANCE_TOL_PCT = 1.0

    balance_rows = []
    for sid, scenario in SCENARIOS.items():
        G_sc         = apply_scenario_mean(base_scenario, scenario)
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

    balance_df   = pd.DataFrame(balance_rows)
    balanced_ids = balance_df.loc[balance_df['balanced'],  'scenario_id'].tolist()
    skipped_ids  = balance_df.loc[~balance_df['balanced'], 'scenario_id'].tolist()

    print(f"Balanced ({len(balanced_ids)}): {balanced_ids}")
    if skipped_ids:
        print(f"Skipped  ({len(skipped_ids)}): {skipped_ids}")

    # ── Build aggregated scenario graphs ───────────────────────────────────────
    scenarios_graph_list = []
    for sid in balanced_ids:
        G_sc      = apply_scenario_mean(base_scenario, SCENARIOS[sid])
        G_agg     = cleaning.aggregate_graph(G_sc, edges)
        G_agg.name = sid
        scenarios_graph_list.append(G_agg)

    # ── Configuration ──────────────────────────────────────────────────────────
    N_GRAPHS_PER_CALL = 2000
    CALLS_PER_K       = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32}
    CLOSENESS_FACTOR  = 1000.0
    MAX_WORKERS       = 10
    OUTPUT_DIR        = "results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_CSV = os.path.join(OUTPUT_DIR, f"simulation_{_RUN_TS}.csv")

    keys    = list(CALLS_PER_K.keys())
    weights = list(CALLS_PER_K.values())

    # ── Start worker processes ─────────────────────────────────────────────────
    result_queue = mp.Queue()
    stop_event   = mp.Event()

    processes = [
        mp.Process(
            target=_worker_process,
            args=(scenarios_graph_list, keys, weights,
                  N_GRAPHS_PER_CALL, CLOSENESS_FACTOR,
                  result_queue, stop_event),
            daemon=True,
        )
        for _ in range(MAX_WORKERS)
    ]
    for p in processes:
        p.start()

    # ── Collect results (main process owns CSV writing) ────────────────────────
    n_errors    = 0
    n_completed = 0

    while True:
        status, name, k, res = result_queue.get()
        if status == 'ok':
            n_errors     = 0
            n_completed += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] #{n_completed} | scenario={name}  k={k}  errors={n_errors}")
            res.to_csv(OUT_CSV, mode='a', header=False, index=False)
        else:
            n_errors += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR #{n_errors} | scenario={name}  k={k}")
            if n_errors >= 1000:
                stop_event.set()
                break

    for p in processes:
        p.join(timeout=5)
