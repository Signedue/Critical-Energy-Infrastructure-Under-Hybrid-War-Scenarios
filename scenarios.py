"""
scenarios.py
------------
All scenario definitions and Monte Carlo sampling machinery for the Danish
grid resilience project. Import this in the main notebook.

Usage:
    from scenarios import SCENARIOS, DEFAULT_SIGMAS
    from scenarios import apply_scenario_mean, sample_scenario, run_mc, eval_grid_balance
"""

import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy


# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

SCENARIO_FALLBACKS = {

    # ── WEATHER & GENERATION SCENARIOS ────────────────────────────────────────

    'S01_high_wind_surplus': {
        'label'               : 'High wind — surplus export',
        'description'         : 'Strong westerly winds. Offshore + onshore wind at ~90% capacity. '
                                'Low demand (weekend night). Grid exporting heavily via all HVDC links. '
                                'Overgeneration risk if any HVDC link fails.',
        'wind_offshore_factor': 0.90,
        'wind_onshore_factor' : 0.85,
        'solar_factor'        : 0.05,
        'thermal_factor'      : 0.20,
        'demand_factor'       : 0.62,
        'hvdc_skagerrak_pct'  : 1.00,
        'hvdc_kontiskan_pct'  : 1.00,
        'hvdc_kontek_pct'     : 1.00,
        'hvdc_cobra_pct'      : 1.00,
        'hvdc_storebelt_pct'  : 1.00,
        'hvdc_viking_pct'     : 1.00,
        'gas_factor'          : 0.1,
        'duration_hours'      : 6,
    },
    'S02_low_wind_import': {
        'label'               : 'Low wind — import dependent',
        'description'         : 'Calm high-pressure system. Wind at ~8% capacity. '
                                'Denmark relying on imports from Norway and Germany. '
                                'Any HVDC cut directly reduces supply adequacy.',
        'wind_offshore_factor': 0.08,
        'wind_onshore_factor' : 0.06,
        'solar_factor'        : 0.25,
        'thermal_factor'      : 0.75,
        'demand_factor'       : 0.88,
        'hvdc_skagerrak_pct'  : 1.00,
        'hvdc_kontiskan_pct'  : 1.00,
        'hvdc_kontek_pct'     : 1.00,
        'hvdc_cobra_pct'      : 1.00,
        'hvdc_storebelt_pct'  : 1.00,
        'hvdc_viking_pct'     : 1.00,
        'gas_factor'          : 0.8,
        'duration_hours'      : 12,
    },
    'S03_high_solar_midday': {
        'label'               : 'High solar — midday duck curve',
        'description'         : 'Clear summer day, solar PV at peak. Low industrial demand midday. '
                                'Classic duck curve — oversupply noon, then evening ramp.',
        'wind_offshore_factor': 0.30,
        'wind_onshore_factor' : 0.25,
        'solar_factor'        : 0.88,
        'thermal_factor'      : 0.25,
        'demand_factor'       : 0.65,
        'hvdc_skagerrak_pct'  : 1.00,
        'hvdc_kontiskan_pct'  : 1.00,
        'hvdc_kontek_pct'     : 1.00,
        'hvdc_cobra_pct'      : 1.00,
        'hvdc_storebelt_pct'  : 1.00,
        'hvdc_viking_pct'     : 1.00,
        'gas_factor'          : 0.2,
        'duration_hours'      : 3,
    },
    'S04_low_renewables_dark': {
        'label'               : 'Dark winter night — low renewables',
        'description'         : 'December, 03:00. No solar. Low wind. Thermal generation '
                                'carrying almost all load. Maximum reliance on CHP plants.',
        'wind_offshore_factor': 0.12,
        'wind_onshore_factor' : 0.10,
        'solar_factor'        : 0.00,
        'thermal_factor'      : 0.85,
        'demand_factor'       : 0.74,
        'hvdc_skagerrak_pct'  : 1.00,
        'hvdc_kontiskan_pct'  : 1.00,
        'hvdc_kontek_pct'     : 1.00,
        'hvdc_cobra_pct'      : 1.00,
        'hvdc_storebelt_pct'  : 1.00,
        'hvdc_viking_pct'     : 1.00,
        'gas_factor'          : 0.85,
        'duration_hours'      : 6,
    },
    'S05_winter_peak_demand': {
        'label'               : 'Winter peak demand',
        'description'         : 'Cold January morning, everyone heating up. '
                                'Highest national consumption of the year (~6 200 MW). '
                                'Tightest supply margin — any generation loss is critical.',
        'wind_offshore_factor': 0.45,
        'wind_onshore_factor' : 0.40,
        'solar_factor'        : 0.00,
        'thermal_factor'      : 0.90,
        'demand_factor'       : 1.00,
        'hvdc_skagerrak_pct'  : 1.00,
        'hvdc_kontiskan_pct'  : 1.00,
        'hvdc_kontek_pct'     : 1.00,
        'hvdc_cobra_pct'      : 1.00,
        'hvdc_storebelt_pct'  : 1.00,
        'hvdc_viking_pct'     : 1.00,
        'gas_factor'          : 0.9,
        'duration_hours'      : 3,
    },
    'S06_summer_trough': {
        'label'               : 'Summer demand trough',
        'description'         : 'August, 03:00, Sunday. Lowest national consumption (~2 500 MW). '
                                'Even modest wind output creates large surplus. '
                                'Grid most prone to overgeneration-triggered instability.',
        'wind_offshore_factor': 0.40,
        'wind_onshore_factor' : 0.35,
        'solar_factor'        : 0.00,
        'thermal_factor'      : 0.15,
        'demand_factor'       : 0.40,
        'hvdc_skagerrak_pct'  : 1.00,
        'hvdc_kontiskan_pct'  : 1.00,
        'hvdc_kontek_pct'     : 1.00,
        'hvdc_cobra_pct'      : 1.00,
        'hvdc_storebelt_pct'  : 1.00,
        'hvdc_viking_pct'     : 1.00,
        'gas_factor'          : 0.08,
        'duration_hours'      : 6,
    },

    # ── INTERCONNECTOR / GEOPOLITICAL SCENARIOS ────────────────────────────────

    'S07_nordic_drought': {
        'label'               : 'Nordic drought — Norway hydro unavailable',
        'description'         : 'Dry year, Norwegian hydro reservoirs critically low. '
                                'Skagerrak export to Norway suspended — Norway is itself importing. '
                                'Denmark loses its largest flexible balancing partner.',
        'wind_offshore_factor': 0.50,
        'wind_onshore_factor' : 0.45,
        'solar_factor'        : 0.10,
        'thermal_factor'      : 0.70,
        'demand_factor'       : 0.88,
        'hvdc_skagerrak_pct'  : 0.00,
        'hvdc_kontiskan_pct'  : 0.30,
        'hvdc_kontek_pct'     : 1.00,
        'hvdc_cobra_pct'      : 1.00,
        'hvdc_storebelt_pct'  : 1.00,
        'hvdc_viking_pct'     : 1.00,
        'gas_factor'          : 0.75,
        'duration_hours'      : 24,
    },
    'S08_german_grid_stress': {
        'label'               : 'German grid stress — limited export south',
        'description'         : 'Germany experiencing its own supply shortage (e.g. Dunkelflaute). '
                                'Kontek cable export restricted; Germany also unable to supply Denmark. '
                                'Northern European grid tighter overall.',
        'wind_offshore_factor': 0.20,
        'wind_onshore_factor' : 0.18,
        'solar_factor'        : 0.05,
        'thermal_factor'      : 0.80,
        'demand_factor'       : 0.95,
        'hvdc_skagerrak_pct'  : 1.00,
        'hvdc_kontiskan_pct'  : 1.00,
        'hvdc_kontek_pct'     : 0.10,
        'hvdc_cobra_pct'      : 1.00,
        'hvdc_storebelt_pct'  : 1.00,
        'hvdc_viking_pct'     : 1.00,
        'gas_factor'          : 0.8,
        'duration_hours'      : 12,
    },
    'S09_cobra_offline': {
        'label'               : 'COBRA cable offline (planned maintenance)',
        'description'         : 'Netherlands interconnector out of service. '
                                'Grid enters attack simulation already missing 700 MW of export capacity. '
                                'High-wind day makes overgeneration likely if any further link is cut.',
        'wind_offshore_factor': 0.75,
        'wind_onshore_factor' : 0.70,
        'solar_factor'        : 0.15,
        'thermal_factor'      : 0.30,
        'demand_factor'       : 0.72,
        'hvdc_skagerrak_pct'  : 1.00,
        'hvdc_kontiskan_pct'  : 1.00,
        'hvdc_kontek_pct'     : 1.00,
        'hvdc_cobra_pct'      : 0.00,
        'hvdc_storebelt_pct'  : 1.00,
        'hvdc_viking_pct'     : 1.00,
        'gas_factor'          : 0.25,
        'duration_hours'      : 48,
    },

    # ── STRESS / COMBINED SCENARIOS ────────────────────────────────────────────

    'S10_storm_day': {
        'label'               : 'Storm day — high wind, derated lines',
        'description'         : 'Gale-force winds. Wind generation at ~95% capacity. '
                                'But overhead transmission lines derated 20% (ice/wind load risk). '
                                'Maximum generation, reduced transmission capacity simultaneously.',
        'wind_offshore_factor': 0.95,
        'wind_onshore_factor' : 0.90,
        'solar_factor'        : 0.00,
        'thermal_factor'      : 0.20,
        'demand_factor'       : 0.80,
        'line_capacity_factor': 0.80,   # overhead lines derated 20%
        'hvdc_skagerrak_pct'  : 0.90,
        'hvdc_kontiskan_pct'  : 0.90,
        'hvdc_kontek_pct'     : 1.00,
        'hvdc_cobra_pct'      : 1.00,
        'hvdc_storebelt_pct'  : 1.00,
        'hvdc_viking_pct'     : 1.00,
        'gas_factor'          : 0.15,
        'duration_hours'      : 6,
    },
    'S11_storebelt_degraded': {
        'label'               : 'Storebælt at 50% — DK1/DK2 weakly coupled',
        'description'         : 'The internal HVDC spine between Western and Eastern Denmark '
                                'operating at half capacity (one pole offline, technical fault). '
                                'DK1 and DK2 become semi-autonomous — tests zone isolation vulnerability.',
        'wind_offshore_factor': 0.55,
        'wind_onshore_factor' : 0.50,
        'solar_factor'        : 0.20,
        'thermal_factor'      : 0.65,
        'demand_factor'       : 0.90,
        'hvdc_skagerrak_pct'  : 1.00,
        'hvdc_kontiskan_pct'  : 1.00,
        'hvdc_kontek_pct'     : 1.00,
        'hvdc_cobra_pct'      : 1.00,
        'hvdc_storebelt_pct'  : 0.50,
        'hvdc_viking_pct'     : 1.00,
        'gas_factor'          : 0.65,
        'duration_hours'      : 24,
    },
    'S12_gas_shortage': {
        'label'               : 'Gas supply disruption — thermal constrained',
        'description'         : 'Gas supply cut or price spike forces thermal CHP plants '
                                'to operate at reduced output. Renewables must carry more load. '
                                'Mirrors 2022-style energy crisis conditions.',
        'wind_offshore_factor': 0.55,
        'wind_onshore_factor' : 0.50,
        'solar_factor'        : 0.10,
        'thermal_factor'      : 0.35,
        'demand_factor'       : 0.92,
        'hvdc_skagerrak_pct'  : 1.00,
        'hvdc_kontiskan_pct'  : 1.00,
        'hvdc_kontek_pct'     : 0.60,
        'hvdc_cobra_pct'      : 1.00,
        'hvdc_storebelt_pct'  : 1.00,
        'hvdc_viking_pct'     : 1.00,
        'gas_factor'          : 0.2,
        'duration_hours'      : 48,
    },
    'S13_second_strike': {
        'label'               : 'Second strike — grid already degraded',
        'description'         : 'A major substation (Endrup 400kV, the highest-betweenness node) '
                                'is already offline from a prior incident. Grid enters the attack '
                                'simulation pre-weakened. Tests residual resilience.',
        'wind_offshore_factor': 0.55,
        'wind_onshore_factor' : 0.50,
        'solar_factor'        : 0.15,
        'thermal_factor'      : 0.70,
        'demand_factor'       : 0.88,
        'hvdc_skagerrak_pct'  : 1.00,
        'hvdc_kontiskan_pct'  : 1.00,
        'hvdc_kontek_pct'     : 1.00,
        'hvdc_cobra_pct'      : 1.00,
        'hvdc_storebelt_pct'  : 1.00,
        'hvdc_viking_pct'     : 1.00,
        'pre_remove_top_n'    : 1,   # dynamically remove highest-betweenness substation
        'gas_factor'          : 0.70,
        'duration_hours'      : 4,
    },
}

SCENARIOS = SCENARIO_FALLBACKS

# =============================================================================
# MC SAMPLER
# =============================================================================


from scipy.stats import norm

# which factor keys we sample over
FACTOR_KEYS = [
    'wind_offshore_factor', 'wind_onshore_factor', 'solar_factor',
    'thermal_factor', 'demand_factor',
    'hvdc_skagerrak_pct', 'hvdc_kontiskan_pct', 'hvdc_kontek_pct',
    'hvdc_cobra_pct', 'hvdc_storebelt_pct', 'hvdc_viking_pct',
    'gas_factor',
]

# correlation groups - variables in the same group share a latent "weather shock"
CORRELATION_GROUPS = {
    "wind"   : ["wind_offshore_factor", "wind_onshore_factor"],  # move together
    "system" : ["demand_factor", "thermal_factor"],              # cold = high demand + high thermal
    "hvdc"   : ["hvdc_skagerrak_pct", "hvdc_kontiskan_pct",
                "hvdc_kontek_pct", "hvdc_cobra_pct"],
}
CORR_STRENGTH = {"wind": 0.8, "system": 0.6, "hvdc": 0.5}

# slightly higher outage prob for storm scenarios
HVDC_OUTAGE_PROB = {'default': 0.02, 'storm': 0.10, 'cyber': 0.15, 'stress': 0.05}


def _beta_params(mean, sigma):
    # convert mean + std to Beta(a,b) - Beta is bounded [0,1] which is
    # perfect for capacity factors
    if sigma <= 0 or mean <= 0 or mean >= 1:
        return None
    mean = np.clip(mean, 0.01, 0.99)
    var  = min(sigma**2, mean * (1 - mean) * 0.9)
    d    = mean * (1 - mean) / var - 1
    return mean * d, (1 - mean) * d

def _sample_beta(rng, mean, sigma):
    p = _beta_params(mean, sigma)
    return float(rng.beta(*p)) if p else float(mean)

def _generate_latent(rng, keys):
    # gaussian copula idea: each group shares a "weather shock"
    # then each variable gets its own idiosyncratic noise on top
    latent = {}
    for group, group_keys in CORRELATION_GROUPS.items():
        z_shared = rng.normal()
        s = CORR_STRENGTH[group]
        for key in group_keys:
            if key in keys:
                latent[key] = s * z_shared + np.sqrt(1 - s**2) * rng.normal()
    for key in keys:
        if key not in latent:
            latent[key] = rng.normal()
    return latent

def _hvdc_outage_prob(scenario_id):
    if 'storm' in scenario_id:  return HVDC_OUTAGE_PROB['storm']
    if 'cyber' in scenario_id:  return HVDC_OUTAGE_PROB['cyber']
    if 'stress' in scenario_id: return HVDC_OUTAGE_PROB['stress']
    return HVDC_OUTAGE_PROB['default']

def _apply_tail_risk(rng, value, prob=0.1):
    # 10% chance of a really bad draw - adds fat tails to the distribution
    return value * rng.uniform(0.5, 0.8) if rng.random() < prob else value


def apply_scenario_to_grid(G, draw):
    # takes a factor dict and returns a modified copy of the graph
    # with supply/demand/capacity scaled accordingly
    # note: gas nodes are left at 100% - no gas_factor in current scenarios
    # (TODO: add gas_factor for S12 gas shortage if needed later)
    G2 = G.copy()

    for n, data in G2.nodes(data=True):
        src = data.get("source", "")
        if   "wind_offshore" in src: f = draw.get("wind_offshore_factor", 1)
        elif "wind_onshore"  in src: f = draw.get("wind_onshore_factor",  1)
        elif "solar"         in src: f = draw.get("solar_factor",         1)
        elif "thermal"       in src: f = draw.get("thermal_factor",       1)
        elif "gas"           in src:
            # use gas_factor if explicitly set, otherwise fall back to thermal_factor
            # this matters most for S12 where gas is severely constrained
            f = draw.get("gas_factor", draw.get("thermal_factor", 1))
        else:                        f = 1  # hydro / substation unchanged

        if "p_max_base" in data:
            p_max = data["p_max_base"]
            data["supply"] = max(data.get("p_min", 0), min(p_max * f, p_max))

        if "demand_base" in data:
            data["demand"] = data["demand_base"] * draw.get("demand_factor", 1)

    for u, v, data in G2.edges(data=True):
        name = data.get("name", "").lower()
        if   "kontek"    in name: f = draw.get("hvdc_kontek_pct",    1)
        elif "cobra"     in name: f = draw.get("hvdc_cobra_pct",     1)
        elif "storebelt" in name or "storebælt" in name:
                                  f = draw.get("hvdc_storebelt_pct", 1)
        elif "skagerrak" in name: f = draw.get("hvdc_skagerrak_pct", 1)
        elif "kontiskan" in name: f = draw.get("hvdc_kontiskan_pct", 1)
        elif "viking"    in name: f = draw.get("hvdc_viking_pct",    1)
        else:                     f = 1
        if "capacity_base" in data:
            data["capacity"] = data["capacity_base"] * f

    return G2


def sample_scenario(scenario_id, scenarios, sigmas, n=500, seed=42):
    # draw n stochastic realisations of a scenario
    rng  = np.random.default_rng(seed)
    base = scenarios[scenario_id]
    sigma_map = sigmas.get(scenario_id, {})
    draws = []

    for i in range(n):
        draw = deepcopy(base)
        draw["_mc_trial"] = i
        latent = _generate_latent(rng, FACTOR_KEYS)

        for key in FACTOR_KEYS:
            if key not in base: continue
            mean  = base[key]
            sigma = sigma_map.get(key, 0.0)

            if sigma == 0 or mean == 0:
                draw[key] = mean
                continue

            if key.startswith("hvdc"):
                # HVDC is either fully out (rare) or running with some derating
                val = (0.0 if rng.random() < _hvdc_outage_prob(scenario_id)
                       else _sample_beta(rng, mean, sigma))
            else:
                val = _sample_beta(rng, mean, sigma)

            draw[key] = _apply_tail_risk(rng, val)

        draws.append(draw)
    return draws


def run_mc(draws, G, eval_func):
    # run each draw through the eval function and collect results
    records = []
    for draw in draws:
        G_state = apply_scenario_to_grid(G, draw)
        metrics = eval_func(G_state)
        metrics["_mc_trial"] = draw["_mc_trial"]
        records.append(metrics)
    return pd.DataFrame(records)




# =============================================================================
# SCENARIO → GRAPH
# =============================================================================

# keys we actually pass to apply_scenario_to_grid
_FACTOR_KEYS = [
    'wind_offshore_factor', 'wind_onshore_factor', 'solar_factor',
    'thermal_factor', 'demand_factor',
    'hvdc_skagerrak_pct', 'hvdc_kontiskan_pct', 'hvdc_kontek_pct',
    'hvdc_cobra_pct', 'hvdc_storebelt_pct', 'hvdc_viking_pct',
    'gas_factor',
]


def apply_scenario_mean(G, scenario):
    # builds a "mean draw" using the scenario's point estimates and runs it
    # through the same function the MC uses - so the representative graph
    # is guaranteed to match what the MC draws look like on average
    mean_draw = {k: scenario.get(k, 1.0) for k in _FACTOR_KEYS}
    mean_draw['_mc_trial'] = -1
    G2 = apply_scenario_to_grid(G, mean_draw)

    # S10 storm day: overhead lines derate 20% due to ice/wind load
    line_f = scenario.get('line_capacity_factor', 1.0)
    if line_f != 1.0:
        for u, v, data in G2.edges(data=True):
            if data.get('edge_type') == 'transmission' and data.get('capacity_base', 0) > 0:
                data['capacity'] = data['capacity_base'] * line_f

    # S13 second strike: remove nodes before attack starts
    # pre_remove_nodes: explicit list (static)
    for node_id in scenario.get('pre_remove_nodes', []):
        if G2.has_node(node_id):
            G2.remove_node(node_id)
            print(f'  removed {node_id} ({scenario.get("label", "")})')
        else:
            print(f'  WARNING: {node_id} not found in graph')

    # pre_remove_top_n: dynamically remove top-N substations by betweenness
    # this way S13 always targets the most critical node regardless of graph version
    n_remove = scenario.get('pre_remove_top_n', 0)
    if n_remove > 0:
        bc = nx.betweenness_centrality(G2, normalized=True)
        # only consider substations - generators aren't standalone attack targets
        sub_bc = {n: v for n, v in bc.items()
                  if G2.nodes[n].get('node_type') == 'substation'}
        top_nodes = sorted(sub_bc, key=sub_bc.get, reverse=True)[:n_remove]
        for node_id in top_nodes:
            name = G2.nodes[node_id].get('name', node_id)
            G2.remove_node(node_id)
            print(f'  pre_remove_top_n: removed {node_id} ({name}, betweenness={bc[node_id]:.4f})')

    return G2


# build one representative graph per scenario


# =============================================================================
# GRID EVALUATION + UNCERTAINTY WIDTHS
# =============================================================================

def eval_grid_balance(G_state):
    # compute national + zonal supply/demand for one MC draw
    supply = dk1_s = dk2_s = 0.0
    demand = dk1_d = dk2_d = 0.0

    for _, d in G_state.nodes(data=True):
        ntype = d.get('node_type', '')
        area  = d.get('area', '')

        if ntype == 'generator':
            s = d.get('supply', 0.0)
            supply += s
            if area == 'DK1':   dk1_s += s
            elif area == 'DK2': dk2_s += s

        elif ntype == 'substation':
            dem = d.get('demand', 0.0)
            demand += dem
            if area == 'DK1':   dk1_d += dem
            elif area == 'DK2': dk2_d += dem

    bal = supply - demand
    return {
        'supply_mw'     : supply,
        'demand_mw'     : demand,
        'balance_mw'    : bal,
        'deficit_mw'    : max(0.0, -bal),   # how short are we?
        'surplus_mw'    : max(0.0,  bal),   # how much are we overproducing?
        'dk1_balance_mw': dk1_s - dk1_d,
        'dk2_balance_mw': dk2_s - dk2_d,
        'dk1_supply_mw' : dk1_s,
        'dk1_demand_mw' : dk1_d,
        'dk2_supply_mw' : dk2_s,
        'dk2_demand_mw' : dk2_d,
    }


# uncertainty widths for each factor - same as NB2
DEFAULT_SIGMAS = {
    'wind_offshore_factor': 0.12,
    'wind_onshore_factor' : 0.10,
    'solar_factor'        : 0.15,  # solar is quite variable
    'thermal_factor'      : 0.08,
    'demand_factor'       : 0.05,
    'hvdc_skagerrak_pct'  : 0.05,
    'hvdc_kontiskan_pct'  : 0.05,
    'hvdc_kontek_pct'     : 0.05,
    'hvdc_cobra_pct'      : 0.05,
    'hvdc_storebelt_pct'  : 0.05,
    'hvdc_viking_pct'     : 0.05,
}

# zero sigma for anything already at 0 - no point sampling around zero
SIGMAS = {
    sid: {k: (0.0 if SCENARIOS[sid].get(k, 1.0) == 0.0 else DEFAULT_SIGMAS.get(k, 0.0))
          for k in DEFAULT_SIGMAS}
    for sid in SCENARIOS
}
