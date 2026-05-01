"""
scenarios.py
------------
Scenario definitions and Monte Carlo sampling machinery for the Danish
grid resilience project. Outputs balanced scenario graphs to be fed into
the attack module.

Usage:
    from scenarios import SCENARIOS, DEFAULT_SIGMAS, SIGMAS
    from scenarios import apply_scenario_mean, sample_scenario
"""

import numpy as np
import networkx as nx
from copy import deepcopy


# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

SCENARIOS = {

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


_FOREIGN_SOURCES = {'sweden', 'norway', 'germany', 'netherlands', 'uk'}

_HVDC_EDGE_KEYS = [
    ('skagerak',   'hvdc_skagerrak_pct'),
    ('skagerrak',  'hvdc_skagerrak_pct'),
    ('kontiskan',  'hvdc_kontiskan_pct'),
    ('kontek',     'hvdc_kontek_pct'),
    ('cobra',      'hvdc_cobra_pct'),
    ('storebelt',  'hvdc_storebelt_pct'),
    ('storebælt',  'hvdc_storebelt_pct'),
    ('viking',     'hvdc_viking_pct'),
]



def _hvdc_factor_for(name, draw):
    """Return the HVDC pct factor for a node/edge name, or None if not HVDC."""
    if not isinstance(name, str):
        return None
    name_l = name.lower()
    for keyword, factor_key in _HVDC_EDGE_KEYS:
        if keyword in name_l:
            return draw.get(factor_key, 1.0)
    return None


def _src_str(source):
    """Normalise a node's source attribute to a lowercase string."""
    if not isinstance(source, str):
        return ''   # float NaN or other non-string → treat as unknown/thermal
    return source.strip().lower()


def _is_foreign(source):
    return _src_str(source) in _FOREIGN_SOURCES


def _is_dispatchable(source):
    """Gas-containing nodes and unknown-source thermal plants are dispatchable slack."""
    src = _src_str(source)
    if src in ('', 'nan'):
        return True   # NaN / unknown source = large thermal/CHP
    return 'gas' in src


def _generation_factor(source, draw):
    """Initial generation scaling factor for a supply node.

    Dispatchable nodes (gas, thermal) get an initial factor too, but the
    rebalancer will override them to enforce supply == demand.
    Returns 1.0 for foreign nodes (HVDC nodes are handled by name lookup).
    """
    src = _src_str(source)
    if src in ('', 'nan'):
        return draw.get('thermal_factor', 1.0)   # unknown source = thermal/CHP
    if src in _FOREIGN_SOURCES:
        return 1.0
    if 'gas' in src:
        return draw.get('gas_factor', draw.get('thermal_factor', 1.0))
    if 'wind_offshore' in src:
        return draw.get('wind_offshore_factor', 1.0)
    if 'wind_onshore' in src:
        return draw.get('wind_onshore_factor', 1.0)
    if 'solar' in src:
        return draw.get('solar_factor', 1.0)
    return 1.0   # hydro / other — unchanged


# ---------------------------------------------------------------------------
# Rebalancer.
# This function was made with the use of Claude Code, a generative AI tool from Anthropic
# ---------------------------------------------------------------------------

def _rebalance(G2):
    """Adjust dispatchable generation so total supply == total demand.

    Uses gas nodes as primary slack, then thermal (NaN-source) nodes.
    Iteratively scales each group proportionally, clipping to [p_min, p_max].
    Nodes that hit a bound are frozen; remaining free nodes absorb the residual.
    Warns if gap cannot be closed.
    """
    def _gap(G):
        s = sum(d.get('supply', 0.0) for _, d in G.nodes(data=True))
        d = sum(d.get('demand', 0.0) for _, d in G.nodes(data=True))
        return d - s   # positive = deficit, negative = surplus

    gap = _gap(G2)
    if abs(gap) < 1.0:
        return

    for src_type in ('gas', 'thermal'):
        free = [(n, d) for n, d in G2.nodes(data=True)
                if d.get('supply', 0.0) > 0
                and _is_dispatchable(d.get('source', ''))
                and (src_type == 'gas') == ('gas' in _src_str(d.get('source', '')))]
        if not free:
            continue

        for _ in range(len(free) + 1):   # at most one node saturates per iteration
            if abs(gap) < 1.0 or not free:
                break

            total = sum(d['supply'] for _, d in free)
            if total <= 0:
                break

            target = total + gap
            newly_frozen = []
            for n, d in free:
                p_min = d.get('p_min', 0.0)
                p_max = d.get('p_max', float('inf'))
                desired = d['supply'] * target / total
                if desired <= p_min:
                    d['supply'] = p_min
                    newly_frozen.append((n, d))
                elif desired >= p_max:
                    d['supply'] = p_max
                    newly_frozen.append((n, d))
                else:
                    d['supply'] = desired

            frozen_ids = {n for n, _ in newly_frozen}
            free = [(n, d) for n, d in free if n not in frozen_ids]
            gap = _gap(G2)

        if abs(gap) < 1.0:
            return

    total_demand = sum(d.get('demand', 0.0) for _, d in G2.nodes(data=True))
    if abs(gap) > max(1.0, 0.01 * total_demand):
        import warnings
        warnings.warn(
            f"_rebalance: residual gap = {gap:.1f} MW "
            f"({gap / total_demand * 100:.1f}% of demand). "
            "Scenario may be structurally infeasible with this grid."
        )


# ---------------------------------------------------------------------------
# Core scenario application
# ---------------------------------------------------------------------------

def apply_scenario_to_grid(G, draw):
    """Return a scenario-adjusted copy of G.

    After all fixed/HVDC scaling, calls _rebalance() to close the
    supply-demand gap using dispatchable (gas/thermal) nodes as slack.
    """
    G2 = G.copy()

    for n, data in G2.nodes(data=True):
        supply_base = data.get('supply', 0.0)
        demand_base = data.get('demand', 0.0)
        name        = data.get('name', '')
        source      = data.get('source', '')

        # HVDC nodes identified by name (e.g. "HVDC STATION SKAGERAK POL 1")
        hvdc_f = _hvdc_factor_for(name, draw)
        if hvdc_f is not None:
            data['supply'] = supply_base * hvdc_f
            data['demand'] = demand_base * hvdc_f
            continue

        # Scale domestic supply
        if supply_base > 0:
            data['supply'] = supply_base * _generation_factor(source, draw)

        # Scale domestic demand (not foreign nodes — their demand is their own)
        if demand_base > 0 and not _is_foreign(source):
            data['demand'] = demand_base * draw.get('demand_factor', 1.0)

    # Scale HVDC edge capacities
    for u, v, data in G2.edges(data=True):
        cap_base = data.get('capacity', 0.0)
        hvdc_f   = _hvdc_factor_for(data.get('name', ''), draw)
        if hvdc_f is not None:
            data['capacity'] = cap_base * hvdc_f

    _rebalance(G2)
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

    # S10 storm day: overhead AC lines derate due to ice/wind load
    # applied AFTER rebalance so the capacity constraint is visible to the attack module
    line_f = scenario.get('line_capacity_factor', 1.0)
    if line_f != 1.0:
        for u, v, data in G2.edges(data=True):
            cap_base = data.get('capacity', 0.0)
            # only derate non-HVDC edges (HVDC are already handled by hvdc_*_pct)
            if cap_base > 0 and _hvdc_factor_for(data.get('name', ''), mean_draw) is None:
                data['capacity'] = cap_base * line_f

    # S13 second strike: remove nodes before attack starts
    # pre_remove_nodes: explicit list (static)
    for node_id in scenario.get('pre_remove_nodes', []):
        if G2.has_node(node_id):
            G2.remove_node(node_id)
            print(f'  removed {node_id} ({scenario.get("label", "")})')
        else:
            print(f'  WARNING: {node_id} not found in graph')

    # pre_remove_top_n: dynamically remove top-N non-generator nodes by betweenness
    # targets pure substations (no generation) so we don't remove supply nodes
    n_remove = scenario.get('pre_remove_top_n', 0)
    if n_remove > 0:
        bc = nx.betweenness_centrality(G2, normalized=True)
        sub_bc = {n: v for n, v in bc.items()
                  if G2.nodes[n].get('supply', 0.0) == 0.0
                  and not _is_foreign(G2.nodes[n].get('source'))}
        top_nodes = sorted(sub_bc, key=sub_bc.get, reverse=True)[:n_remove]
        for node_id in top_nodes:
            name = G2.nodes[node_id].get('name', node_id)
            G2.remove_node(node_id)
            print(f'  pre_remove_top_n: removed {node_id} ({name}, betweenness={bc[node_id]:.4f})')

    return G2


# =============================================================================
# UNCERTAINTY WIDTHS
# =============================================================================

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
