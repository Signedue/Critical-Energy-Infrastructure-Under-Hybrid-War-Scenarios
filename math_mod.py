import networkx as nx
import pyomo.environ as pyo

# graph_object is a NetworkX DiGraph with the following schema:
# Node attributes: name, supply (MW), demand (MW), p_min (MW), p_max (MW), source (bool)
# Edge attributes: name, capacity (MW)
# Supply/demand are treated as fixed constants — this model simulates post-disruption flow.
# All values in Megawatts (MW).

def mathematical_model(graph_object):
    if not isinstance(graph_object, nx.DiGraph):
        graph_object = nx.DiGraph(graph_object)
    model = pyo.ConcreteModel()

    # Deinfe the sets
    nodes = list(graph_object.nodes)
    arcs = list(graph_object.edges)

    model.Nodes = pyo.Set(initialize=nodes)
    model.Arcs = pyo.Set(initialize=arcs, dimen=2)

    # Deinfe the parameters 
    supply_data = {n: graph_object.nodes[n].get("supply", 0.0) for n in nodes}
    demand_data = {n: graph_object.nodes[n].get("demand", 0.0) for n in nodes}
    capacity_data = {(u, v): graph_object.edges[u, v].get("capacity", 0.0) for u, v in arcs}

    model.supply = pyo.Param(model.Nodes, initialize=supply_data)
    model.demand = pyo.Param(model.Nodes, initialize=demand_data)
    model.capacity = pyo.Param(model.Arcs, initialize=capacity_data)

    # Deinfe the decision Variables
    # t[u,v]: signed power flow on arc (u,v); positive = u->v, negative = v->u
    model.t = pyo.Var(model.Arcs, within=pyo.Reals)

    # s_pos[n]: oversupply at node n (excess generation with nowhere to go)
    # s_neg[n]: unmet demand at node n (demand that cannot be served)
    model.s_pos = pyo.Var(model.Nodes, within=pyo.NonNegativeReals)
    model.s_neg = pyo.Var(model.Nodes, within=pyo.NonNegativeReals)

    # Deinfe the Objective function: minimize total imbalance
    model.objective = pyo.Objective(
        expr=sum(model.s_pos[n] + model.s_neg[n] for n in model.Nodes),
        sense=pyo.minimize
    )

    # Deinfe the constraints

    # 1. Flow balance: for each node, generation + inflows - outflows + oversupply - unmet_demand = 0
    def flow_balance_rule(m, n):
        inflow = sum(m.t[u, n] for u in graph_object.predecessors(n))
        outflow = sum(m.t[n, v] for v in graph_object.successors(n))
        return (m.supply[n] - m.demand[n] + inflow - outflow
                - m.s_pos[n] + m.s_neg[n] == 0)

    model.flow_balance = pyo.Constraint(model.Nodes, rule=flow_balance_rule)

    # 2. Arc capacity (bidirectional): -capacity <= t[u,v] <= capacity
    def arc_cap_upper_rule(m, u, v):
        return m.t[u, v] <= m.capacity[u, v]

    def arc_cap_lower_rule(m, u, v):
        return m.t[u, v] >= -m.capacity[u, v]

    model.arc_cap_upper = pyo.Constraint(model.Arcs, rule=arc_cap_upper_rule)
    model.arc_cap_lower = pyo.Constraint(model.Arcs, rule=arc_cap_lower_rule)

    # Solve
    solver = pyo.SolverFactory("glpk")
    result = solver.solve(model, tee=False)

    # Extract results
    status = str(result.solver.termination_condition)

    node_results = {}
    for n in nodes:
        inflow = sum(pyo.value(model.t[u, n]) for u in graph_object.predecessors(n))
        outflow = sum(pyo.value(model.t[n, v]) for v in graph_object.successors(n))
        node_results[n] = {
            "oversupply_MW": pyo.value(model.s_pos[n]),
            "unmet_demand_MW": pyo.value(model.s_neg[n]),
            "net_flow_in_MW": inflow - outflow,
        }

    arc_results = {
        (u, v): {"flow_MW": pyo.value(model.t[u, v])}
        for u, v in arcs
    }

    total_imbalance = sum(
        node_results[n]["oversupply_MW"] + node_results[n]["unmet_demand_MW"]
        for n in nodes
    )

    return model, {
        "status": status,
        "total_imbalance_MW": total_imbalance,
        "node_results": node_results,
        "arc_results": arc_results,
    }