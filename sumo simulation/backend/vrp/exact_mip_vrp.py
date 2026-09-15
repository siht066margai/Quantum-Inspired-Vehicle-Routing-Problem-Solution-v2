from __future__ import annotations

import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

import scipy.optimize
from scipy.optimize import milp, LinearConstraint, Bounds

from backend.vrp.problem_instance import ProblemInstance, AlgorithmResult, FleetVehicleRoute, VehicleConfig
from backend.vrp.vrp_evaluator import VRPEvaluator


class ExactMIPVRPSolver:
    """
    Exact Mixed Integer Linear Programming (MILP) Solver for Capacitated Vehicle Routing (CVRP).
    Formulated using the classic Miller-Tucker-Zemlin (MTZ) subtour elimination and capacity constraints.
    Solved via scipy.optimize.milp using the exact HiGHS branch-and-cut optimization engine.

    Mathematical Formulation:
    - Sets: Depot {0}, Customers {1..N}. Total nodes V = {0..N}.
    - Decision Variables:
        x_{i, j} in {0, 1}: Binary edge transit indicator from node i to node j (i != j).
        u_i in [d_i, C]: Continuous cumulative demand variable for customer i.
    - Objective:
        min sum_{i in V} sum_{j in V, j != i} c_{i, j} * x_{i, j}
    - Constraints:
        1. Each customer visited exactly once:
           sum_{i != j} x_{i, j} == 1  for all j in {1..N}
        2. Each customer departed exactly once:
           sum_{j != i} x_{i, j} == 1  for all i in {1..N}
        3. At most K vehicles depart depot:
           sum_{j=1..N} x_{0, j} <= K
        4. Depot flow balance:
           sum_{i=1..N} x_{i, 0} - sum_{j=1..N} x_{0, j} == 0
        5. MTZ Subtour Elimination & Capacity bounds:
           u_i - u_j + C * x_{i, j} <= C - d_j  for all i, j in {1..N}, i != j
           d_i <= u_i <= C                      for all i in {1..N}
    """

    def __init__(self, time_limit: float = 30.0, max_exact_customers: int = 10, time_limit_s: Optional[float] = None):
        self.time_limit = time_limit_s if time_limit_s is not None else time_limit
        self.max_exact_customers = max_exact_customers

    def solve(
        self,
        problem: ProblemInstance,
        time_limit: Optional[float] = None,
    ) -> AlgorithmResult:
        started = time.perf_counter()
        t_limit = time_limit or self.time_limit

        origin = problem.origin
        destinations = problem.destinations
        N = len(destinations)
        K = len(problem.vehicles)

        is_valid, msg = problem.is_valid()
        if not is_valid:
            is_infeasible = "infeasible" in msg.lower() or "exceeds" in msg.lower()
            standby_routes = [
                FleetVehicleRoute(
                    vehicle_id=v.vehicle_id,
                    capacity=v.capacity,
                    assigned_orders=[],
                    visit_sequence=[origin, origin],
                    node_path=[origin],
                    edge_path=[],
                    geometry=[],
                    total_cost=0.0,
                    total_travel_time=0.0,
                    total_distance=0.0,
                    load_used=0.0,
                    capacity_utilization_pct=0.0,
                    initial_load=0.0,
                    delivered_load=0.0,
                    remaining_load=0.0,
                    operational_steps=[{
                        "step": 0,
                        "type": "standby",
                        "node": origin,
                        "action": f"Vehicle {v.vehicle_id} on Standby at Depot. ({msg})",
                        "delivered": 0.0,
                        "remaining": 0.0,
                        "cumulative_distance_m": 0.0,
                        "cumulative_time_s": 0.0,
                    }],
                    segment_calculations=[],
                    bottlenecks=[],
                    is_active=False,
                    status="standby",
                    color=v.color,
                )
                for v in problem.vehicles
            ]
            return AlgorithmResult(
                algorithm="Exact MIP Solver (HiGHS MTZ)",
                status="infeasible" if is_infeasible else "error",
                success=False,
                visit_sequence=[origin, origin],
                node_path=[],
                edge_path=[],
                total_cost=0.0,
                total_travel_time=0.0,
                total_distance=0.0,
                average_speed_ms=0.0,
                bottleneck_count=0,
                bottlenecks=[],
                segment_calculations=[],
                geometry=[],
                feasible=False,
                constraint_violations=[msg],
                computation_time_ms=(time.perf_counter() - started) * 1000,
                math_proof={"infeasibility_reason": msg, "pre_check_failed": True},
                fleet_routes=standby_routes,
                error=msg,
            )

        # Handle N=0 trivial scenario
        if N == 0:
            standby_routes = [
                FleetVehicleRoute(
                    vehicle_id=v.vehicle_id,
                    capacity=v.capacity,
                    assigned_orders=[],
                    visit_sequence=[origin, origin],
                    node_path=[origin],
                    edge_path=[],
                    geometry=[],
                    total_cost=0.0,
                    total_travel_time=0.0,
                    total_distance=0.0,
                    load_used=0.0,
                    capacity_utilization_pct=0.0,
                    initial_load=0.0,
                    delivered_load=0.0,
                    remaining_load=0.0,
                    operational_steps=[{
                        "step": 0,
                        "type": "standby",
                        "node": origin,
                        "action": f"Vehicle {v.vehicle_id} on Standby. No customer orders.",
                        "delivered": 0.0,
                        "remaining": 0.0,
                        "cumulative_distance_m": 0.0,
                        "cumulative_time_s": 0.0,
                    }],
                    segment_calculations=[],
                    bottlenecks=[],
                    is_active=False,
                    status="standby",
                    color=v.color,
                )
                for v in problem.vehicles
            ]
            return AlgorithmResult(
                algorithm="Exact MIP Solver (HiGHS MTZ)",
                status="OPTIMAL",
                success=True,
                visit_sequence=[origin, origin],
                node_path=[origin],
                edge_path=[],
                total_cost=0.0,
                total_travel_time=0.0,
                total_distance=0.0,
                average_speed_ms=0.0,
                bottleneck_count=0,
                bottlenecks=[],
                segment_calculations=[],
                geometry=[],
                feasible=True,
                constraint_violations=[],
                computation_time_ms=(time.perf_counter() - started) * 1000,
                math_proof={"status": "OPTIMAL", "optimal_objective": 0.0, "proven_optimal": True},
                fleet_routes=standby_routes,
            )

        # Scale limit check: exact NP-hard MIP scale boundary
        if N > self.max_exact_customers:
            error_msg = (
                f"Exact MIP formulation has N={N} customers ({N*(N+1)} binary variables + {N*(N-1)} MTZ constraints). "
                f"Exact branch-and-cut optimization is restricted to N <= {self.max_exact_customers} to guarantee provable termination within timeout. "
                f"Please use QPSO, ALNS, or HGS for larger instances."
            )
            standby_routes = [
                FleetVehicleRoute(
                    vehicle_id=v.vehicle_id,
                    capacity=v.capacity,
                    assigned_orders=[],
                    visit_sequence=[origin, origin],
                    node_path=[origin],
                    edge_path=[],
                    geometry=[],
                    total_cost=0.0,
                    total_travel_time=0.0,
                    total_distance=0.0,
                    load_used=0.0,
                    capacity_utilization_pct=0.0,
                    initial_load=0.0,
                    delivered_load=0.0,
                    remaining_load=0.0,
                    operational_steps=[],
                    segment_calculations=[],
                    bottlenecks=[],
                    is_active=False,
                    status="standby",
                    color=v.color,
                )
                for v in problem.vehicles
            ]
            return AlgorithmResult(
                algorithm="Exact MIP Solver (HiGHS MTZ)",
                status="UNSUPPORTED_SIZE",
                success=False,
                visit_sequence=[origin, origin],
                node_path=[],
                edge_path=[],
                total_cost=0.0,
                total_travel_time=0.0,
                total_distance=0.0,
                average_speed_ms=0.0,
                bottleneck_count=0,
                bottlenecks=[],
                segment_calculations=[],
                geometry=[],
                feasible=False,
                constraint_violations=[error_msg],
                computation_time_ms=(time.perf_counter() - started) * 1000,
                math_proof={"status": "UNSUPPORTED_SIZE", "max_customers": self.max_exact_customers, "customer_count": N},
                fleet_routes=standby_routes,
                error=error_msg,
            )

        # Extract pairwise costs on G(t)
        nodes = [origin] + destinations
        num_nodes = len(nodes)
        cost_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    cost_matrix[i, j] = 1e7
                else:
                    c, _, _, _, _, _, _, _ = VRPEvaluator.evaluate_leg(problem, nodes[i], nodes[j])
                    cost_matrix[i, j] = c

        # Demands array (index 0 is depot = 0 demand)
        demands = np.zeros(num_nodes)
        for idx, dest in enumerate(destinations):
            demands[idx + 1] = float(problem.customer_demands.get(dest, 1.0))

        # Vehicle capacity (use maximum vehicle capacity for MTZ upper bound)
        cap = max(v.capacity for v in problem.vehicles)

        # Mapping of variable indices
        # 1. Binary edge variables: x_{i, j} for all i != j in 0..N
        edge_vars: Dict[Tuple[int, int], int] = {}
        var_to_edge: Dict[int, Tuple[int, int]] = {}
        var_count = 0

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_vars[(i, j)] = var_count
                    var_to_edge[var_count] = (i, j)
                    var_count += 1

        num_edge_vars = var_count

        # 2. Continuous MTZ load variables: u_i for i in 1..N
        u_vars: Dict[int, int] = {}
        for i in range(1, num_nodes):
            u_vars[i] = var_count
            var_count += 1

        total_vars = var_count

        # Objective vector c
        c_obj = np.zeros(total_vars)
        for (i, j), v_idx in edge_vars.items():
            c_obj[v_idx] = cost_matrix[i, j]

        # Integrality: 1 for edge variables, 0 for continuous MTZ variables
        integrality = np.zeros(total_vars)
        integrality[:num_edge_vars] = 1

        # Bounds
        lb = np.zeros(total_vars)
        ub = np.ones(total_vars)

        for i in range(1, num_nodes):
            u_idx = u_vars[i]
            lb[u_idx] = demands[i]
            ub[u_idx] = cap

        var_bounds = Bounds(lb, ub)

        # Constraint matrices
        constraint_rows = []
        lhs_bounds = []
        rhs_bounds = []

        # Constraint 1: Each customer visited exactly once (sum_{i != j} x_{i, j} == 1 for j in 1..N)
        for j in range(1, num_nodes):
            row = np.zeros(total_vars)
            for i in range(num_nodes):
                if i != j:
                    row[edge_vars[(i, j)]] = 1.0
            constraint_rows.append(row)
            lhs_bounds.append(1.0)
            rhs_bounds.append(1.0)

        # Constraint 2: Each customer departed exactly once (sum_{j != i} x_{i, j} == 1 for i in 1..N)
        for i in range(1, num_nodes):
            row = np.zeros(total_vars)
            for j in range(num_nodes):
                if i != j:
                    row[edge_vars[(i, j)]] = 1.0
            constraint_rows.append(row)
            lhs_bounds.append(1.0)
            rhs_bounds.append(1.0)

        # Constraint 3: At most K vehicles leave depot (sum_{j=1..N} x_{0, j} <= K)
        row_depot_out = np.zeros(total_vars)
        for j in range(1, num_nodes):
            row_depot_out[edge_vars[(0, j)]] = 1.0
        constraint_rows.append(row_depot_out)
        lhs_bounds.append(1.0 if N > 0 else 0.0)  # at least 1 vehicle if customers exist
        rhs_bounds.append(float(K))

        # Constraint 4: Depot flow balance (sum_{i=1..N} x_{i, 0} - sum_{j=1..N} x_{0, j} == 0)
        row_depot_bal = np.zeros(total_vars)
        for i in range(1, num_nodes):
            row_depot_bal[edge_vars[(i, 0)]] = 1.0
        for j in range(1, num_nodes):
            row_depot_bal[edge_vars[(0, j)]] = -1.0
        constraint_rows.append(row_depot_bal)
        lhs_bounds.append(0.0)
        rhs_bounds.append(0.0)

        # Constraint 5: MTZ Subtour Elimination & Capacity:
        # u_i - u_j + C * x_{i, j} <= C - d_j  for all i, j in 1..N, i != j
        for i in range(1, num_nodes):
            for j in range(1, num_nodes):
                if i != j:
                    row_mtz = np.zeros(total_vars)
                    row_mtz[u_vars[i]] = 1.0
                    row_mtz[u_vars[j]] = -1.0
                    row_mtz[edge_vars[(i, j)]] = cap
                    constraint_rows.append(row_mtz)
                    lhs_bounds.append(-np.inf)
                    rhs_bounds.append(cap - demands[j])

        A_matrix = np.array(constraint_rows)
        constraints = LinearConstraint(A_matrix, lhs_bounds, rhs_bounds)

        # Solve via scipy.optimize.milp (HiGHS branch-and-cut)
        try:
            res_milp = milp(
                c=c_obj,
                integrality=integrality,
                bounds=var_bounds,
                constraints=constraints,
                options={"time_limit": t_limit, "disp": False},
            )
        except Exception as e:
            comp_time = (time.perf_counter() - started) * 1000
            return AlgorithmResult(
                algorithm="Exact MIP Solver (HiGHS MTZ)",
                status="ERROR",
                success=False,
                visit_sequence=[origin, origin],
                node_path=[],
                edge_path=[],
                total_cost=0.0,
                total_travel_time=0.0,
                total_distance=0.0,
                average_speed_ms=0.0,
                bottleneck_count=0,
                bottlenecks=[],
                segment_calculations=[],
                geometry=[],
                feasible=False,
                constraint_violations=[f"MIP solver exception: {e}"],
                computation_time_ms=comp_time,
                math_proof={"status": "ERROR", "error": str(e)},
                fleet_routes=[],
                error=str(e),
            )

        elapsed_ms = (time.perf_counter() - started) * 1000

        # Status translation according to HiGHS specifications
        # status 0: Optimal solution found
        # status 1: Iteration / time limit reached
        # status 2: Problem is infeasible
        # status 3: Problem is unbounded
        is_proven_optimal = (res_milp.status == 0)
        status_label = "OPTIMAL" if is_proven_optimal else ("TIME_LIMIT" if res_milp.status == 1 else "INFEASIBLE")

        if not res_milp.success and not (res_milp.status == 1 and res_milp.x is not None):
            return AlgorithmResult(
                algorithm="Exact MIP Solver (HiGHS MTZ)",
                status=status_label,
                success=False,
                visit_sequence=[origin, origin],
                node_path=[],
                edge_path=[],
                total_cost=0.0,
                total_travel_time=0.0,
                total_distance=0.0,
                average_speed_ms=0.0,
                bottleneck_count=0,
                bottlenecks=[],
                segment_calculations=[],
                geometry=[],
                feasible=False,
                constraint_violations=[f"MIP solver failed to find feasible solution: {res_milp.message}"],
                computation_time_ms=elapsed_ms,
                math_proof={"status": status_label, "highs_status": int(res_milp.status), "message": res_milp.message},
                fleet_routes=[
                    FleetVehicleRoute(
                        vehicle_id=v.vehicle_id,
                        capacity=v.capacity,
                        assigned_orders=[],
                        visit_sequence=[origin, origin],
                        node_path=[origin],
                        edge_path=[],
                        geometry=[],
                        total_cost=0.0,
                        total_travel_time=0.0,
                        total_distance=0.0,
                        load_used=0.0,
                        capacity_utilization_pct=0.0,
                        initial_load=0.0,
                        delivered_load=0.0,
                        remaining_load=0.0,
                        operational_steps=[],
                        segment_calculations=[],
                        bottlenecks=[],
                        is_active=False,
                        status="standby",
                        color=v.color,
                    )
                    for v in problem.vehicles
                ],
                error=res_milp.message,
            )

        # Decode active edges from optimal x
        x_sol = res_milp.x
        active_edges: List[Tuple[int, int]] = []
        for (i, j), v_idx in edge_vars.items():
            if x_sol[v_idx] > 0.5:
                active_edges.append((i, j))

        # Reconstruct vehicle cycles starting from depot (node 0)
        depot_departures = [j for (i, j) in active_edges if i == 0]
        vehicle_routes: List[Tuple[VehicleConfig, List[str]]] = []

        used_vehicles = 0
        visited_nodes = set()

        for dep_j in depot_departures:
            if used_vehicles >= K:
                break
            v_cfg = problem.vehicles[used_vehicles]
            curr_node = dep_j
            route_seq = [origin, nodes[curr_node]]
            visited_nodes.add(curr_node)

            while curr_node != 0:
                # Find outgoing edge from curr_node
                next_nodes = [j for (i, j) in active_edges if i == curr_node]
                if not next_nodes:
                    route_seq.append(origin)
                    break
                next_node = next_nodes[0]
                if next_node == 0:
                    route_seq.append(origin)
                    break
                else:
                    route_seq.append(nodes[next_node])
                    visited_nodes.add(next_node)
                    curr_node = next_node

            vehicle_routes.append((v_cfg, route_seq))
            used_vehicles += 1

        # Unused vehicles assigned to Standby
        while used_vehicles < K:
            v_cfg = problem.vehicles[used_vehicles]
            vehicle_routes.append((v_cfg, [origin, origin]))
            used_vehicles += 1

        # Evaluate via VRPEvaluator
        result = VRPEvaluator.evaluate_fleet_routes(
            problem=problem,
            routes=vehicle_routes,
            algorithm_name="Exact MIP Solver (HiGHS MTZ)",
            computation_time_ms=elapsed_ms,
        )

        math_proof = {
            "solver_engine": "HiGHS Branch-and-Cut (scipy.optimize.milp)",
            "formulation": "Miller-Tucker-Zemlin (MTZ) Subtour & Capacity CVRP",
            "solver_status": status_label,
            "proven_optimal": is_proven_optimal,
            "optimal_objective": round(float(res_milp.fun), 2),
            "lower_bound": round(float(res_milp.fun), 2) if is_proven_optimal else None,
            "optimality_gap_pct": 0.0 if is_proven_optimal else None,
            "num_binary_variables": num_edge_vars,
            "num_continuous_variables": len(u_vars),
            "num_constraints": len(constraint_rows),
            "highs_iterations": getattr(res_milp, "mip_node_count", None),
        }

        solver_details = {
            "status": status_label,
            "highs_message": res_milp.message,
            "active_edges_count": len(active_edges),
            "depot_departures_count": len(depot_departures),
            "optimal_value": round(float(res_milp.fun), 4),
        }

        result.status = status_label
        result.success = result.feasible and (is_proven_optimal or res_milp.status == 1)
        if result.math_proof:
            math_proof.update(result.math_proof)
        result.math_proof = math_proof
        result.solver_details = solver_details
        return result
