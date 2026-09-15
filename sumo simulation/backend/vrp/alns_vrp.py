from __future__ import annotations

import time
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from backend.vrp.problem_instance import ProblemInstance, AlgorithmResult, VehicleConfig, FleetVehicleRoute
from backend.vrp.vrp_evaluator import VRPEvaluator


class ALNSVRPSolver:
    """
    Classical Adaptive Large Neighborhood Search (ALNS) Metaheuristic Solver for Dynamic Fleet CVRP.
    Formulated strictly from scratch based on Classical ALNS principles (Ropke & Pisinger 2006)
    and methodological concepts in Paper 1 (adaptive operator selection, destroy/repair heuristics,
    multi-vehicle fleet routing, roulette-wheel selection, and research convergence tracking).

    Key Algorithmic Mechanics:
    1. Multi-Vehicle Fleet Representation:
       Directly maintains vehicle routes S = [R_1, R_2, ..., R_K], where each R_k is a customer
       sequence [c_1, ..., c_m] assigned to vehicle k respecting individual capacity C_k.
    2. Initial Solution Construction:
       Greedy cheapest-insertion heuristic across fleet vehicles using dynamic SUMO graph costs W(t).
    3. Destroy Operators:
       - Random Removal: removes q customers chosen uniformly at random.
       - Worst Removal: calculates exact marginal cost contribution Delta(c) = cost(R_k) - cost(R_k \\ {c}),
         and removes high-cost customers using Shaw randomized greediness (p=3).
       - Shaw / Relatedness Removal: removes customers based on dynamic graph travel time distance and demand difference.
    4. Repair Operators:
       - Random Repair: inserts unserved customers into random feasible positions across vehicles.
       - Greedy Repair: evaluates exact insertion cost Delta f(c, k, p) for all unserved customers across
         all feasible vehicle positions and inserts the minimum-cost customer.
       - Regret-2 / Regret-3 Repair: computes regret r(c) = sum_{j=2}^k (c_j(c) - c_1(c)) across alternative
         cheapest vehicle positions and prioritizes insertion of the customer with maximum regret.
    5. Adaptive Operator Weights & Roulette-Wheel Selection:
       Separate weight vectors w_d and w_r updated periodically (epoch = 10) with reaction factor lambda = 0.8
       and score bonuses: sigma_1 = 33 (global best), sigma_2 = 9 (improving), sigma_3 = 13 (accepted).
    6. Simulated Annealing Acceptance Criterion:
       P(accept) = exp(-(f(S') - f(S)) / T) with geometric cooling T <- T * 0.97.
    """

    def __init__(
        self,
        max_iter: int = 60,
        decay_factor: float = 0.8,
        seed: int = 42,
        time_limit_s: float = 15.0,
    ):
        self.max_iter = max_iter
        self.decay_factor = decay_factor
        self.seed = seed
        self.time_limit_s = time_limit_s

    def solve(
        self,
        problem: ProblemInstance,
        max_iter: Optional[int] = None,
        time_limit_s: Optional[float] = None,
    ) -> AlgorithmResult:
        started = time.perf_counter()
        t_limit = time_limit_s or self.time_limit_s
        T_max = max_iter or self.max_iter

        # 1. Problem Instance Validation
        is_valid, msg = problem.is_valid()
        if not is_valid:
            is_infeasible = "infeasible" in msg.lower() or "exceeds" in msg.lower() or "capacity" in msg.lower()
            standby_routes = [
                FleetVehicleRoute(
                    vehicle_id=v.vehicle_id,
                    capacity=v.capacity,
                    assigned_orders=[],
                    visit_sequence=[problem.origin, problem.origin],
                    node_path=[problem.origin],
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
                        "node": problem.origin,
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
                algorithm="Classical ALNS (From Scratch)",
                status="infeasible" if is_infeasible else "validation_error",
                success=False,
                visit_sequence=[problem.origin, problem.origin],
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

        origin = problem.origin
        destinations = problem.destinations[:]
        N = len(destinations)
        vehicles = [
            v for v in problem.vehicles
            if getattr(v, "is_available", True) and getattr(v, "status", "available") != "unavailable"
        ]
        K = len(vehicles)

        if N == 0:
            return VRPEvaluator.evaluate_sequence(problem, [origin, origin], algorithm_name="Classical ALNS (From Scratch)")

        rng = random.Random(self.seed)

        # Precompute & cache pairwise costs on dynamic graph G(t)
        cost_cache: Dict[Tuple[str, str], float] = {}

        def get_cost(u: str, v: str) -> float:
            pair = (u, v)
            if pair not in cost_cache:
                try:
                    c, _, _, _, _, _, _, _ = VRPEvaluator.evaluate_leg(problem, u, v)
                    cost_cache[pair] = c
                except Exception:
                    cost_cache[pair] = 1e6
            return cost_cache[pair]

        def route_cost(r: List[str]) -> float:
            if not r:
                return 0.0
            total = get_cost(origin, r[0]) + get_cost(r[-1], origin)
            for idx in range(len(r) - 1):
                total += get_cost(r[idx], r[idx + 1])
            return total

        def solution_cost(routes: List[List[str]]) -> float:
            return sum(route_cost(r) for r in routes)

        # 2. Build Initial Feasible Multi-Vehicle Solution
        current_routes = self._build_initial_solution(destinations, vehicles, problem, get_cost, rng)
        current_cost = solution_cost(current_routes)

        best_routes = [r[:] for r in current_routes]
        best_cost = current_cost

        # 3. Setup Adaptive Operators & Roulette Wheel
        destroy_ops = ["RandomRemoval", "WorstRemoval", "ShawRemoval"]
        repair_ops = ["RandomRepair", "GreedyRepair", "RegretRepair"]

        d_weights = [1.0, 1.0, 1.0]
        d_scores = [0.0, 0.0, 0.0]
        d_counts = [0, 0, 0]

        r_weights = [1.0, 1.0, 1.0]
        r_scores = [0.0, 0.0, 0.0]
        r_counts = [0, 0, 0]

        accepted_moves = 0
        rejected_moves = 0
        improving_moves = 0

        convergence_history = [round(best_cost, 2)]

        # Simulated Annealing initial temperature (accepts 50% worse moves with ~50% probability)
        temperature = max(1.0, (current_cost * 0.05) / math.log(2.0))
        cooling_rate = 0.97
        epoch_length = 10

        stopping_reason = "MAX_ITERATIONS"

        # 4. Main ALNS Iteration Loop
        for iteration in range(T_max):
            if (time.perf_counter() - started) >= t_limit:
                stopping_reason = "TIME_LIMIT"
                break

            # Roulette-wheel select destroy & repair operators
            d_idx = self._roulette_select(d_weights, rng)
            r_idx = self._roulette_select(r_weights, rng)

            d_counts[d_idx] += 1
            r_counts[r_idx] += 1

            # Determine number of customers to remove: q in [1, min(N, max(2, int(0.4 * N)))]
            max_q = min(N, max(2, int(0.4 * N))) if N >= 3 else 1
            min_q = 1
            q = rng.randint(min_q, max_q)

            # Destroy step
            destroyed_routes, unserved = self._apply_destroy(
                current_routes, d_idx, q, problem, get_cost, rng
            )

            # Repair step
            candidate_routes = self._apply_repair(
                destroyed_routes, unserved, r_idx, vehicles, problem, get_cost, rng
            )

            candidate_cost = solution_cost(candidate_routes)
            delta = candidate_cost - current_cost

            # Acceptance Criterion (Simulated Annealing)
            score_awarded = 0.0
            if delta < 0:
                current_routes = [r[:] for r in candidate_routes]
                current_cost = candidate_cost
                accepted_moves += 1
                improving_moves += 1
                score_awarded = 9.0  # Improving solution

                if candidate_cost < best_cost - 1e-4:
                    best_routes = [r[:] for r in candidate_routes]
                    best_cost = candidate_cost
                    score_awarded = 33.0  # Global best solution
            else:
                prob_accept = math.exp(-delta / max(0.001, temperature))
                if rng.random() < prob_accept:
                    current_routes = [r[:] for r in candidate_routes]
                    current_cost = candidate_cost
                    accepted_moves += 1
                    score_awarded = 13.0  # Non-improving solution accepted
                else:
                    rejected_moves += 1

            d_scores[d_idx] += score_awarded
            r_scores[r_idx] += score_awarded

            # Cool temperature
            temperature *= cooling_rate
            convergence_history.append(round(best_cost, 2))

            # Periodic Adaptive Weight Update (Reaction factor lambda = 0.8)
            if (iteration + 1) % epoch_length == 0:
                for i in range(len(d_weights)):
                    if d_counts[i] > 0:
                        performance = d_scores[i] / d_counts[i]
                        d_weights[i] = self.decay_factor * d_weights[i] + (1.0 - self.decay_factor) * performance
                        d_weights[i] = max(0.1, d_weights[i])
                        d_scores[i] = 0.0
                        d_counts[i] = 0

                for i in range(len(r_weights)):
                    if r_counts[i] > 0:
                        performance = r_scores[i] / r_counts[i]
                        r_weights[i] = self.decay_factor * r_weights[i] + (1.0 - self.decay_factor) * performance
                        r_weights[i] = max(0.1, r_weights[i])
                        r_scores[i] = 0.0
                        r_counts[i] = 0

        elapsed_ms = (time.perf_counter() - started) * 1000

        # 5. Format and Evaluate Final Fleet Solution with Common VRPEvaluator
        fleet_inputs: List[Tuple[VehicleConfig, List[str]]] = []
        for idx, v in enumerate(vehicles):
            cust_seq = best_routes[idx] if idx < len(best_routes) else []
            full_seq = [origin] + cust_seq + [origin] if cust_seq else [origin, origin]
            fleet_inputs.append((v, full_seq))

        final_res = VRPEvaluator.evaluate_fleet_routes(
            problem=problem,
            routes=fleet_inputs,
            algorithm_name="Classical ALNS (From Scratch)",
            computation_time_ms=elapsed_ms,
        )

        math_proof = {
            "algorithm_methodology": "Classical ALNS with Adaptive Operator Selection & Simulated Annealing",
            "paper_references": [
                "Paper 1: Adaptive Mechanism for Large Neighborhood Search (Methodological Basis for Classical ALNS)",
                "Ropke & Pisinger: Adaptive Large Neighborhood Search for the Pickup and Delivery Problem with Time Windows (Transp. Sci. 40(4), 2006)"
            ],
            "acceptance_criterion": "Simulated Annealing: P(accept) = exp(-delta / T), T_0 derived from initial cost, cooled at c=0.97",
            "acceptance_rationale": "Enables exploration of non-improving search neighborhoods in early iterations while progressively exploiting high-quality basins near incumbent solutions.",
            "adaptive_weight_formula": "w_i(t+1) = lambda * w_i(t) + (1 - lambda) * (score_i / count_i)",
            "score_bonuses": {"sigma_1_global_best": 33, "sigma_2_improving": 9, "sigma_3_accepted_worse": 13},
            "stopping_reason": stopping_reason,
            "seed": self.seed,
        }

        solver_details = {
            "iterations_executed": len(convergence_history) - 1,
            "max_iterations": T_max,
            "destroy_operators": destroy_ops,
            "repair_operators": repair_ops,
            "final_destroy_weights": {op: round(d_weights[i], 3) for i, op in enumerate(destroy_ops)},
            "final_repair_weights": {op: round(r_weights[i], 3) for i, op in enumerate(repair_ops)},
            "destroy_operator_selection_counts": {op: d_counts[i] for i, op in enumerate(destroy_ops)},
            "repair_operator_selection_counts": {op: r_counts[i] for i, op in enumerate(repair_ops)},
            "accepted_moves_count": accepted_moves,
            "rejected_moves_count": rejected_moves,
            "improving_moves_count": improving_moves,
            "convergence_history": convergence_history,
            "seed": self.seed,
            "stopping_reason": stopping_reason,
        }

        final_res.algorithm = "Classical ALNS (From Scratch)"
        final_res.math_proof.update(math_proof)
        final_res.solver_details = solver_details
        return final_res

    def _build_initial_solution(
        self,
        destinations: List[str],
        vehicles: List[VehicleConfig],
        problem: ProblemInstance,
        get_cost,
        rng: random.Random,
    ) -> List[List[str]]:
        """
        Builds an initial feasible multi-vehicle solution using parallel greedy insertion.
        Respects individual vehicle capacities and minimizes dynamic travel time on G(t).
        """
        K = len(vehicles)
        routes: List[List[str]] = [[] for _ in range(K)]
        unassigned = destinations[:]
        rng.shuffle(unassigned)

        for cust in unassigned:
            demand = problem.customer_demands.get(cust, 1.0)
            best_cost_delta = float("inf")
            best_v_idx = -1
            best_pos = -1

            for v_idx in range(K):
                cap = vehicles[v_idx].capacity
                curr_load = sum(problem.customer_demands.get(c, 1.0) for c in routes[v_idx])
                if curr_load + demand > cap:
                    continue

                r = routes[v_idx]
                for p in range(len(r) + 1):
                    prev_node = problem.origin if p == 0 else r[p - 1]
                    next_node = problem.origin if p == len(r) else r[p]
                    delta = get_cost(prev_node, cust) + get_cost(cust, next_node) - get_cost(prev_node, next_node)
                    if delta < best_cost_delta:
                        best_cost_delta = delta
                        best_v_idx = v_idx
                        best_pos = p

            if best_v_idx >= 0:
                routes[best_v_idx].insert(best_pos, cust)
            else:
                # Fallback: assign to vehicle with least overload
                least_overload = float("inf")
                target_v = 0
                for v_idx in range(K):
                    load = sum(problem.customer_demands.get(c, 1.0) for c in routes[v_idx])
                    if load < least_overload:
                        least_overload = load
                        target_v = v_idx
                routes[target_v].append(cust)

        return routes

    def _roulette_select(self, weights: List[float], rng: random.Random) -> int:
        total = sum(weights)
        if total <= 0:
            return rng.randint(0, len(weights) - 1)
        r = rng.uniform(0, total)
        acc = 0.0
        for idx, w in enumerate(weights):
            acc += w
            if r <= acc:
                return idx
        return len(weights) - 1

    def _apply_destroy(
        self,
        routes: List[List[str]],
        d_idx: int,
        q: int,
        problem: ProblemInstance,
        get_cost,
        rng: random.Random,
    ) -> Tuple[List[List[str]], List[str]]:
        """
        Applies selected destroy operator:
        0: Random Removal
        1: Worst Removal (exact marginal cost contribution with Shaw randomized greediness p=3)
        2: Shaw / Relatedness Removal (distance and demand similarity)
        """
        new_routes = [r[:] for r in routes]
        all_customers = [c for r in new_routes for c in r]

        if not all_customers or q <= 0:
            return new_routes, []

        q = min(q, len(all_customers))

        if d_idx == 0:
            # 1. Random Removal
            removed = rng.sample(all_customers, q)
            removed_set = set(removed)
            for v_idx in range(len(new_routes)):
                new_routes[v_idx] = [c for c in new_routes[v_idx] if c not in removed_set]
            return new_routes, removed

        elif d_idx == 1:
            # 2. Worst Removal: Calculate exact marginal cost contribution
            customer_deltas: List[Tuple[float, str, int]] = []
            for v_idx, r in enumerate(new_routes):
                for p, c in enumerate(r):
                    prev_node = problem.origin if p == 0 else r[p - 1]
                    next_node = problem.origin if p == len(r) - 1 else r[p + 1]
                    marginal_delta = get_cost(prev_node, c) + get_cost(c, next_node) - get_cost(prev_node, next_node)
                    customer_deltas.append((marginal_delta, c, v_idx))

            customer_deltas.sort(key=lambda item: item[0], reverse=True)

            removed: List[str] = []
            p_greediness = 3
            while len(removed) < q and customer_deltas:
                y = rng.random()
                sel_idx = int(math.floor((y ** p_greediness) * len(customer_deltas)))
                sel_idx = min(sel_idx, len(customer_deltas) - 1)
                chosen = customer_deltas.pop(sel_idx)
                removed.append(chosen[1])

            removed_set = set(removed)
            for v_idx in range(len(new_routes)):
                new_routes[v_idx] = [c for c in new_routes[v_idx] if c not in removed_set]
            return new_routes, removed

        else:
            # 3. Shaw / Relatedness Removal
            pivot = rng.choice(all_customers)
            removed = [pivot]
            remaining = [c for c in all_customers if c != pivot]

            max_dist = max([get_cost(pivot, c) for c in remaining], default=1.0) or 1.0
            max_dem = max([abs(problem.customer_demands.get(pivot, 1.0) - problem.customer_demands.get(c, 1.0)) for c in remaining], default=1.0) or 1.0

            p_greediness = 3
            while len(removed) < q and remaining:
                relatedness_list = []
                for c in remaining:
                    d = get_cost(pivot, c)
                    dem_diff = abs(problem.customer_demands.get(pivot, 1.0) - problem.customer_demands.get(c, 1.0))
                    r_val = 0.7 * (d / max_dist) + 0.3 * (dem_diff / max_dem)
                    relatedness_list.append((r_val, c))

                relatedness_list.sort(key=lambda item: item[0])
                y = rng.random()
                sel_idx = int(math.floor((y ** p_greediness) * len(relatedness_list)))
                sel_idx = min(sel_idx, len(relatedness_list) - 1)
                chosen_cust = relatedness_list[sel_idx][1]

                removed.append(chosen_cust)
                remaining.remove(chosen_cust)

            removed_set = set(removed)
            for v_idx in range(len(new_routes)):
                new_routes[v_idx] = [c for c in new_routes[v_idx] if c not in removed_set]
            return new_routes, removed

    def _apply_repair(
        self,
        routes: List[List[str]],
        unserved: List[str],
        r_idx: int,
        vehicles: List[VehicleConfig],
        problem: ProblemInstance,
        get_cost,
        rng: random.Random,
    ) -> List[List[str]]:
        """
        Applies selected repair operator:
        0: Random Repair (random unserved customer into random feasible position)
        1: Greedy Repair (cheapest insertion cost across all vehicles and positions)
        2: Regret-2 / Regret-3 Repair (maximizes difference between 2nd cheapest and cheapest insertion)
        """
        new_routes = [r[:] for r in routes]
        pending = unserved[:]
        K = len(vehicles)

        if not pending:
            return new_routes

        if r_idx == 0:
            # 1. Random Repair
            rng.shuffle(pending)
            for cust in pending:
                demand = problem.customer_demands.get(cust, 1.0)
                feasible_positions: List[Tuple[int, int]] = []
                for v_idx in range(K):
                    cap = vehicles[v_idx].capacity
                    load = sum(problem.customer_demands.get(c, 1.0) for c in new_routes[v_idx])
                    if load + demand <= cap:
                        for p in range(len(new_routes[v_idx]) + 1):
                            feasible_positions.append((v_idx, p))

                if feasible_positions:
                    chosen_v, chosen_p = rng.choice(feasible_positions)
                    new_routes[chosen_v].insert(chosen_p, cust)
                else:
                    min_v = min(range(K), key=lambda idx: sum(problem.customer_demands.get(c, 1.0) for c in new_routes[idx]))
                    new_routes[min_v].append(cust)
            return new_routes

        elif r_idx == 1:
            # 2. Greedy Repair
            while pending:
                best_cust = None
                best_delta = float("inf")
                best_v = -1
                best_p = -1

                for cust in pending:
                    demand = problem.customer_demands.get(cust, 1.0)
                    for v_idx in range(K):
                        cap = vehicles[v_idx].capacity
                        load = sum(problem.customer_demands.get(c, 1.0) for c in new_routes[v_idx])
                        if load + demand > cap:
                            continue

                        r = new_routes[v_idx]
                        for p in range(len(r) + 1):
                            prev_node = problem.origin if p == 0 else r[p - 1]
                            next_node = problem.origin if p == len(r) else r[p]
                            delta = get_cost(prev_node, cust) + get_cost(cust, next_node) - get_cost(prev_node, next_node)
                            if delta < best_delta:
                                best_delta = delta
                                best_cust = cust
                                best_v = v_idx
                                best_p = p

                if best_cust is not None:
                    new_routes[best_v].insert(best_p, best_cust)
                    pending.remove(best_cust)
                else:
                    fallback_cust = pending.pop(0)
                    min_v = min(range(K), key=lambda idx: sum(problem.customer_demands.get(c, 1.0) for c in new_routes[idx]))
                    new_routes[min_v].append(fallback_cust)
            return new_routes

        else:
            # 3. Regret Repair (Regret-2 / Regret-3)
            regret_k = min(3, K)
            while pending:
                best_regret = -1.0
                selected_cust = None
                selected_v = -1
                selected_p = -1

                for cust in pending:
                    demand = problem.customer_demands.get(cust, 1.0)
                    insertion_costs: List[Tuple[float, int, int]] = []

                    for v_idx in range(K):
                        cap = vehicles[v_idx].capacity
                        load = sum(problem.customer_demands.get(c, 1.0) for c in new_routes[v_idx])
                        if load + demand > cap:
                            continue

                        r = new_routes[v_idx]
                        for p in range(len(r) + 1):
                            prev_node = problem.origin if p == 0 else r[p - 1]
                            next_node = problem.origin if p == len(r) else r[p]
                            delta = get_cost(prev_node, cust) + get_cost(cust, next_node) - get_cost(prev_node, next_node)
                            insertion_costs.append((delta, v_idx, p))

                    if not insertion_costs:
                        continue

                    insertion_costs.sort(key=lambda item: item[0])
                    c_1 = insertion_costs[0][0]

                    if len(insertion_costs) >= 2:
                        regret_val = sum((insertion_costs[j][0] - c_1) for j in range(1, min(regret_k, len(insertion_costs))))
                    else:
                        regret_val = 1e5

                    if regret_val > best_regret:
                        best_regret = regret_val
                        selected_cust = cust
                        selected_v = insertion_costs[0][1]
                        selected_p = insertion_costs[0][2]

                if selected_cust is not None:
                    new_routes[selected_v].insert(selected_p, selected_cust)
                    pending.remove(selected_cust)
                else:
                    fallback_cust = pending.pop(0)
                    min_v = min(range(K), key=lambda idx: sum(problem.customer_demands.get(c, 1.0) for c in new_routes[idx]))
                    new_routes[min_v].append(fallback_cust)

            return new_routes

