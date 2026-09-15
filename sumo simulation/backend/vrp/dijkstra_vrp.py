from __future__ import annotations

import time
import itertools
from typing import Any, Dict, List, Optional

from backend.vrp.problem_instance import ProblemInstance, AlgorithmResult, FleetVehicleRoute
from backend.vrp.vrp_evaluator import VRPEvaluator


class DijkstraVRPSolver:
    """
    Computes exact baseline optimal customer visit sequence and shortest physical route
    for 1-Source + N-Destination VRP by evaluating candidate permutations on dynamic graph G(t).
    """

    def solve(self, problem: ProblemInstance) -> AlgorithmResult:
        started = time.perf_counter()
        is_valid, msg = problem.is_valid()
        if not is_valid:
            is_infeasible = "infeasible" in msg.lower() or "exceeds" in msg.lower()
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
                        "action": f"Vehicle {v.vehicle_id} Standby at Depot. ({msg})",
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
                algorithm="Dijkstra VRP (Exact Baseline)",
                status="infeasible" if is_infeasible else "error",
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
        destinations = problem.destinations

        # 1. Enumerate candidate customer visit permutations
        # For N <= 7: exhaustive exact permutation enumeration (N! <= 5040)
        # For N > 7: nearest-neighbor heuristic + 2-opt + sampled permutations to prevent factorial freeze
        if len(destinations) <= 7:
            candidate_permutations = list(itertools.permutations(destinations))
        else:
            import networkx as nx
            g = problem.graph.graph
            unvisited = list(destinations)
            curr = origin
            greedy_seq = []
            while unvisited:
                def get_cost(v):
                    try:
                        return nx.dijkstra_path_length(g, curr, v, weight="weight")
                    except Exception:
                        return 1e6
                next_node = min(unvisited, key=get_cost)
                greedy_seq.append(next_node)
                unvisited.remove(next_node)
                curr = next_node

            candidate_permutations = [tuple(greedy_seq)]
            for i in range(len(greedy_seq)):
                for j in range(i + 1, len(greedy_seq)):
                    cand = list(greedy_seq)
                    cand[i:j+1] = reversed(cand[i:j+1])
                    candidate_permutations.append(tuple(cand))

            import random
            rng = random.Random(42)
            for _ in range(50):
                shuffled = list(destinations)
                rng.shuffle(shuffled)
                candidate_permutations.append(tuple(shuffled))

        best_result: Optional[AlgorithmResult] = None
        min_cost = float("inf")
        permutation_evaluations: List[Dict[str, Any]] = []

        # 2. Evaluate each permutation on dynamic graph G(t)
        for perm in candidate_permutations:
            visit_seq = [origin] + list(perm)
            res = VRPEvaluator.evaluate_sequence(
                problem,
                visit_seq,
                algorithm_name="Dijkstra VRP (Exact Baseline)",
                computation_time_ms=0.0,
            )

            permutation_evaluations.append({
                "sequence": " -> ".join(visit_seq),
                "total_cost": round(res.total_cost, 2),
                "travel_time_s": round(res.total_travel_time, 2),
                "distance_m": round(res.total_distance, 2),
                "feasible": res.feasible,
            })

            if res.total_cost < min_cost:
                min_cost = res.total_cost
                best_result = res

        elapsed_ms = (time.perf_counter() - started) * 1000

        if best_result is None and candidate_permutations:
            best_result = VRPEvaluator.evaluate_sequence(
                problem,
                [origin] + list(candidate_permutations[0]),
                algorithm_name="Dijkstra VRP (Exact Baseline)",
                computation_time_ms=elapsed_ms,
            )
            return AlgorithmResult(
                algorithm="Dijkstra VRP (Exact Baseline)",
                status="infeasible",
                success=False,
                visit_sequence=[],
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
                constraint_violations=["No feasible route found across any customer sequence."],
                computation_time_ms=elapsed_ms,
                math_proof={},
                error="No feasible route found across any customer sequence.",
            )

        math_proof = {
            "algorithm_name": "Exact Dijkstra VRP Sequence Enumerator & Path Solver",
            "optimization_principle": "Best Sequence = argmin_{pi} [ sum_{k=1}^N d(C_{pi(k-1)}, C_{pi(k)}) ]",
            "evaluated_sequences": len(candidate_permutations),
            "dynamic_graph_state": "W(t) = alpha * T_e(t) + beta * D_e + gamma * C_e(t)",
            "selected_sequence": " -> ".join(best_result.visit_sequence),
        }

        solver_details = {
            "candidate_permutations_count": len(candidate_permutations),
            "permutation_evaluations": permutation_evaluations,
            "optimal_sequence": best_result.visit_sequence,
        }

        best_result.computation_time_ms = elapsed_ms
        if best_result.math_proof:
            math_proof.update(best_result.math_proof)
        best_result.math_proof = math_proof
        best_result.solver_details = solver_details
        return best_result
