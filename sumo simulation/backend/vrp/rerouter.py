from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from backend.vrp.problem_instance import ProblemInstance, AlgorithmResult, FleetVehicleRoute
from backend.vrp.vrp_evaluator import VRPEvaluator


class DynamicRerouter:
    """
    Dynamic Rerouting Engine for dynamic transportation networks.
    Monitors live SUMO traffic and incidents to distinguish:
    1. Local Road Rerouting: Sequence of customer stops remains identical, but road path is recalculated.
    2. Global VRP Re-optimization: Sequence of customers or vehicle assignment changes due to major bottlenecks/accidents.
    """

    def __init__(self, cost_increase_threshold_pct: float = 20.0):
        self.cost_increase_threshold_pct = cost_increase_threshold_pct

    def evaluate_rerouting_policy(
        self,
        problem: ProblemInstance,
        current_result: AlgorithmResult,
        active_incidents: List[str],
        solver_fn,
    ) -> Tuple[str, AlgorithmResult, Dict[str, Any]]:
        """
        Evaluates whether current route is affected by traffic/incidents and applies local vs global rerouting.

        Returns (action_type, updated_result, reroute_metadata)
        where action_type in ['NO_CHANGE', 'LOCAL_REROUTE', 'GLOBAL_REOPTIMIZATION']
        """
        if not current_result or not current_result.success:
            # Re-solve if current result was invalid
            new_res = solver_fn(problem)
            return "GLOBAL_REOPTIMIZATION", new_res, {"reason": "Initial route invalid or missing"}

        # Check if any edge in current route is affected by an incident
        route_edges = set(current_result.edge_path)
        affected_incidents = [e for e in active_incidents if e in route_edges]

        # Re-evaluate current customer sequence on new graph G(t)
        if current_result.fleet_routes:
            current_seq_routes = [(v, fr.visit_sequence) for v, fr in zip(problem.vehicles, current_result.fleet_routes)]
            re_eval_res = VRPEvaluator.evaluate_fleet_routes(problem, current_seq_routes, algorithm_name="Local Reroute Evaluator")
        else:
            re_eval_res = VRPEvaluator.evaluate_sequence(problem, current_result.visit_sequence, algorithm_name="Local Reroute Evaluator")

        if not re_eval_res.feasible:
            # Current sequence no longer feasible -> Global Re-optimization required
            new_res = solver_fn(problem)
            return "GLOBAL_REOPTIMIZATION", new_res, {
                "reason": "Road blockage rendered current customer sequence physically unviable",
                "affected_incidents": affected_incidents,
            }

        prev_cost = max(current_result.total_cost, 0.1)
        new_cost = re_eval_res.total_cost
        cost_increase_pct = ((new_cost - prev_cost) / prev_cost) * 100.0

        if affected_incidents or cost_increase_pct >= self.cost_increase_threshold_pct:
            # Run full solver to test if global re-optimization yields better sequence
            solver_res = solver_fn(problem)

            if solver_res.feasible and solver_res.total_cost < new_cost * 0.95:
                # Global sequence change is significantly better
                return "GLOBAL_REOPTIMIZATION", solver_res, {
                    "reason": f"Global re-optimization found a faster sequence (Saved {round(new_cost - solver_res.total_cost, 1)}s)",
                    "affected_incidents": affected_incidents,
                    "cost_increase_pct": round(cost_increase_pct, 1),
                    "old_sequence": current_result.visit_sequence,
                    "new_sequence": solver_res.visit_sequence,
                }
            else:
                # Keep customer sequence, update physical road path
                return "LOCAL_REROUTE", re_eval_res, {
                    "reason": "Customer sequence remains optimal, recalculated physical road path around congestion",
                    "affected_incidents": affected_incidents,
                    "cost_increase_pct": round(cost_increase_pct, 1),
                }

        return "NO_CHANGE", re_eval_res, {
            "reason": "Route unaffected by traffic changes",
            "cost_increase_pct": round(cost_increase_pct, 1),
        }
