from __future__ import annotations

import time
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from backend.vrp.problem_instance import ProblemInstance, AlgorithmResult, FleetVehicleRoute
from backend.vrp.vrp_evaluator import VRPEvaluator


class QPSOVRPSolver:
    """
    Quantum-Behaved Particle Swarm Optimization (QPSO) VRP Solver module.
    Formulated strictly according to Herrera, Coelho, Steiner (Pesquisa Operacional 35(3), 2015).

    Mathematical Pipeline & Equations:
    1. Quantum Potential Field & Wave Collapse Position Update (Eq. 13 & Algorithm 1):
       x_{i,d}(t+1) = p_{i,d} +/- alpha(t) * |mbest_d - x_{i,d}(t)| * ln(1/u)
       where u ~ Uniform(0, 1), +/- chosen with 0.5 probability.
    2. Learning Inclination Point (LIP) p_{i,d} (Algorithm 1):
       p_{i,d} = (fi_1 * pbest_{i,d} + fi_2 * gbest_d) / (fi_1 + fi_2)
       where fi_1, fi_2 ~ Uniform(0, 1).
    3. Mean Best Position mbest (Eq. 11):
       mbest_d = (1 / M) * sum_{i=1}^M pbest_{i,d}
    4. Contraction-Expansion Coefficient alpha (Sec 4):
       Decreases linearly from alpha_start = 1.0 down to alpha_end = 0.5 over iterations.
    5. Rank Discretization Mapping (Sec 3.3, Page 14-15):
       Ascending sort order of continuous particle position vector x_i defines discrete customer sequence pi.
    6. Fitness Evaluation:
       Sum of dynamic edge travel times / costs on SUMO graph G(t).
    """

    def __init__(
        self,
        num_particles: int = 20,
        max_iter: int = 25,
        alpha_start: float = 1.0,
        alpha_end: float = 0.5,
        seed: int = 42,
    ):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.seed = seed

    def solve(
        self,
        problem: ProblemInstance,
        num_particles: Optional[int] = None,
        max_iter: Optional[int] = None,
    ) -> AlgorithmResult:
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
                algorithm="QPSO VRP Swarm Solver (Herrera et al. 2015)",
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

        M = num_particles or self.num_particles
        T_max = max_iter or self.max_iter
        rng = np.random.default_rng(self.seed)

        origin = problem.origin
        destinations = problem.destinations
        D = len(destinations)  # Dimension D = number of customers to serve

        # 1. Initialize continuous particle swarm population M x D
        # x_i ~ Uniform(-2.0, 2.0)
        x = rng.uniform(-2.0, 2.0, size=(M, D))
        pbest = x.copy()
        pbest_fitness = np.full(M, float("inf"))
        pbest_results: List[Optional[AlgorithmResult]] = [None] * M

        # Evaluate initial population fitness
        for i in range(M):
            seq = self._discretize_to_customer_sequence(x[i], origin, destinations)
            res = VRPEvaluator.evaluate_sequence(problem, seq, algorithm_name="QPSO VRP Swarm Solver")
            pbest_fitness[i] = res.total_cost if res.feasible else 1e9 + res.total_cost
            pbest_results[i] = res

        gbest_idx = int(np.argmin(pbest_fitness))
        gbest = pbest[gbest_idx].copy()
        gbest_fitness = float(pbest_fitness[gbest_idx])
        gbest_result = pbest_results[gbest_idx]

        convergence_history = [gbest_fitness]

        # 2. Main QPSO Swarm Iteration Loop (Herrera et al., Algorithm 1)
        for t in range(T_max):
            # Linear Contraction-Expansion decay: alpha(t) from 1.0 to 0.5 (Sec 4)
            alpha_t = self.alpha_start - (t / max(1, T_max - 1)) * (self.alpha_start - self.alpha_end)

            # Eq. 11: mbest = mean of pbest across swarm population M
            mbest = np.mean(pbest, axis=0)

            for i in range(M):
                for d in range(D):
                    # Algorithm 1: Random cognitive and social parameters fi1, fi2
                    fi1 = rng.uniform(0.0, 1.0)
                    fi2 = rng.uniform(0.0, 1.0)
                    denom = fi1 + fi2
                    if denom < 1e-9:
                        denom = 1.0

                    # Learning Inclination Point (LIP) p_{i,d}
                    p_id = (fi1 * pbest[i, d] + fi2 * gbest[d]) / denom

                    # Eq. 13 & Algorithm 1: Quantum wave collapse position update
                    u = rng.uniform(1e-7, 1.0)
                    ln_u = math.log(1.0 / u)
                    diff = abs(mbest[d] - x[i, d])

                    if rng.uniform(0.0, 1.0) > 0.5:
                        x[i, d] = p_id - alpha_t * diff * ln_u
                    else:
                        x[i, d] = p_id + alpha_t * diff * ln_u

                # Evaluate new position fitness
                seq = self._discretize_to_customer_sequence(x[i], origin, destinations)
                res = VRPEvaluator.evaluate_sequence(problem, seq, algorithm_name="QPSO VRP Swarm Solver")
                current_fitness = res.total_cost if res.feasible else 1e9 + res.total_cost

                # Update Personal Best pbest_i
                if current_fitness < pbest_fitness[i]:
                    pbest[i] = x[i].copy()
                    pbest_fitness[i] = current_fitness
                    pbest_results[i] = res

                    # Update Global Best gbest
                    if current_fitness < gbest_fitness:
                        gbest = x[i].copy()
                        gbest_fitness = current_fitness
                        gbest_result = res

            convergence_history.append(gbest_fitness)

        elapsed_ms = (time.perf_counter() - started) * 1000

        if gbest_result is None or not gbest_result.feasible:
            # Fallback to evaluating best possible sequence
            best_seq = self._discretize_to_customer_sequence(gbest, origin, destinations)
            gbest_result = VRPEvaluator.evaluate_sequence(problem, best_seq, algorithm_name="QPSO VRP Swarm Solver")

        rank_discretization_table = []
        for d_idx, dest_node in enumerate(destinations):
            rank_discretization_table.append({
                "dimension_index": d_idx,
                "customer_node": dest_node,
                "gbest_value": round(float(gbest[d_idx]), 4),
                "mbest_value": round(float(mbest[d_idx]), 4),
                "visit_rank": sorted(range(D), key=lambda k: float(gbest[k])).index(d_idx) + 1,
            })

        math_proof = {
            "paper_reference": "Herrera, Coelho, Steiner (Pesquisa Operacional 35(3), 2015 / pp. 1-20)",
            "position_update_eq": "x_{i,d}(t+1) = p_{i,d} +/- alpha(t) * |mbest_d - x_{i,d}(t)| * ln(1/u)",
            "lip_eq": "p_{i,d} = (fi_1 * pbest_{i,d} + fi_2 * gbest_d) / (fi_1 + fi_2)",
            "mbest_eq": "mbest_d = (1 / M) * sum_{i=1}^M pbest_{i,d}",
            "alpha_decay_eq": "alpha(t) = alpha_start - (t / (T_max - 1)) * (alpha_start - alpha_end)",
            "rank_discretization_rule": "Ascending sort order of continuous particle values x_d defines customer visit permutation pi (Sec 3.3, pp. 14-15)",
        }

        solver_details = {
            "particles_M": M,
            "iterations_T": T_max,
            "dimensions_D": D,
            "alpha_start": self.alpha_start,
            "alpha_end": self.alpha_end,
            "final_alpha": round(alpha_t, 4),
            "gbest_vector": [round(float(v), 4) for v in gbest],
            "mbest_vector": [round(float(v), 4) for v in mbest],
            "rank_discretization_table": rank_discretization_table,
            "convergence_history": [round(f, 2) for f in convergence_history],
        }

        gbest_result.algorithm = "QPSO VRP Swarm Solver (Herrera et al. 2015)"
        gbest_result.computation_time_ms = elapsed_ms
        if gbest_result.math_proof:
            math_proof.update(gbest_result.math_proof)
        gbest_result.math_proof = math_proof
        gbest_result.solver_details = solver_details
        return gbest_result

    @staticmethod
    def _discretize_to_customer_sequence(
        x_particle: np.ndarray, origin: str, destinations: List[str]
    ) -> List[str]:
        """
        Applies Paper Sec 3.3 Rank Discretization Rule:
        Ascending sort order of continuous values x_particle defines customer visit order sequence.
        """
        D = len(destinations)
        if D == 1:
            return [origin, destinations[0]]

        # Sort destination indices based on continuous particle x values
        sorted_customer_indices = sorted(range(D), key=lambda idx: float(x_particle[idx]))
        return [origin] + [destinations[i] for i in sorted_customer_indices]
