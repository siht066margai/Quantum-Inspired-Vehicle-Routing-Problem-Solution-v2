from __future__ import annotations

import time
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from backend.vrp.problem_instance import ProblemInstance, AlgorithmResult, VehicleConfig, FleetVehicleRoute
from backend.vrp.vrp_evaluator import VRPEvaluator


@dataclass
class HGSIndividual:
    chromosome: List[str]  # Giant-tour customer permutation without depot
    routes: List[List[str]]  # Customer sequences per vehicle
    cost: float  # Pure routing cost (travel times on G(t))
    excess_load: float  # Sum of capacity violations across vehicles
    penalized_cost: float  # cost + omega_cap * excess_load
    is_feasible: bool
    biased_fitness: float = 0.0
    diversity_contribution: float = 0.0


class HGSVRPSolver:
    """
    Hybrid Genetic Search (HGS) Metaheuristic Solver for Dynamic Fleet CVRP.
    Formulated strictly from scratch based on Vidal et al. (Operations Research 60(3), 2012;
    Computers & Operations Research 140, 2022) and Paper 2.

    Key Algorithmic Mechanics:
    1. Giant-Tour Chromosome Representation:
       A customer sequence pi = [c_1, c_2, ..., c_N] without explicit depot markers.
       Decoded into K capacity-constrained / penalized vehicle routes via Prins' DP Split algorithm.
    2. Dual Subpopulations:
       - Feasible Subpopulation P_feas: solutions with zero capacity violations.
       - Infeasible Subpopulation P_infeas: solutions evaluated with dynamic penalty parameter omega_cap.
    3. Population Diversity & Biased Fitness:
       - Broken-pairs / adjacent edge distance delta(S_1, S_2) measures sequence diversity.
       - Biased fitness combines solution cost rank with diversity rank:
         fit(S) = rank_cost(S) + (1 - mu / |P|) * rank_div(S)
    4. Parent Selection & Crossover:
       - Binary tournament selection using biased fitness.
       - Ordered Crossover (OX) applied to giant-tour chromosomes.
    5. Local Search Neighborhood Operators:
       - Relocate Operator: shifts customer c_i to another position.
       - Swap Operator: exchanges positions of c_i and c_j.
       - 2-Opt Operator: inverts segment [i..j].
       - Evaluated against penalized objective function f_pen(S).
    6. Dynamic Penalty Adaptation:
       Adjusts omega_cap every tau = 20 offspring based on the target feasibility ratio (20% to 50%).
    7. Survivor Selection:
       Truncates subpopulations when exceeding (mu + lambda) back to mu based on biased fitness.
    """

    def __init__(
        self,
        mu: int = 20,  # Minimum subpopulation size
        lambda_offspring: int = 15,  # Offspring generation buffer before survivor selection
        pop_size: Optional[int] = None,
        max_iter: int = 50,
        mutation_rate: float = 0.35,
        seed: int = 42,
        time_limit_s: float = 15.0,
    ):
        self.mu = pop_size or mu
        self.lambda_offspring = lambda_offspring
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.time_limit_s = time_limit_s

    def solve(
        self,
        problem: ProblemInstance,
        pop_size: Optional[int] = None,
        max_iter: Optional[int] = None,
        time_limit_s: Optional[float] = None,
    ) -> AlgorithmResult:
        started = time.perf_counter()
        t_limit = time_limit_s or self.time_limit_s
        T_max = max_iter or self.max_iter
        target_mu = pop_size or self.mu

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
                algorithm="Hybrid Genetic Search HGS (From Scratch)",
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
            return VRPEvaluator.evaluate_sequence(problem, [origin, origin], algorithm_name="Hybrid Genetic Search HGS (From Scratch)")

        rng = random.Random(self.seed)

        # Precompute leg costs
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

        # Penalty coefficient for capacity overloads
        omega_cap = 20.0
        feas_count_in_window = 0
        total_in_window = 0
        adaptation_window = 20

        # 2. Initialize Dual Subpopulations (P_feas and P_infeas)
        p_feas: List[HGSIndividual] = []
        p_infeas: List[HGSIndividual] = []

        # Population seeding: random permutations + greedy sequence
        initial_candidates: List[List[str]] = []

        # Seed 1: Greedy nearest-neighbor customer sequence
        curr = origin
        unvisited = destinations[:]
        nn_seq = []
        while unvisited:
            next_c = min(unvisited, key=lambda c: get_cost(curr, c))
            nn_seq.append(next_c)
            unvisited.remove(next_c)
            curr = next_c
        initial_candidates.append(nn_seq)

        # Seed 2..target_mu * 2: Randomized permutations
        while len(initial_candidates) < max(target_mu * 2, 10):
            shuffled = destinations[:]
            rng.shuffle(shuffled)
            initial_candidates.append(shuffled)

        for perm in initial_candidates:
            ind = self._evaluate_and_split(perm, vehicles, problem, get_cost, omega_cap)
            if ind.is_feasible:
                p_feas.append(ind)
            else:
                p_infeas.append(ind)

        best_feasible: Optional[HGSIndividual] = min(p_feas, key=lambda ind: ind.cost) if p_feas else None
        convergence_history = [round(best_feasible.cost, 2) if best_feasible else 99999.0]

        stopping_reason = "MAX_GENERATIONS"

        # 3. Main HGS Evolutionary Generation Loop
        for generation in range(T_max):
            if (time.perf_counter() - started) >= t_limit:
                stopping_reason = "TIME_LIMIT"
                break

            # Recompute biased fitness for subpopulations
            self._update_biased_fitness(p_feas, target_mu)
            self._update_biased_fitness(p_infeas, target_mu)

            # Combined pool for parent selection
            parent_pool = p_feas + p_infeas
            if not parent_pool:
                break

            # Parent Selection via Binary Tournament
            p1 = self._binary_tournament(parent_pool, rng)
            p2 = self._binary_tournament(parent_pool, rng)

            # Ordered Crossover (OX) on giant tours
            offspring_perm = self._ordered_crossover(p1.chromosome, p2.chromosome, rng)

            # Local Search Improvement (Relocate, Swap, 2-Opt)
            if rng.random() < self.mutation_rate or True:
                offspring_perm = self._local_search(
                    offspring_perm, vehicles, problem, get_cost, omega_cap, rng
                )

            # Evaluate offspring via Prins' Split
            offspring = self._evaluate_and_split(offspring_perm, vehicles, problem, get_cost, omega_cap)

            total_in_window += 1
            if offspring.is_feasible:
                p_feas.append(offspring)
                feas_count_in_window += 1
                if best_feasible is None or offspring.cost < best_feasible.cost:
                    best_feasible = offspring
            else:
                p_infeas.append(offspring)

            # Survivor Selection: prune when exceeding mu + lambda
            if len(p_feas) > target_mu + self.lambda_offspring:
                self._update_biased_fitness(p_feas, target_mu)
                p_feas.sort(key=lambda ind: ind.biased_fitness)
                p_feas = p_feas[:target_mu]

            if len(p_infeas) > target_mu + self.lambda_offspring:
                self._update_biased_fitness(p_infeas, target_mu)
                p_infeas.sort(key=lambda ind: ind.biased_fitness)
                p_infeas = p_infeas[:target_mu]

            # Dynamic Penalty Adaptation
            if total_in_window >= adaptation_window:
                feas_ratio = feas_count_in_window / max(1, total_in_window)
                if feas_ratio < 0.20:
                    omega_cap = min(2000.0, omega_cap * 1.25)
                elif feas_ratio > 0.50:
                    omega_cap = max(0.1, omega_cap * 0.80)
                feas_count_in_window = 0
                total_in_window = 0

            cur_best_cost = best_feasible.cost if best_feasible else (min((ind.penalized_cost for ind in p_infeas), default=99999.0))
            convergence_history.append(round(cur_best_cost, 2))

        elapsed_ms = (time.perf_counter() - started) * 1000

        # If no feasible individual was found in the population, try best feasible split
        if best_feasible is None and p_infeas:
            best_infeas = min(p_infeas, key=lambda ind: ind.cost)
            # Try to evaluate best available
            best_feasible = best_infeas

        # 4. Final Solution Construction & Evaluation
        if best_feasible is not None:
            fleet_inputs: List[Tuple[VehicleConfig, List[str]]] = []
            for idx, v in enumerate(vehicles):
                cust_seq = best_feasible.routes[idx] if idx < len(best_feasible.routes) else []
                full_seq = [origin] + cust_seq + [origin] if cust_seq else [origin, origin]
                fleet_inputs.append((v, full_seq))

            final_res = VRPEvaluator.evaluate_fleet_routes(
                problem=problem,
                routes=fleet_inputs,
                algorithm_name="Hybrid Genetic Search HGS (From Scratch)",
                computation_time_ms=elapsed_ms,
            )
        else:
            final_res = VRPEvaluator.evaluate_sequence(
                problem=problem,
                visit_sequence=[origin, origin],
                algorithm_name="Hybrid Genetic Search HGS (From Scratch)",
                computation_time_ms=elapsed_ms,
            )

        # Compute final population diversity
        all_inds = p_feas + p_infeas
        avg_div = 0.0
        if len(all_inds) > 1:
            divs = [self._calc_diversity(all_inds[i].chromosome, all_inds[j].chromosome) for i in range(len(all_inds)) for j in range(i + 1, len(all_inds))]
            avg_div = sum(divs) / max(1, len(divs))

        math_proof = {
            "algorithm_methodology": "Hybrid Genetic Search (HGS) with Dual Subpopulations & Dynamic Penalties",
            "paper_references": [
                "Paper 2: Hybrid Genetic Search for Dynamic Vehicle Routing with Time Windows (Methodological Reference for HGS)",
                "Vidal et al.: A hybrid genetic algorithm with adaptive diversity management for a large class of vehicle routing problems with time-windows (Comput. Oper. Res. 40(1), 2013)",
                "Vidal et al.: Hybrid genetic search for the CVRP: 22 years later (Comput. Oper. Res. 140, 2022)"
            ],
            "solution_representation": "Giant-tour customer permutation decoded via Prins' DP Split algorithm across heterogeneous fleet capacities.",
            "subpopulation_architecture": {
                "feasible_subpopulation_size": len(p_feas),
                "infeasible_subpopulation_size": len(p_infeas),
                "target_mu": target_mu,
            },
            "penalty_mechanism": {
                "penalized_formula": "f_pen(S) = cost(S) + omega_cap * sum(max(0, load(R_k) - cap(R_k)))",
                "final_omega_cap": round(omega_cap, 2),
                "adaptation_rule": "Adjusted every 20 offspring based on feasibility window [20%, 50%]",
            },
            "diversity_management": {
                "metric": "Broken-pairs / adjacent edge distance delta(S1, S2)",
                "biased_fitness_formula": "fit(S) = rank_cost(S) + (1 - mu / |P|) * rank_div(S)",
                "average_population_diversity": round(avg_div, 3),
            },
            "stopping_reason": stopping_reason,
            "seed": self.seed,
        }

        solver_details = {
            "generations_executed": len(convergence_history) - 1,
            "max_generations": T_max,
            "target_mu": target_mu,
            "mutation_rate": self.mutation_rate,
            "final_omega_cap": round(omega_cap, 2),
            "feasible_solutions_count": len(p_feas),
            "infeasible_solutions_count": len(p_infeas),
            "average_diversity": round(avg_div, 3),
            "convergence_history": convergence_history,
            "seed": self.seed,
            "stopping_reason": stopping_reason,
        }

        final_res.algorithm = "Hybrid Genetic Search HGS (From Scratch)"
        final_res.math_proof.update(math_proof)
        final_res.solver_details = solver_details
        return final_res

    def _evaluate_and_split(
        self,
        perm: List[str],
        vehicles: List[VehicleConfig],
        problem: ProblemInstance,
        get_cost,
        omega_cap: float,
    ) -> HGSIndividual:
        """
        Applies Prins' Split algorithm to decode giant-tour permutation into K vehicle routes,
        computing routing cost, capacity overloads, and penalized fitness.
        """
        N = len(perm)
        K = len(vehicles)
        origin = problem.origin

        if N == 0:
            return HGSIndividual(
                chromosome=perm,
                routes=[[] for _ in range(K)],
                cost=0.0,
                excess_load=0.0,
                penalized_cost=0.0,
                is_feasible=True,
            )

        # K-stage DP: V[k][j] = min penalized cost to serve customers 0..j-1 with at most k vehicles
        V = [[float("inf")] * (N + 1) for _ in range(K + 1)]
        P = [[-1] * (N + 1) for _ in range(K + 1)]
        V[0][0] = 0.0

        for k in range(1, K + 1):
            cap_k = vehicles[k - 1].capacity
            # Option 1: vehicle k unused
            for j in range(N + 1):
                V[k][j] = V[k - 1][j]
                P[k][j] = -2

            # Option 2: vehicle k serves customers i..j-1
            for i in range(N):
                if V[k - 1][i] == float("inf"):
                    continue

                load = 0.0
                route_cost = 0.0
                for j in range(i + 1, N + 1):
                    cust = perm[j - 1]
                    load += problem.customer_demands.get(cust, 1.0)

                    if j == i + 1:
                        route_cost = get_cost(origin, cust) + get_cost(cust, origin)
                    else:
                        prev_cust = perm[j - 2]
                        route_cost = route_cost - get_cost(prev_cust, origin) + get_cost(prev_cust, cust) + get_cost(cust, origin)

                    overload = max(0.0, load - cap_k)
                    penalized = route_cost + omega_cap * overload

                    if V[k - 1][i] + penalized < V[k][j]:
                        V[k][j] = V[k - 1][i] + penalized
                        P[k][j] = i

        # Reconstruct vehicle route partitions
        routes_assigned: List[List[str]] = [[] for _ in range(K)]
        curr_j = N
        curr_k = K
        while curr_k > 0 and curr_j > 0:
            prev_i = P[curr_k][curr_j]
            if prev_i == -2:
                curr_k -= 1
            elif prev_i >= 0:
                routes_assigned[curr_k - 1] = perm[prev_i:curr_j]
                curr_j = prev_i
                curr_k -= 1
            else:
                break

        # Fallback if DP could not partition all customers
        if curr_j > 0:
            routes_assigned = [[] for _ in range(K)]
            curr_v_idx = 0
            curr_load = 0.0
            for cust in perm:
                d = problem.customer_demands.get(cust, 1.0)
                if curr_v_idx < K and (curr_load + d <= vehicles[curr_v_idx].capacity or not routes_assigned[curr_v_idx]):
                    routes_assigned[curr_v_idx].append(cust)
                    curr_load += d
                else:
                    curr_v_idx += 1
                    if curr_v_idx < K:
                        routes_assigned[curr_v_idx].append(cust)
                        curr_load = d
                    else:
                        routes_assigned[-1].append(cust)

        # Compute exact routing cost and capacity overloads
        pure_cost = 0.0
        total_overload = 0.0
        for idx in range(K):
            r = routes_assigned[idx]
            if r:
                pure_cost += get_cost(origin, r[0]) + get_cost(r[-1], origin)
                for p in range(len(r) - 1):
                    pure_cost += get_cost(r[p], r[p + 1])
                v_load = sum(problem.customer_demands.get(c, 1.0) for c in r)
                total_overload += max(0.0, v_load - vehicles[idx].capacity)

        penalized_cost = pure_cost + omega_cap * total_overload
        is_feas = (total_overload <= 1e-4)

        return HGSIndividual(
            chromosome=perm[:],
            routes=routes_assigned,
            cost=pure_cost,
            excess_load=total_overload,
            penalized_cost=penalized_cost,
            is_feasible=is_feas,
        )

    def _calc_diversity(self, c1: List[str], c2: List[str]) -> float:
        """
        Computes broken-pairs distance delta(c1, c2):
        The fraction of adjacent customer pairs in c1 that do not appear adjacent in c2.
        """
        N = len(c1)
        if N <= 2:
            return 0.0

        pairs2 = set()
        for i in range(len(c2) - 1):
            pairs2.add((c2[i], c2[i + 1]))
            pairs2.add((c2[i + 1], c2[i]))

        broken = 0
        for i in range(len(c1) - 1):
            if (c1[i], c1[i + 1]) not in pairs2:
                broken += 1

        return broken / max(1, N - 1)

    def _update_biased_fitness(self, population: List[HGSIndividual], mu: int):
        """
        Updates biased fitness for all individuals in subpopulation:
        fit(S) = rank_cost(S) + (1.0 - mu / |P|) * rank_div(S)
        """
        pop_len = len(population)
        if pop_len == 0:
            return

        if pop_len == 1:
            population[0].biased_fitness = 1.0
            population[0].diversity_contribution = 1.0
            return

        # 1. Cost ranking
        population.sort(key=lambda ind: ind.penalized_cost)
        for rank, ind in enumerate(population):
            ind.cost_rank = rank + 1

        # 2. Average diversity contribution
        for i in range(pop_len):
            div_sum = 0.0
            for j in range(pop_len):
                if i != j:
                    div_sum += self._calc_diversity(population[i].chromosome, population[j].chromosome)
            population[i].diversity_contribution = div_sum / (pop_len - 1)

        # 3. Diversity ranking (higher diversity gets rank 1)
        div_sorted = sorted(range(pop_len), key=lambda idx: population[idx].diversity_contribution, reverse=True)
        div_ranks = [0] * pop_len
        for r, orig_idx in enumerate(div_sorted):
            div_ranks[orig_idx] = r + 1

        # 4. Biased fitness combination
        div_weight = max(0.0, 1.0 - (mu / pop_len))
        for i in range(pop_len):
            population[i].biased_fitness = population[i].cost_rank + div_weight * div_ranks[i]

    def _binary_tournament(self, pool: List[HGSIndividual], rng: random.Random) -> HGSIndividual:
        i1, i2 = rng.sample(range(len(pool)), 2)
        ind1 = pool[i1]
        ind2 = pool[i2]
        return ind1 if ind1.biased_fitness <= ind2.biased_fitness else ind2

    def _ordered_crossover(self, p1: List[str], p2: List[str], rng: random.Random) -> List[str]:
        """
        Applies Ordered Crossover (OX) to giant-tour chromosomes.
        Preserves relative customer ordering from p1 and circular sequence from p2.
        """
        N = len(p1)
        if N <= 2:
            return p1[:]

        cut1 = rng.randint(0, N - 2)
        cut2 = rng.randint(cut1 + 1, N - 1)

        offspring = [None] * N
        offspring[cut1:cut2 + 1] = p1[cut1:cut2 + 1]

        copied_set = set(offspring[cut1:cut2 + 1])

        p2_idx = (cut2 + 1) % N
        fill_idx = (cut2 + 1) % N

        for _ in range(N):
            cand = p2[p2_idx]
            if cand not in copied_set:
                offspring[fill_idx] = cand
                fill_idx = (fill_idx + 1) % N
            p2_idx = (p2_idx + 1) % N

        return [x for x in offspring if x is not None]

    def _local_search(
        self,
        perm: List[str],
        vehicles: List[VehicleConfig],
        problem: ProblemInstance,
        get_cost,
        omega_cap: float,
        rng: random.Random,
    ) -> List[str]:
        """
        Applies local search neighborhood operators (Relocate, Swap, 2-Opt) to giant tour.
        Moves are evaluated against penalized objective and accepted if improving.
        """
        N = len(perm)
        if N <= 2:
            return perm[:]

        current_perm = perm[:]
        current_ind = self._evaluate_and_split(current_perm, vehicles, problem, get_cost, omega_cap)
        best_penalized = current_ind.penalized_cost

        # Perform a bounded number of neighborhood trials
        max_trials = min(20, N * 2)
        for _ in range(max_trials):
            move_type = rng.choice(["relocate", "swap", "2opt"])
            cand_perm = current_perm[:]

            if move_type == "relocate":
                i = rng.randint(0, N - 1)
                j = rng.randint(0, N - 1)
                if i != j:
                    cust = cand_perm.pop(i)
                    cand_perm.insert(j, cust)

            elif move_type == "swap":
                i = rng.randint(0, N - 1)
                j = rng.randint(0, N - 1)
                if i != j:
                    cand_perm[i], cand_perm[j] = cand_perm[j], cand_perm[i]

            elif move_type == "2opt":
                i = rng.randint(0, N - 2)
                j = rng.randint(i + 1, N - 1)
                cand_perm[i:j + 1] = reversed(cand_perm[i:j + 1])

            # Evaluate move
            cand_ind = self._evaluate_and_split(cand_perm, vehicles, problem, get_cost, omega_cap)
            if cand_ind.penalized_cost < best_penalized - 1e-4:
                best_penalized = cand_ind.penalized_cost
                current_perm = cand_perm[:]

        return current_perm

