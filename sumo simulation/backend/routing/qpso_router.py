"""
Quantum-Behaved Particle Swarm Optimization (QPSO) Router module.
Formulated strictly according to the research paper:
"QUANTUM INSPIRED PARTICLE SWARM COMBINED WITH LIN-KERNIGHAN-HELSGAUN METHOD
TO THE TRAVELING SALESMAN PROBLEM"
(Herrera, Coelho, Steiner, Pesquisa Operacional 35(3), 2015).

Core Pipeline & Equations:
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
   Ascending sort order of continuous particle position vector x_i defines discrete node tour pi.
6. Fitness Evaluation:
   Sum of dynamic edge travel times / costs on SUMO graph G(t).
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from backend.graph.dynamic_graph import DynamicTrafficGraph


class QPSORouter:
    """
    QPSO Router implementing quantum-behaved particle swarm search on dynamic SUMO graph G(t).
    """

    def __init__(
        self,
        num_particles: int = 40,
        max_iter: int = 50,
        alpha_start: float = 1.0,
        alpha_end: float = 0.5,
        seed: int = 42,
    ):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.seed = seed

    def find_route(
        self,
        dt_graph: DynamicTrafficGraph,
        origin_node: str,
        destination_node: str,
        num_particles: Optional[int] = None,
        max_iter: Optional[int] = None,
        weight_key: str = "weight",
    ) -> Dict[str, Any]:
        """
        Executes QPSO route optimization on the SUMO dynamic graph G(t).
        """
        started = time.perf_counter()
        graph = dt_graph.graph

        if origin_node not in graph or destination_node not in graph:
            return self._failure("Origin or destination node not present in graph.", started)

        if origin_node == destination_node:
            return self._failure("Origin and destination must be distinct nodes.", started)

        M = num_particles or self.num_particles
        T_max = max_iter or self.max_iter
        rng = np.random.default_rng(self.seed)

        # 1. Extract local sub-graph nodes between origin and destination
        nodes = self._extract_candidate_nodes(graph, origin_node, destination_node, weight_key=weight_key)
        D = len(nodes)

        if D < 2:
            return self._failure("Could not extract candidate nodes for QPSO.", started)

        node_to_idx = {n: i for i, n in enumerate(nodes)}
        origin_idx = node_to_idx[origin_node]
        dest_idx = node_to_idx[destination_node]

        # 2. Initialize continuous particle swarm population M x D
        # x_i ~ Uniform(-2.0, 2.0)
        x = rng.uniform(-2.0, 2.0, size=(M, D))
        pbest = x.copy()
        pbest_fitness = np.full(M, float("inf"))

        # Evaluate initial population fitness
        for i in range(M):
            pbest_fitness[i] = self._evaluate_fitness(x[i], nodes, dt_graph, weight_key)

        gbest_idx = int(np.argmin(pbest_fitness))
        gbest = pbest[gbest_idx].copy()
        gbest_fitness = float(pbest_fitness[gbest_idx])

        fitness_history = [gbest_fitness]

        # 3. Main QPSO Swarm Iteration Loop (Herrera et al., Algorithm 1)
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
                current_fitness = self._evaluate_fitness(x[i], nodes, dt_graph, weight_key)

                # Update Personal Best pbest_i
                if current_fitness < pbest_fitness[i]:
                    pbest[i] = x[i].copy()
                    pbest_fitness[i] = current_fitness

                    # Update Global Best gbest
                    if current_fitness < gbest_fitness:
                        gbest = x[i].copy()
                        gbest_fitness = current_fitness

            fitness_history.append(gbest_fitness)

        # 4. Decode gbest into discrete node tour using Paper Sec 3.3 Rank Discretization
        best_node_tour = self._discretize_to_node_tour(gbest, nodes, origin_node, destination_node)

        # 5. Extract SUMO route metrics
        route_metrics = self._extract_route_metrics(dt_graph, best_node_tour)
        elapsed_ms = (time.perf_counter() - started) * 1000

        rank_discretization_table = []
        for d_idx, node_id in enumerate(nodes):
            rank_discretization_table.append({
                "dimension_index": d_idx,
                "node": node_id,
                "gbest_value": round(float(gbest[d_idx]), 4),
                "mbest_value": round(float(mbest[d_idx]), 4),
                "is_terminal": node_id in (origin_node, destination_node)
            })

        swarm_details = {
            "particles_M": M,
            "iterations_T": T_max,
            "dimensions_D": D,
            "candidate_nodes": nodes,
            "gbest_vector": [round(float(v), 4) for v in gbest],
            "mbest_vector": [round(float(v), 4) for v in mbest],
            "alpha_decay": {
                "alpha_start": self.alpha_start,
                "alpha_end": self.alpha_end,
                "final_alpha": round(alpha_t, 4)
            },
            "rank_discretization_table": rank_discretization_table,
            "discretized_tour": best_node_tour
        }

        math_proof = {
            "paper_reference": "Herrera, Coelho, Steiner (Pesquisa Operacional 35(3), 2015 / pp. 1-20)",
            "position_update_eq": "x_{i,d}(t+1) = p_{i,d} +/- alpha(t) * |mbest_d - x_{i,d}(t)| * ln(1/u)",
            "lip_eq": "p_{i,d} = (fi_1 * pbest_{i,d} + fi_2 * gbest_d) / (fi_1 + fi_2)",
            "mbest_eq": "mbest_d = (1 / M) * sum_{i=1}^M pbest_{i,d}",
            "alpha_decay_eq": "alpha(t) = alpha_start - (t / (T_max - 1)) * (alpha_start - alpha_end)",
            "rank_discretization_rule": "Ascending sort of continuous values x_particle defines discrete node sequence pi (Sec 3.3, pp. 14-15)",
        }

        return {
            "success": True,
            "algorithm": "QPSO (Herrera et al. 2015)",
            "origin_node": origin_node,
            "destination_node": destination_node,
            **route_metrics,
            "qpso_particles": M,
            "qpso_iterations": T_max,
            "alpha_start": self.alpha_start,
            "alpha_end": self.alpha_end,
            "final_alpha": round(alpha_t, 4),
            "best_fitness_cost": round(gbest_fitness, 3),
            "candidate_subgraph_nodes": D,
            "swarm_details": swarm_details,
            "math_proof": math_proof,
            "computation_time_ms": round(elapsed_ms, 3),
        }

    def _extract_candidate_nodes(
        self,
        graph: nx.DiGraph,
        origin_node: str,
        destination_node: str,
        weight_key: str = "weight",
        max_nodes: int = 12,
    ) -> List[str]:
        """
        Extracts candidate nodes for QPSO search space.
        """
        try:
            shortest_path = nx.dijkstra_path(graph, origin_node, destination_node, weight=weight_key)
        except nx.NetworkXNoPath:
            return []

        nodes_set = list(shortest_path)

        # Add 1-hop neighbor nodes to allow swarm exploration
        for n in list(nodes_set):
            if len(nodes_set) >= max_nodes:
                break
            for nbr in graph.neighbors(n):
                if nbr not in nodes_set and graph.has_edge(nbr, destination_node):
                    nodes_set.append(nbr)
                    if len(nodes_set) >= max_nodes:
                        break

        # Ensure origin is at start and destination at end
        nodes = [origin_node] + [n for n in nodes_set if n not in (origin_node, destination_node)] + [destination_node]
        return list(dict.fromkeys(nodes))

    def _discretize_to_node_tour(
        self,
        x_particle: np.ndarray,
        nodes: List[str],
        origin_node: str,
        destination_node: str,
    ) -> List[str]:
        """
        Applies Paper Sec 3.3 (Page 14-15) Rank Discretization Rule:
        Ascending sort order of continuous values x_particle defines node tour sequence.
        """
        D = len(nodes)
        if D <= 2:
            return [origin_node, destination_node]

        # Intermediate nodes exclude origin (0) and destination (D-1)
        intermediate_indices = list(range(1, D - 1))
        # Sort intermediate node indices based on continuous particle x values
        sorted_intermediates = sorted(intermediate_indices, key=lambda idx: float(x_particle[idx]))

        tour_indices = [0] + sorted_intermediates + [D - 1]
        return [nodes[i] for i in tour_indices]

    def _evaluate_fitness(
        self,
        x_particle: np.ndarray,
        nodes: List[str],
        dt_graph: DynamicTrafficGraph,
        weight_key: str = "weight",
    ) -> float:
        """
        Evaluates dynamic route travel cost for discretized node tour on G(t).
        """
        tour = self._discretize_to_node_tour(x_particle, nodes, nodes[0], nodes[-1])
        graph = dt_graph.graph
        total_cost = 0.0

        for u, v in zip(tour, tour[1:]):
            if graph.has_edge(u, v):
                total_cost += float(graph[u][v].get(weight_key, 10.0))
            else:
                try:
                    total_cost += float(nx.dijkstra_path_length(graph, u, v, weight=weight_key))
                except nx.NetworkXNoPath:
                    total_cost += 1000.0  # Disconnect penalty

        return total_cost

    def _extract_route_metrics(
        self, dt_graph: DynamicTrafficGraph, node_path: Sequence[str]
    ) -> Dict[str, Any]:
        """
        Extracts SUMO route metrics (travel time, distance, speed, bottlenecks, geometry).
        """
        graph = dt_graph.graph
        edge_path = []
        geometry = []
        total_travel_time = 0.0
        total_distance = 0.0
        total_cost = 0.0
        bottlenecks = []

        init_pos = dt_graph.node_positions.get(node_path[0])
        if init_pos:
            geometry.append(init_pos)

        segment_calculations = []
        for u, v in zip(node_path, node_path[1:]):
            if graph.has_edge(u, v):
                edge_attrs = graph[u][v]
                edge_id = edge_attrs.get("key", edge_attrs.get("edge_id", ""))
                edge_path.append(edge_id)
                edge_data = dt_graph.get_edge_data(edge_id) or edge_attrs
                length = float(edge_data.get("length", 0.0))
                tt = float(edge_data.get("travel_time", edge_data.get("free_flow_tt", 0.0)))
                speed = float(edge_data.get("speed_limit", edge_data.get("mean_speed", 13.89)))
                w = float(edge_data.get("weight", tt))
                cong = float(edge_data.get("congestion_ratio", 0.0))

                total_distance += length
                total_travel_time += tt
                total_cost += w

                segment_calculations.append({
                    "edge_id": edge_id,
                    "from_node": u,
                    "to_node": v,
                    "length_m": round(length, 2),
                    "speed_limit_ms": round(speed, 2),
                    "congestion_ratio": round(cong, 3),
                    "travel_time_s": round(tt, 2),
                    "accumulated_distance_m": round(total_distance, 2),
                    "accumulated_time_s": round(total_travel_time, 2),
                })

                if cong > 0.3 or edge_id in dt_graph.incidents:
                    bottlenecks.append({
                        "edge_id": edge_id,
                        "congestion_ratio": cong,
                        "is_incident": edge_id in dt_graph.incidents
                    })

                for p in edge_data.get("shape", []):
                    pt = (float(p[0]), float(p[1]))
                    if not geometry or geometry[-1] != pt:
                        geometry.append(pt)
            else:
                try:
                    sub_nodes = nx.dijkstra_path(graph, u, v, weight="weight")
                    for s_u, s_v in zip(sub_nodes, sub_nodes[1:]):
                        e_attrs = graph[s_u][s_v]
                        e_id = e_attrs.get("key", "")
                        edge_path.append(e_id)
                        e_data = dt_graph.get_edge_data(e_id) or e_attrs
                        length = float(e_data.get("length", 0.0))
                        tt = float(e_data.get("travel_time", 0.0))
                        speed = float(e_data.get("speed_limit", e_data.get("mean_speed", 13.89)))
                        w = float(e_data.get("weight", tt))
                        cong = float(e_data.get("congestion_ratio", 0.0))
                        total_distance += length
                        total_travel_time += tt
                        total_cost += w

                        segment_calculations.append({
                            "edge_id": e_id,
                            "from_node": s_u,
                            "to_node": s_v,
                            "length_m": round(length, 2),
                            "speed_limit_ms": round(speed, 2),
                            "congestion_ratio": round(cong, 3),
                            "travel_time_s": round(tt, 2),
                            "accumulated_distance_m": round(total_distance, 2),
                            "accumulated_time_s": round(total_travel_time, 2),
                        })

                        for p in e_data.get("shape", []):
                            pt = (float(p[0]), float(p[1]))
                            if not geometry or geometry[-1] != pt:
                                geometry.append(pt)
                except nx.NetworkXNoPath:
                    pass

        avg_speed = total_distance / max(total_travel_time, 0.1)

        return {
            "node_path": list(node_path),
            "edge_path": edge_path,
            "total_cost": round(total_cost, 2),
            "total_travel_time": round(total_travel_time, 2),
            "total_distance": round(total_distance, 2),
            "average_speed_ms": round(avg_speed, 2),
            "bottleneck_count": len(bottlenecks),
            "bottlenecks": bottlenecks,
            "segment_calculations": segment_calculations,
            "geometry": geometry,
        }

    @staticmethod
    def _failure(error_msg: str, started: float) -> Dict[str, Any]:
        return {
            "success": False,
            "algorithm": "QPSO",
            "error": error_msg,
            "computation_time_ms": round((time.perf_counter() - started) * 1000, 3),
        }
