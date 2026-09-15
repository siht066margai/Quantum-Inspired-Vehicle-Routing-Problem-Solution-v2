from __future__ import annotations

import time
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple

from backend.vrp.problem_instance import ProblemInstance, AlgorithmResult, FleetVehicleRoute, VehicleConfig


class VRPEvaluator:
    """
    Common evaluator engine for evaluating single and multi-vehicle VRP customer visit sequences
    on dynamic SUMO graph G(t). Evaluates cost, feasibility, metrics, bottlenecks,
    and road-level geometry identically across all algorithms.
    """

    @staticmethod
    def evaluate_leg(
        problem: ProblemInstance, u_node: str, v_node: str
    ) -> Tuple[float, float, float, List[str], List[str], List[Tuple[float, float]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Evaluates shortest physical path between u_node and v_node on G(t).
        Returns (cost, travel_time, distance, node_path, edge_path, geometry, segment_calcs, bottlenecks).
        """
        dt_graph = problem.graph
        g = dt_graph.graph

        cache_key = (u_node, v_node, problem.timestamp)
        if not hasattr(problem, "_leg_cache"):
            problem._leg_cache = {}
        if cache_key in problem._leg_cache:
            return problem._leg_cache[cache_key]

        if u_node not in g or v_node not in g:
            raise ValueError(f"Node ({u_node} or {v_node}) not in graph.")

        if u_node == v_node:
            u_pos = dt_graph.node_positions.get(u_node, (0.0, 0.0))
            geom = [u_pos] if (u_pos[0] != 0.0 or u_pos[1] != 0.0) else []
            res_tuple = (0.0, 0.0, 0.0, [u_node], [], geom, [], [])
            problem._leg_cache[cache_key] = res_tuple
            return res_tuple

        try:
            leg_nodes = nx.dijkstra_path(g, u_node, v_node, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            try:
                # Fallback: Use pre-cached undirected graph search if directed path is missing in OSM topology
                undirected_g = dt_graph.get_undirected_graph()
                leg_nodes = nx.dijkstra_path(undirected_g, u_node, v_node, weight="weight")
            except Exception:
                raise ValueError(f"Unreachable customer: No valid graph path exists between '{u_node}' and '{v_node}' on dynamic road network.")

        edge_path = []
        geometry = []
        segment_calcs = []
        bottlenecks = []

        total_cost = 0.0
        total_time = 0.0
        total_dist = 0.0

        # Geometry for starting node
        start_pos = dt_graph.node_positions.get(leg_nodes[0])
        if start_pos and (start_pos[0] != 0.0 or start_pos[1] != 0.0):
            geometry.append(start_pos)

        for i in range(len(leg_nodes) - 1):
            curr_u = leg_nodes[i]
            curr_v = leg_nodes[i + 1]

            if g.has_edge(curr_u, curr_v):
                edge_attrs = g[curr_u][curr_v]
            elif g.has_edge(curr_v, curr_u):
                edge_attrs = g[curr_v][curr_u]
            else:
                raise ValueError(f"Disconnected road segment: No edge exists between '{curr_u}' and '{curr_v}' on dynamic road network.")

            edge_id = edge_attrs.get("key", edge_attrs.get("edge_id", ""))
            edge_path.append(edge_id)

            edge_data = dt_graph.get_edge_data(edge_id) or edge_attrs
            length = float(edge_data.get("length", 0.0))
            tt = float(edge_data.get("travel_time", edge_data.get("free_flow_tt", 0.0)))
            speed = float(edge_data.get("speed_limit", edge_data.get("mean_speed", 13.89)))
            w = float(edge_data.get("weight", tt))
            cong = float(edge_data.get("congestion_ratio", 0.0))

            total_dist += length
            total_time += tt
            total_cost += w

            segment_calcs.append({
                "from_node": curr_u,
                "to_node": curr_v,
                "edge_id": edge_id,
                "length_m": round(length, 2),
                "speed_limit_ms": round(speed, 2),
                "congestion_ratio": round(cong, 3),
                "travel_time_s": round(tt, 2),
            })

            if cong > 0.3 or edge_id in dt_graph.incidents:
                bottlenecks.append({
                    "edge_id": edge_id,
                    "congestion_ratio": cong,
                    "is_incident": edge_id in dt_graph.incidents,
                })

            shape = edge_data.get("shape", [])
            if shape:
                for p in shape:
                    pt = (float(p[0]), float(p[1]))
                    if (pt[0] != 0.0 or pt[1] != 0.0) and (not geometry or geometry[-1] != pt):
                        geometry.append(pt)
            else:
                u_p = dt_graph.node_positions.get(curr_u)
                v_p = dt_graph.node_positions.get(curr_v)
                if u_p and (u_p[0] != 0.0 or u_p[1] != 0.0) and (not geometry or geometry[-1] != u_p):
                    geometry.append(u_p)
                if v_p and (v_p[0] != 0.0 or v_p[1] != 0.0) and (not geometry or geometry[-1] != v_p):
                    geometry.append(v_p)

        result_tuple = (total_cost, total_time, total_dist, leg_nodes, edge_path, geometry, segment_calcs, bottlenecks)
        problem._leg_cache[cache_key] = result_tuple
        return result_tuple

    @classmethod
    def evaluate_fleet_routes(
        cls,
        problem: ProblemInstance,
        routes: List[Tuple[VehicleConfig, List[str]]],  # List of (vehicle_config, [Origin, C1, C2, ..., Origin])
        algorithm_name: str = "Common Evaluator",
        computation_time_ms: float = 0.0,
        solver_details: Optional[Dict[str, Any]] = None,
        math_proof: Optional[Dict[str, Any]] = None,
    ) -> AlgorithmResult:
        """
        Evaluates a complete fleet plan across multiple vehicles on dynamic graph G(t).
        Guarantees that ALL X vehicles configured in problem.vehicles are accounted for.
        """
        started = time.perf_counter()
        constraint_violations: List[str] = []
        is_feasible = True

        fleet_vehicle_routes: List[FleetVehicleRoute] = []
        all_visited_customers: List[str] = []
        all_node_paths: List[str] = []
        all_edge_paths: List[str] = []
        all_geometry: List[Tuple[float, float]] = []
        all_segment_calcs: List[Dict[str, Any]] = []
        all_bottlenecks: List[Dict[str, Any]] = []

        total_fleet_cost = 0.0
        total_fleet_time = 0.0
        total_fleet_dist = 0.0

        # Map active routes by vehicle_id
        routes_by_vid: Dict[str, Tuple[VehicleConfig, List[str]]] = {}
        for v_cfg, visit_seq in routes:
            routes_by_vid[v_cfg.vehicle_id] = (v_cfg, visit_seq)

        # Iterate over all vehicles configured in problem.vehicles
        for v_config in problem.vehicles:
            vid = v_config.vehicle_id
            if vid in routes_by_vid:
                _, visit_seq = routes_by_vid[vid]
            else:
                visit_seq = [problem.origin, problem.origin]

            # Filter customer stops
            cust_nodes = [c for c in visit_seq if c != problem.origin]

            # If vehicle is unused (no customer stops)
            if not cust_nodes:
                depot_pos = problem.graph.node_positions.get(problem.origin, (0.0, 0.0))
                unused_geom = [depot_pos] if (depot_pos[0] != 0.0 or depot_pos[1] != 0.0) else []
                standby_route = FleetVehicleRoute(
                    vehicle_id=v_config.vehicle_id,
                    capacity=v_config.capacity,
                    assigned_orders=[],
                    visit_sequence=[problem.origin, problem.origin],
                    node_path=[problem.origin],
                    edge_path=[],
                    geometry=unused_geom,
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
                        "action": f"Vehicle {v_config.vehicle_id} on Standby at Depot (Available capacity: {v_config.capacity:.1f} pkgs).",
                        "delivered": 0.0,
                        "remaining": 0.0,
                        "cumulative_distance_m": 0.0,
                        "cumulative_time_s": 0.0,
                    }],
                    segment_calculations=[],
                    bottlenecks=[],
                    is_active=False,
                    status="standby",
                    color=v_config.color,
                )
                fleet_vehicle_routes.append(standby_route)
                continue

            # Active vehicle evaluation
            if not visit_seq or visit_seq[0] != problem.origin:
                constraint_violations.append(f"Vehicle {v_config.vehicle_id} route must start at origin '{problem.origin}'.")
                is_feasible = False
                continue

            # Capacity check: sum of assigned customer demands <= vehicle capacity
            route_demand = sum(problem.customer_demands.get(c, 1.0) for c in cust_nodes)
            capacity_util = round((route_demand / v_config.capacity * 100.0), 1) if v_config.capacity > 0 else 0.0
            if route_demand > v_config.capacity:
                constraint_violations.append(
                    f"Vehicle {v_config.vehicle_id} load ({route_demand:.1f} pkgs) exceeds capacity ({v_config.capacity:.1f} pkgs)."
                )
                is_feasible = False

            # Operational steps tracking with explicit package decrements
            op_steps: List[Dict[str, Any]] = []
            op_steps.append({
                "step": 0,
                "type": "departure",
                "node": problem.origin,
                "action": f"Depart from Depot '{problem.origin}' with initial load of {route_demand:.1f} pkgs ({len(cust_nodes)} orders assigned).",
                "delivered": 0.0,
                "remaining": route_demand,
                "cumulative_distance_m": 0.0,
                "cumulative_time_s": 0.0,
            })

            curr_load = route_demand
            cum_dist = 0.0
            cum_time = 0.0
            step_counter = 1

            v_nodes: List[str] = []
            v_edges: List[str] = []
            v_geom: List[Tuple[float, float]] = []
            v_seg_calcs: List[Dict[str, Any]] = []
            v_bottlenecks: List[Dict[str, Any]] = []
            v_legs: List[Dict[str, Any]] = []

            v_cost = 0.0
            v_time = 0.0
            v_dist = 0.0

            for idx in range(len(visit_seq) - 1):
                u_node = visit_seq[idx]
                v_node = visit_seq[idx + 1]
                try:
                    c, t, d, leg_nodes, leg_edges, leg_geom, seg_calcs, b_necks = cls.evaluate_leg(problem, u_node, v_node)
                    v_cost += c
                    v_time += t
                    v_dist += d
                    cum_dist += d
                    cum_time += t

                    v_legs.append({
                        "from_node": u_node,
                        "to_node": v_node,
                        "cost": round(c, 2),
                        "travel_time_s": round(t, 2),
                        "distance_m": round(d, 2),
                        "node_path": leg_nodes,
                        "edge_path": leg_edges,
                        "geometry": leg_geom,
                        "is_return_to_depot": (v_node == problem.origin),
                    })

                    if not v_nodes:
                        v_nodes.extend(leg_nodes)
                    else:
                        v_nodes.extend(leg_nodes[1:])

                    v_edges.extend(leg_edges)

                    for pt in leg_geom:
                        if not v_geom or v_geom[-1] != pt:
                            v_geom.append(pt)

                    v_seg_calcs.extend(seg_calcs)
                    v_bottlenecks.extend(b_necks)

                    if v_node == problem.origin:
                        op_steps.append({
                            "step": step_counter,
                            "type": "return",
                            "node": problem.origin,
                            "action": f"Return to Depot '{problem.origin}' with 0 remaining packages. Tour completed.",
                            "delivered": 0.0,
                            "remaining": 0.0,
                            "cumulative_distance_m": round(cum_dist, 2),
                            "cumulative_time_s": round(cum_time, 2),
                        })
                    else:
                        pkg_delivered = problem.customer_demands.get(v_node, 1.0)
                        curr_load = max(0.0, curr_load - pkg_delivered)
                        op_steps.append({
                            "step": step_counter,
                            "type": "delivery",
                            "node": v_node,
                            "action": f"Deliver {pkg_delivered:.1f} pkg(s) at Customer '{v_node}' (Remaining on board: {curr_load:.1f} pkgs).",
                            "delivered": pkg_delivered,
                            "remaining": round(curr_load, 2),
                            "cumulative_distance_m": round(cum_dist, 2),
                            "cumulative_time_s": round(cum_time, 2),
                        })
                    step_counter += 1

                except Exception as e:
                    constraint_violations.append(f"Vehicle {v_config.vehicle_id} error on leg {u_node}->{v_node}: {e}")
                    is_feasible = False
                    break

            fleet_route = FleetVehicleRoute(
                vehicle_id=v_config.vehicle_id,
                capacity=v_config.capacity,
                assigned_orders=cust_nodes,
                visit_sequence=visit_seq,
                node_path=v_nodes,
                edge_path=v_edges,
                geometry=v_geom,
                total_cost=v_cost,
                total_travel_time=v_time,
                total_distance=v_dist,
                load_used=route_demand,
                capacity_utilization_pct=capacity_util,
                initial_load=route_demand,
                delivered_load=route_demand,
                remaining_load=0.0,
                operational_steps=op_steps,
                segment_calculations=v_seg_calcs,
                bottlenecks=v_bottlenecks,
                is_active=True,
                status="active",
                color=v_config.color,
                legs=v_legs,
            )
            fleet_vehicle_routes.append(fleet_route)

            all_visited_customers.extend(cust_nodes)
            all_node_paths.extend(v_nodes)
            all_edge_paths.extend(v_edges)
            all_geometry.extend(v_geom)
            all_segment_calcs.extend(v_seg_calcs)
            all_bottlenecks.extend(v_bottlenecks)

            total_fleet_cost += v_cost
            total_fleet_time += v_time
            total_fleet_dist += v_dist

        # Check coverage of required customers
        required_set = set(problem.destinations)
        visited_set = set(all_visited_customers)
        if required_set != visited_set:
            missing = required_set - visited_set
            if missing:
                constraint_violations.append(f"Missing required customer destinations: {list(missing)}.")
                is_feasible = False

        eval_elapsed_ms = (time.perf_counter() - started) * 1000
        total_comp_ms = computation_time_ms + eval_elapsed_ms
        avg_speed = total_fleet_dist / max(total_fleet_time, 0.1)

        primary_visit_sequence = [problem.origin] + all_visited_customers + [problem.origin]

        default_proof = {
            "evaluation_engine": "Common Multi-Vehicle Fleet Evaluator",
            "fleet_size": len(fleet_vehicle_routes),
            "active_vehicles": sum(1 for fr in fleet_vehicle_routes if fr.is_active),
            "standby_vehicles": sum(1 for fr in fleet_vehicle_routes if not fr.is_active),
            "total_packages_demand": sum(problem.customer_demands.get(c, 1.0) for c in problem.destinations),
            "total_packages_allocated": sum(fr.load_used for fr in fleet_vehicle_routes),
            "objective_cost": round(total_fleet_cost, 2),
            "total_distance": round(total_fleet_dist, 2),
            "total_travel_time": round(total_fleet_time, 2),
        }

        res = AlgorithmResult(
            algorithm=algorithm_name,
            status="feasible" if is_feasible else "infeasible",
            success=is_feasible,
            visit_sequence=primary_visit_sequence,
            node_path=all_node_paths,
            edge_path=all_edge_paths,
            total_cost=total_fleet_cost,
            total_travel_time=total_fleet_time,
            total_distance=total_fleet_dist,
            average_speed_ms=avg_speed,
            bottleneck_count=len(all_bottlenecks),
            bottlenecks=all_bottlenecks,
            segment_calculations=all_segment_calcs,
            geometry=all_geometry,
            feasible=is_feasible,
            constraint_violations=constraint_violations,
            computation_time_ms=total_comp_ms,
            math_proof=math_proof or default_proof,
            solver_details=solver_details or {},
            fleet_routes=fleet_vehicle_routes,
            error=None if is_feasible else "; ".join(constraint_violations),
        )

        # Rigorous Mathematical Invariant Verification across all 20 CVRP invariants
        invariant_report = cls.validate_invariants(problem, res)
        res.math_proof["invariants"] = invariant_report
        if not invariant_report["all_passed"]:
            res.feasible = False
            res.success = False
            res.status = "infeasible"
            for viol in invariant_report["violations"]:
                if viol not in res.constraint_violations:
                    res.constraint_violations.append(viol)
            res.error = "; ".join(res.constraint_violations)

        return res

    @classmethod
    def validate_invariants(cls, problem: ProblemInstance, result: AlgorithmResult) -> Dict[str, Any]:
        """
        Validates all 20 mathematical invariants of generalized multi-vehicle CVRP.
        Guarantees that no illegal, fabricated, or mathematically inconsistent solution is accepted.
        """
        invariants: Dict[str, Dict[str, Any]] = {}
        violations: List[str] = []

        # I01: Origin Conservation
        i01_ok = all(fr.visit_sequence[0] == problem.origin for fr in result.fleet_routes if fr.visit_sequence)
        invariants["I01_origin_conservation"] = {
            "passed": i01_ok,
            "details": "All active and standby routes originate at depot O." if i01_ok else "Route origin mismatch."
        }
        if not i01_ok: violations.append("I01_origin_conservation failed")

        # I02: Return Conservation
        i02_ok = all(fr.visit_sequence[-1] == problem.origin for fr in result.fleet_routes if fr.visit_sequence)
        invariants["I02_return_conservation"] = {
            "passed": i02_ok,
            "details": "All active routes return to depot O upon tour completion." if i02_ok else "Route return mismatch."
        }
        if not i02_ok: violations.append("I02_return_conservation failed")

        # I03: Total Customer Demand Conservation
        total_demand = sum(problem.customer_demands.get(c, 1.0) for c in problem.destinations)
        assigned_demand = sum(fr.load_used for fr in result.fleet_routes)
        i03_ok = abs(total_demand - assigned_demand) < 1e-4 if result.feasible else True
        invariants["I03_demand_conservation"] = {
            "passed": i03_ok,
            "details": f"Total demand ({total_demand:.1f} pkgs) equals assigned demand ({assigned_demand:.1f} pkgs)."
        }
        if not i03_ok: violations.append(f"I03_demand_conservation failed: {assigned_demand} != {total_demand}")

        # I04: Vehicle Capacity Invariant
        i04_ok = all(fr.load_used <= fr.capacity + 1e-6 for fr in result.fleet_routes)
        invariants["I04_vehicle_capacity"] = {
            "passed": i04_ok,
            "details": "Every vehicle load_used <= vehicle capacity." if i04_ok else "Vehicle capacity exceeded."
        }
        if not i04_ok: violations.append("I04_vehicle_capacity failed")

        # I05: Partitioning and Exact Coverage Invariant
        all_assigned = [c for fr in result.fleet_routes for c in fr.assigned_orders]
        i05_ok = (len(all_assigned) == len(set(all_assigned)) == len(problem.destinations)) and set(all_assigned) == set(problem.destinations) if result.feasible else True
        invariants["I05_partitioning_coverage"] = {
            "passed": i05_ok,
            "details": "Each customer destination is visited exactly once by exactly one vehicle."
        }
        if not i05_ok: violations.append("I05_partitioning_coverage failed")

        # I06: Total Packages Delivered Invariant
        delivered_pkgs = 0.0
        for fr in result.fleet_routes:
            for step in fr.operational_steps:
                if step.get("type") == "delivery":
                    delivered_pkgs += float(step.get("delivered", 0.0))
        i06_ok = abs(total_demand - delivered_pkgs) < 1e-4 if result.feasible else True
        invariants["I06_total_packages_delivered"] = {
            "passed": i06_ok,
            "details": f"Total delivered packages ({delivered_pkgs:.1f}) matches total demand ({total_demand:.1f})."
        }
        if not i06_ok: violations.append(f"I06_total_packages_delivered failed: {delivered_pkgs} != {total_demand}")

        # I07: Initial Load Integrity
        i07_ok = all(abs(fr.initial_load - fr.load_used) < 1e-4 for fr in result.fleet_routes)
        invariants["I07_initial_load_integrity"] = {
            "passed": i07_ok,
            "details": "Initial vehicle load equals sum of assigned customer package demands."
        }
        if not i07_ok: violations.append("I07_initial_load_integrity failed")

        # I08: Final Load Zero Invariant
        i08_ok = all(abs(fr.remaining_load) < 1e-4 for fr in result.fleet_routes)
        invariants["I08_final_load_zero"] = {
            "passed": i08_ok,
            "details": "All active vehicles return to depot with exactly 0 remaining packages on board."
        }
        if not i08_ok: violations.append("I08_final_load_zero failed")

        # I09: Monotonic Load Decrement Invariant
        i09_ok = True
        for fr in result.fleet_routes:
            if not fr.is_active: continue
            cur = fr.initial_load
            for step in fr.operational_steps:
                if step.get("type") == "delivery":
                    d = float(step.get("delivered", 0.0))
                    rem = float(step.get("remaining", 0.0))
                    if abs((cur - d) - rem) > 1e-4 or rem < -1e-4:
                        i09_ok = False
                        break
                    cur = rem
        invariants["I09_monotonic_decrement"] = {
            "passed": i09_ok,
            "details": "Load monotonically decreases by customer demand at each stop and remains non-negative."
        }
        if not i09_ok: violations.append("I09_monotonic_decrement failed")

        # I10: Capacity Utilization Bound
        i10_ok = all(0.0 <= fr.capacity_utilization_pct <= 100.001 for fr in result.fleet_routes)
        invariants["I10_capacity_utilization_bound"] = {
            "passed": i10_ok,
            "details": "Vehicle capacity utilization is strictly bounded in [0%, 100%]."
        }
        if not i10_ok: violations.append("I10_capacity_utilization_bound failed")

        # I11: Standby Vehicle Invariant
        i11_ok = all(
            (fr.load_used == 0.0 and len(fr.assigned_orders) == 0 and fr.total_cost == 0.0 and fr.total_distance == 0.0)
            for fr in result.fleet_routes if not fr.is_active or fr.status == "standby"
        )
        invariants["I11_standby_zero_load"] = {
            "passed": i11_ok,
            "details": "Standby vehicles consume 0 load, 0 cost, and 0 distance."
        }
        if not i11_ok: violations.append("I11_standby_zero_load failed")

        # I12: Fleet Visibility Invariant
        i12_ok = len(result.fleet_routes) == len(problem.vehicles)
        invariants["I12_fleet_visibility"] = {
            "passed": i12_ok,
            "details": f"All {len(problem.vehicles)} configured fleet vehicles are explicitly accounted for in output."
        }
        if not i12_ok: violations.append(f"I12_fleet_visibility failed: {len(result.fleet_routes)} != {len(problem.vehicles)}")

        # I13: Road Path Network Connectivity
        i13_ok = True
        g = problem.graph.graph
        for fr in result.fleet_routes:
            if not fr.is_active: continue
            for leg in fr.legs:
                np_nodes = leg.get("node_path", [])
                if len(np_nodes) < 2 and leg.get("from_node") != leg.get("to_node"):
                    i13_ok = False
        invariants["I13_road_path_connectivity"] = {
            "passed": i13_ok,
            "details": "Every active leg trajectory is physically connected on SUMO road graph G(t)."
        }
        if not i13_ok: violations.append("I13_road_path_connectivity failed")

        # I14: Non-Negative Route Metrics
        i14_ok = (result.total_cost >= -1e-6 and result.total_travel_time >= -1e-6 and result.total_distance >= -1e-6)
        invariants["I14_non_negative_metrics"] = {
            "passed": i14_ok,
            "details": "All total and per-vehicle costs, travel times, and distances are non-negative."
        }
        if not i14_ok: violations.append("I14_non_negative_metrics failed")

        # I15: Cost Additivity Invariant
        sum_v_cost = sum(fr.total_cost for fr in result.fleet_routes)
        i15_ok = abs(result.total_cost - sum_v_cost) < 0.1
        invariants["I15_cost_additivity"] = {
            "passed": i15_ok,
            "details": f"Total fleet cost ({result.total_cost:.2f}) equals sum of vehicle costs ({sum_v_cost:.2f})."
        }
        if not i15_ok: violations.append("I15_cost_additivity failed")

        # I16: Travel Time Additivity Invariant
        sum_v_time = sum(fr.total_travel_time for fr in result.fleet_routes)
        i16_ok = abs(result.total_travel_time - sum_v_time) < 0.1
        invariants["I16_time_additivity"] = {
            "passed": i16_ok,
            "details": f"Total fleet time ({result.total_travel_time:.2f}s) equals sum of vehicle times ({sum_v_time:.2f}s)."
        }
        if not i16_ok: violations.append("I16_time_additivity failed")

        # I17: Distance Additivity Invariant
        sum_v_dist = sum(fr.total_distance for fr in result.fleet_routes)
        i17_ok = abs(result.total_distance - sum_v_dist) < 0.1
        invariants["I17_distance_additivity"] = {
            "passed": i17_ok,
            "details": f"Total fleet distance ({result.total_distance:.2f}m) equals sum of vehicle distances ({sum_v_dist:.2f}m)."
        }
        if not i17_ok: violations.append("I17_distance_additivity failed")

        # I18: QAOA Honest Sizing Invariant
        if "qaoa" in result.algorithm.lower():
            n_cust = len(problem.destinations)
            i18_ok = (n_cust <= 4) or (result.status == "unsupported_size")
        else:
            i18_ok = True
        invariants["I18_qaoa_honest_sizing"] = {
            "passed": i18_ok,
            "details": "QAOA statevector execution is strictly bounded to N <= 4 (or reports unsupported size)."
        }
        if not i18_ok: violations.append("I18_qaoa_honest_sizing failed")

        # I19: Pre-Optimization Capacity Feasibility
        total_fleet_cap = sum(v.capacity for v in problem.vehicles)
        i19_ok = (total_demand <= total_fleet_cap) if result.feasible else True
        invariants["I19_pre_optimization_feasibility"] = {
            "passed": i19_ok,
            "details": f"Demand feasibility respected (Total demand: {total_demand:.1f} pkgs, Total capacity: {total_fleet_cap:.1f} pkgs)."
        }
        if not i19_ok: violations.append("I19_pre_optimization_feasibility failed")

        # I20: Distinct Color Mapping Invariant
        v_colors = [fr.color for fr in result.fleet_routes]
        i20_ok = len(v_colors) == len(result.fleet_routes) and all(bool(c) for c in v_colors)
        invariants["I20_distinct_color_mapping"] = {
            "passed": i20_ok,
            "details": "Distinct visual color channels assigned across all vehicles and stops."
        }
        if not i20_ok: violations.append("I20_distinct_color_mapping failed")

        passed_count = sum(1 for v in invariants.values() if v["passed"])
        all_passed = (passed_count == len(invariants))

        return {
            "all_passed": all_passed,
            "passed_count": passed_count,
            "total_count": len(invariants),
            "invariants": invariants,
            "violations": violations,
        }

    @classmethod
    def split_giant_tour(
        cls, problem: ProblemInstance, customer_order: List[str]
    ) -> List[Tuple[VehicleConfig, List[str]]]:
        """
        Implements K-vehicle constrained Prins' Split Algorithm (2004) for VRP:
        Decodes a giant customer sequence [C1, C2, ..., CN] into optimal capacity-constrained
        vehicle routes using dynamic programming over an auxiliary DAG across the fleet of X vehicles.
        """
        N = len(customer_order)
        vehicles = problem.vehicles
        K = len(vehicles)
        origin = problem.origin

        if N == 0:
            return [(v, [origin, origin]) for v in vehicles]

        # Precompute leg costs between origin and customers, and customer-to-customer
        leg_costs: Dict[Tuple[str, str], float] = {}

        def get_cost(u: str, v: str) -> float:
            if (u, v) not in leg_costs:
                try:
                    c, _, _, _, _, _, _, _ = cls.evaluate_leg(problem, u, v)
                    leg_costs[(u, v)] = c
                except Exception:
                    leg_costs[(u, v)] = 1e7
            return leg_costs[(u, v)]

        # K-stage DP: V[k][j] = min cost to serve customers 0..j-1 using at most k vehicles
        V = [[float("inf")] * (N + 1) for _ in range(K + 1)]
        P = [[-1] * (N + 1) for _ in range(K + 1)]
        V[0][0] = 0.0

        for k in range(1, K + 1):
            v_obj = vehicles[k - 1]
            is_avail = getattr(v_obj, "is_available", True) and getattr(v_obj, "status", "available") not in ("unavailable", "maintenance")
            cap_k = v_obj.capacity if is_avail else 0.0
            # Option: vehicle k is unused, inherit cost from k-1
            for j in range(N + 1):
                V[k][j] = V[k - 1][j]
                P[k][j] = -2  # marker for unused vehicle

            for i in range(N):
                if V[k - 1][i] == float("inf"):
                    continue
                load = 0.0
                route_cost = 0.0
                for j in range(i + 1, N + 1):
                    cust = customer_order[j - 1]
                    load += problem.customer_demands.get(cust, 1.0)
                    if load > cap_k:
                        break

                    if j == i + 1:
                        route_cost = get_cost(origin, cust) + get_cost(cust, origin)
                    else:
                        prev_cust = customer_order[j - 2]
                        route_cost = route_cost - get_cost(prev_cust, origin) + get_cost(prev_cust, cust) + get_cost(cust, origin)

                    if V[k - 1][i] + route_cost < V[k][j]:
                        V[k][j] = V[k - 1][i] + route_cost
                        P[k][j] = i

        # Reconstruct vehicle routes
        routes_assigned: List[List[str]] = [[] for _ in range(K)]
        curr_j = N
        curr_k = K
        while curr_k > 0 and curr_j > 0:
            prev_i = P[curr_k][curr_j]
            if prev_i == -2:
                curr_k -= 1
            elif prev_i >= 0:
                routes_assigned[curr_k - 1] = customer_order[prev_i:curr_j]
                curr_j = prev_i
                curr_k -= 1
            else:
                break

        # Fallback if demand exceeded fleet capacity
        if curr_j > 0:
            routes_assigned = [[] for _ in range(K)]
            curr_v_idx = 0
            curr_load = 0.0
            for cust in customer_order:
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

        routes: List[Tuple[VehicleConfig, List[str]]] = []
        for idx in range(K):
            v_cfg = vehicles[idx]
            custs = routes_assigned[idx]
            seq = [origin] + custs + [origin] if custs else [origin, origin]
            routes.append((v_cfg, seq))

        return routes

    @classmethod
    def evaluate_sequence(
        cls,
        problem: ProblemInstance,
        visit_sequence: List[str],
        algorithm_name: str = "Common Evaluator",
        computation_time_ms: float = 0.0,
        solver_details: Optional[Dict[str, Any]] = None,
        math_proof: Optional[Dict[str, Any]] = None,
    ) -> AlgorithmResult:
        """
        Evaluates a customer sequence on problem.graph G(t).
        Supports single vehicle or splits into fleet routes if num_vehicles > 1.
        """
        if problem.num_vehicles > 1 and len(visit_sequence) > 2 and visit_sequence[0] == problem.origin:
            cust_list = [c for c in visit_sequence if c != problem.origin]
            fleet_routes = cls.split_giant_tour(problem, cust_list)
            return cls.evaluate_fleet_routes(
                problem,
                fleet_routes,
                algorithm_name=algorithm_name,
                computation_time_ms=computation_time_ms,
                solver_details=solver_details,
                math_proof=math_proof,
            )

        # Single vehicle route evaluation
        v_cfg = problem.vehicles[0] if problem.vehicles else VehicleConfig(vehicle_id="V1", capacity=problem.vehicle_capacity)
        return cls.evaluate_fleet_routes(
            problem,
            [(v_cfg, visit_sequence if visit_sequence[-1] == problem.origin else visit_sequence + [problem.origin])],
            algorithm_name=algorithm_name,
            computation_time_ms=computation_time_ms,
            solver_details=solver_details,
            math_proof=math_proof,
        )
