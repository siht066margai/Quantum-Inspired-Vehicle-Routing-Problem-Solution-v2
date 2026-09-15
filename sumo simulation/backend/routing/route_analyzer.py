"""
Route Comparison & Analysis Engine for 3 Routing Algorithms:
1. Dijkstra (Exact Shortest Path Baseline)
2. QAOA (Quantum Approximate Optimization Algorithm — Azfar et al., 2025)
3. QPSO (Quantum-Behaved Particle Swarm Optimization — Herrera et al., 2015)

Evaluates and compares all three algorithms side-by-side using raw output metrics
derived directly from live executions on SUMO traffic state G(t).
"""

from __future__ import annotations
from typing import Any, Dict
from backend.graph.dynamic_graph import DynamicTrafficGraph


def summarize_route(dt_graph: DynamicTrafficGraph, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts dynamic metrics from a Dijkstra, QAOA, or QPSO algorithm execution result.
    """
    if not result.get("success"):
        return {
            "available": False,
            "algorithm": result.get("algorithm", "Unknown"),
            "error": result.get("error", "No route result is available."),
        }

    edge_path = result.get("edge_path", [])
    congestion_values = []
    incident_edges = []

    for edge_id in edge_path:
        edge_data = dt_graph.get_edge_data(edge_id) or {}
        congestion_values.append(float(edge_data.get("congestion_ratio", 0.0)))
        if edge_id in dt_graph.incidents:
            incident_edges.append(edge_id)

    average_congestion = sum(congestion_values) / len(congestion_values) if congestion_values else 0.0

    summary = {
        "available": True,
        "algorithm": result.get("algorithm", "Unknown"),
        "dynamic_cost": round(float(result.get("total_cost", 0.0)), 3),
        "travel_time_seconds": round(float(result.get("total_travel_time", 0.0)), 3),
        "distance_metres": round(float(result.get("total_distance", 0.0)), 3),
        "average_speed_ms": round(float(result.get("average_speed_ms", 0.0)), 2),
        "average_congestion_percent": round(average_congestion * 100.0, 2),
        "bottleneck_count": int(result.get("bottleneck_count", 0)),
        "incident_edge_count": len(incident_edges),
        "computation_time_ms": round(float(result.get("computation_time_ms", 0.0)), 3),
        "segment_calculations": result.get("segment_calculations", []),
        "math_proof": result.get("math_proof", {}),
    }

    # Add algorithm-specific technical parameters
    algo_name = result.get("algorithm", "")
    if "QAOA" in algo_name:
        summary.update({
            "qaoa_qubits": result.get("qaoa_qubits"),
            "qaoa_depth": result.get("qaoa_depth"),
            "penalty_multiplier_P": result.get("penalty_multiplier_P"),
            "normalization_factor": result.get("normalization_factor"),
            "selected_probability_percent": round(float(result.get("selected_probability", 0.0)) * 100.0, 2),
            "optimizer": result.get("optimizer", "COBYLA"),
            "qubo_details": result.get("qubo_details", {}),
        })
    elif "QPSO" in algo_name:
        summary.update({
            "qpso_particles": result.get("qpso_particles"),
            "qpso_iterations": result.get("qpso_iterations"),
            "final_alpha": result.get("final_alpha"),
            "candidate_subgraph_nodes": result.get("candidate_subgraph_nodes"),
            "swarm_details": result.get("swarm_details", {}),
        })

    return summary


def analyze_routes(
    dt_graph: DynamicTrafficGraph,
    dijkstra_result: Dict[str, Any],
    qaoa_result: Dict[str, Any],
    qpso_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compares Dijkstra, QAOA, and QPSO algorithm executions side-by-side using live output metrics.
    """
    dijkstra = summarize_route(dt_graph, dijkstra_result)
    qaoa = summarize_route(dt_graph, qaoa_result)
    qpso = summarize_route(dt_graph, qpso_result) if qpso_result else {"available": False}

    response: Dict[str, Any] = {
        "success": dijkstra["available"] or qaoa["available"] or qpso["available"],
        "comparison_basis": (
            "Dynamic route cost, travel time, distance, speed, and computation latency "
            "derived strictly from live algorithm executions on SUMO traffic state G(t)."
        ),
        "algorithms": {
            "dijkstra": dijkstra,
            "qaoa": qaoa,
            "qpso": qpso
        }
    }

    # Raw metrics dictionary for 3-way side-by-side numerical inspection
    available_algos = {
        "Dijkstra": dijkstra,
        "QAOA": qaoa,
        "QPSO": qpso
    }

    valid_algos = {k: v for k, v in available_algos.items() if v.get("available")}

    if valid_algos:
        lowest_cost_algo = min(valid_algos.items(), key=lambda x: x[1]["dynamic_cost"])[0]
        lowest_time_algo = min(valid_algos.items(), key=lambda x: x[1]["travel_time_seconds"])[0]
        fastest_compute_algo = min(valid_algos.items(), key=lambda x: x[1]["computation_time_ms"])[0]

        response["outcome"] = {
            "best_algorithm": lowest_cost_algo,
            "lowest_cost_algorithm": lowest_cost_algo,
            "lowest_traffic_risk_algorithm": lowest_cost_algo,
            "lowest_travel_time_algorithm": lowest_time_algo,
            "fastest_computation_algorithm": fastest_compute_algo,
        }

    return response
