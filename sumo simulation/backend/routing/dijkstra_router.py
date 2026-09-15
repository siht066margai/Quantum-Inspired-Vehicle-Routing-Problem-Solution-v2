import time
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from backend.graph.dynamic_graph import DynamicTrafficGraph

class DijkstraRouter:
    """
    Computes exact baseline shortest paths on dynamic graph G(t) using Dijkstra's algorithm.
    """

    def __init__(self):
        pass

    def find_shortest_path(
        self,
        dt_graph: DynamicTrafficGraph,
        origin_node: str,
        destination_node: str,
        weight_key: str = 'weight'
    ) -> Dict[str, Any]:
        """
        Computes the shortest path between origin_node and destination_node.
        Returns detailed route results including travel time, distance, bottlenecks, and polylines.
        """
        g = dt_graph.graph

        if origin_node not in g:
            raise ValueError(f"Origin node '{origin_node}' not present in graph.")
        if destination_node not in g:
            raise ValueError(f"Destination node '{destination_node}' not present in graph.")

        start_time = time.perf_counter()

        try:
            node_path = nx.dijkstra_path(g, origin_node, destination_node, weight=weight_key)
            path_cost = nx.dijkstra_path_length(g, origin_node, destination_node, weight=weight_key)
        except nx.NetworkXNoPath:
            return {
                'success': False,
                'error': f"No path found between {origin_node} and {destination_node}",
                'computation_time_ms': round((time.perf_counter() - start_time) * 1000, 3)
            }

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Construct edge path & metrics
        edge_path = []
        geometry = []
        total_travel_time = 0.0
        total_distance = 0.0
        bottlenecks = []

        segment_calculations = []

        # Add initial node coord
        init_pos = dt_graph.node_positions.get(node_path[0])
        if init_pos:
            geometry.append(init_pos)

        for i in range(len(node_path) - 1):
            u = node_path[i]
            v = node_path[i + 1]

            edge_attrs = g[u][v]
            edge_id = edge_attrs.get('key', edge_attrs.get('edge_id', ''))
            edge_path.append(edge_id)

            edge_data = dt_graph.get_edge_data(edge_id) or edge_attrs
            length = float(edge_data.get('length', 0.0))
            tt = float(edge_data.get('travel_time', edge_data.get('free_flow_tt', 0.0)))
            speed = float(edge_data.get('speed_limit', edge_data.get('mean_speed', 13.89)))
            cong = float(edge_data.get('congestion_ratio', 0.0))

            total_distance += length
            total_travel_time += tt

            segment_calculations.append({
                'edge_id': edge_id,
                'from_node': u,
                'to_node': v,
                'length_m': round(length, 2),
                'speed_limit_ms': round(speed, 2),
                'congestion_ratio': round(cong, 3),
                'travel_time_s': round(tt, 2),
                'accumulated_distance_m': round(total_distance, 2),
                'accumulated_time_s': round(total_travel_time, 2),
            })

            if cong > 0.3 or edge_id in dt_graph.incidents:
                bottlenecks.append({
                    'edge_id': edge_id,
                    'congestion_ratio': cong,
                    'is_incident': edge_id in dt_graph.incidents
                })

            shape = edge_data.get('shape', [])
            if shape:
                # Add shape points avoiding duplication
                for p in shape:
                    if not geometry or geometry[-1] != (p[0], p[1]):
                        geometry.append((p[0], p[1]))

        avg_speed = total_distance / max(total_travel_time, 0.1)

        math_proof = {
            'algorithm_name': "Dijkstra's Single-Source Shortest Path Algorithm",
            'graph_weight_function': "w_e(t) = T_e(t) = L_e / max(v_e, 0.1) * (1 + penalty_e)",
            'relaxation_equation': "d[v] = min(d[v], d[u] + w(u, v)) for all v in Adj(u)",
            'total_distance_formula': "Total Distance = sum_{e in Path} L_e",
            'total_time_formula': "Total Travel Time = sum_{e in Path} T_e(t)",
            'average_speed_formula': "Average Speed = (Total Distance) / max(Total Travel Time, 0.1)",
        }

        return {
            'success': True,
            'algorithm': 'Dijkstra',
            'origin_node': origin_node,
            'destination_node': destination_node,
            'node_path': node_path,
            'edge_path': edge_path,
            'total_cost': round(path_cost, 2),
            'total_travel_time': round(total_travel_time, 2),
            'total_distance': round(total_distance, 2),
            'average_speed_ms': round(avg_speed, 2),
            'bottleneck_count': len(bottlenecks),
            'bottlenecks': bottlenecks,
            'segment_calculations': segment_calculations,
            'math_proof': math_proof,
            'geometry': geometry,
            'computation_time_ms': round(elapsed_ms, 3)
        }
