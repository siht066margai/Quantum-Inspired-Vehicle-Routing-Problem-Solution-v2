import os
import networkx as nx
import sumolib
from typing import Dict, List, Tuple, Any, Optional

class DynamicTrafficGraph:
    """
    Graph Model G(t) = (V, E, W(t)) representing transportation network topology
    and dynamic edge weights updated in real time from SUMO traffic state.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._undirected_graph: Optional[nx.Graph] = None
        self.node_positions: Dict[str, Tuple[float, float]] = {}
        self.edge_metadata: Dict[str, Dict[str, Any]] = {}
        self.net_file_path: Optional[str] = None
        self.incidents: Dict[str, float] = {}  # edge_id -> multiplier or extra weight penalty

    def get_undirected_graph(self) -> nx.Graph:
        if self._undirected_graph is None or self._undirected_graph.number_of_nodes() != self.graph.number_of_nodes():
            self._undirected_graph = self.graph.to_undirected()
        return self._undirected_graph

    def load_net_file(self, net_file_path: str) -> None:
        """
        Parses SUMO network file (.net.xml or .net.xml.gz) and builds the directed graph topology.
        """
        if not os.path.exists(net_file_path):
            raise FileNotFoundError(f"SUMO network file not found: {net_file_path}")

        self.net_file_path = net_file_path
        net = sumolib.net.readNet(net_file_path)

        self.graph.clear()
        self._undirected_graph = None
        self.node_positions.clear()
        self.edge_metadata.clear()

        # Build nodes
        for node in net.getNodes():
            node_id = node.getID()
            x, y = node.getCoord()
            self.node_positions[node_id] = (x, y)
            self.graph.add_node(node_id, x=x, y=y)

        # Build edges (excluding internal connector edges)
        for edge in net.getEdges():
            edge_id = edge.getID()
            if edge_id.startswith(':'):
                continue

            from_node = edge.getFromNode().getID()
            to_node = edge.getToNode().getID()
            length = float(edge.getLength())
            speed = float(edge.getSpeed())
            lane_count = len(edge.getLanes())
            shape = [(float(p[0]), float(p[1])) for p in edge.getShape()]
            free_flow_tt = length / max(speed, 0.1)

            edge_data = {
                'edge_id': edge_id,
                'from_node': from_node,
                'to_node': to_node,
                'length': length,
                'speed_limit': speed,
                'lane_count': lane_count,
                'shape': shape,
                'free_flow_tt': free_flow_tt,
                'vehicle_count': 0,
                'mean_speed': speed,
                'halting_count': 0,
                'travel_time': free_flow_tt,
                'congestion_ratio': 0.0,
                'weight': free_flow_tt
            }

            self.edge_metadata[edge_id] = edge_data
            self.graph.add_edge(from_node, to_node, key=edge_id, **edge_data)

    def set_incident(self, edge_id: str, penalty_multiplier: float = 100.0) -> bool:
        """
        Injects or updates an incident on a target edge.
        penalty_multiplier > 1.0 increases the edge weight dramatically.
        """
        if edge_id in self.edge_metadata:
            self.incidents[edge_id] = penalty_multiplier
            metadata = self.edge_metadata[edge_id]
            from_node = metadata['from_node']
            to_node = metadata['to_node']
            new_weight = metadata['travel_time'] * penalty_multiplier
            metadata['weight'] = new_weight
            if self.graph.has_edge(from_node, to_node):
                self.graph[from_node][to_node]['weight'] = new_weight
            return True
        return False

    def clear_incident(self, edge_id: str) -> bool:
        """
        Removes an incident from a target edge.
        """
        if edge_id in self.incidents:
            del self.incidents[edge_id]
            if edge_id in self.edge_metadata:
                metadata = self.edge_metadata[edge_id]
                from_node = metadata['from_node']
                to_node = metadata['to_node']
                new_weight = metadata['travel_time']
                metadata['weight'] = new_weight
                if self.graph.has_edge(from_node, to_node):
                    self.graph[from_node][to_node]['weight'] = new_weight
            return True
        return False

    def update_weights(
        self,
        traffic_state_dict: Dict[str, Dict[str, Any]],
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.0
    ) -> None:
        """
        Updates dynamic weights W(t) for all edges in G(t).
        w_e(t) = alpha * T_e(t) + beta * D_e + gamma * C_e(t) (+ incident penalties)
        """
        for edge_id, info in traffic_state_dict.items():
            if edge_id in self.edge_metadata:
                metadata = self.edge_metadata[edge_id]
                metadata['vehicle_count'] = info.get('vehicle_count', 0)
                metadata['mean_speed'] = info.get('mean_speed', metadata['speed_limit'])
                metadata['halting_count'] = info.get('halting_count', 0)
                metadata['travel_time'] = info.get('travel_time', metadata['free_flow_tt'])
                metadata['congestion_ratio'] = info.get('congestion_ratio', 0.0)

                tt = metadata['travel_time']
                dist = metadata['length']
                cong = metadata['congestion_ratio']

                # Dynamic cost equation
                weight = alpha * tt + beta * dist + gamma * (cong * 100.0)

                # Apply incident multiplier if active
                if edge_id in self.incidents:
                    multiplier = self.incidents[edge_id]
                    weight *= multiplier

                metadata['weight'] = weight

                from_node = metadata['from_node']
                to_node = metadata['to_node']
                if self.graph.has_edge(from_node, to_node):
                    self.graph[from_node][to_node]['weight'] = weight
                    self.graph[from_node][to_node]['travel_time'] = tt
                    self.graph[from_node][to_node]['congestion_ratio'] = cong
                    self.graph[from_node][to_node]['vehicle_count'] = metadata['vehicle_count']

    def get_edge_data(self, edge_id: str) -> Optional[Dict[str, Any]]:
        return self.edge_metadata.get(edge_id)

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Returns top-level summary stats of graph state.
        """
        congested_edges = [
            e for e in self.edge_metadata.values()
            if e['congestion_ratio'] > 0.3
        ]
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'congested_edge_count': len(congested_edges),
            'incident_count': len(self.incidents)
        }
