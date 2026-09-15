import time
from typing import Dict, List, Any, Optional
import traci

class TrafficStateExtractor:
    """
    Extracts edge-centric traffic state data from SUMO via TraCI.
    Filters out internal junction connector edges (starting with ':').
    """

    def __init__(self, net_edges_info: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        :param net_edges_info: Pre-parsed dictionary of static edge properties (length, speed, lanes, nodes).
        """
        self.net_edges_info = net_edges_info or {}

    def set_net_edges_info(self, net_edges_info: Dict[str, Dict[str, Any]]) -> None:
        self.net_edges_info = net_edges_info

    def extract_state(self) -> Dict[str, Any]:
        """
        Extracts current state for all normal edges from TraCI.
        Returns a dictionary mapping edge_id -> edge_traffic_state dict.
        """
        if not traci.isLoaded():
            return {}

        sim_time = traci.simulation.getTime()
        all_edges = traci.edge.getIDList()
        normal_edges = [e for e in all_edges if not e.startswith(':')]

        traffic_state = {}

        for edge_id in normal_edges:
            static_info = self.net_edges_info.get(edge_id, {})
            length = static_info.get('length', 10.0)
            free_flow_speed = static_info.get('speed', 13.89)  # default ~50 km/h
            lane_count = static_info.get('lane_count', 1)
            from_node = static_info.get('from_node', '')
            to_node = static_info.get('to_node', '')

            free_flow_tt = length / max(free_flow_speed, 0.1)

            # Live TraCI measurements
            vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
            mean_speed = traci.edge.getLastStepMeanSpeed(edge_id)
            halting_count = traci.edge.getLastStepHaltingNumber(edge_id)
            traci_travel_time = traci.edge.getTraveltime(edge_id)

            # Defensive speed evaluation
            if vehicle_count > 0:
                effective_speed = max(mean_speed, 0.1)
                calculated_tt = length / effective_speed
                current_travel_time = max(traci_travel_time, calculated_tt, free_flow_tt)
                congestion_ratio = max(0.0, min(1.0, 1.0 - (mean_speed / max(free_flow_speed, 0.1))))
            else:
                current_travel_time = free_flow_tt
                congestion_ratio = 0.0

            # Additional penalty if vehicles are halting/queued
            if halting_count > 0:
                congestion_ratio = max(congestion_ratio, min(1.0, halting_count * 0.25))

            traffic_state[edge_id] = {
                'edge_id': edge_id,
                'from_node': from_node,
                'to_node': to_node,
                'length': length,
                'lane_count': lane_count,
                'free_flow_speed': free_flow_speed,
                'free_flow_tt': free_flow_tt,
                'vehicle_count': vehicle_count,
                'mean_speed': mean_speed,
                'halting_count': halting_count,
                'travel_time': current_travel_time,
                'congestion_ratio': round(congestion_ratio, 4),
                'timestamp': sim_time
            }

        return traffic_state
