import os
import sys
import time
import traci
from typing import Dict, List, Tuple, Any, Optional

from backend.traffic.traffic_extractor import TrafficStateExtractor
from backend.graph.dynamic_graph import DynamicTrafficGraph

class SumoManager:
    """
    Manages SUMO simulation lifecycle via TraCI, vehicle observation,
    incident injection, and real-time rerouting.
    """

    def __init__(self, sumocfg_path: str, dt_graph: DynamicTrafficGraph):
        self.sumocfg_path = os.path.abspath(sumocfg_path)
        self.dt_graph = dt_graph
        self.extractor = TrafficStateExtractor()
        self.is_running = False
        self.is_paused = False
        self.step_delay_sec = 0.05
        self.active_incidents: Dict[str, float] = {}  # edge_id -> original speed limit

    def start(self, use_gui: bool = True) -> bool:
        """
        Launches SUMO and initializes TraCI connection.
        """
        if traci.isLoaded():
            traci.close()

        sumo_binary = "sumo-gui" if use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.sumocfg_path,
            "--start",
            "--quit-on-end", "false"
        ]

        try:
            traci.start(sumo_cmd)
            self.is_running = True
            self.is_paused = False

            # Update extractor static edge info from graph metadata
            self.extractor.set_net_edges_info(self.dt_graph.edge_metadata)
            return True
        except Exception as e:
            print(f"Error starting TraCI with command {sumo_cmd}: {e}")
            self.is_running = False
            return False

    def step(self) -> Dict[str, Any]:
        """
        Advances simulation by 1 step, updates dynamic graph weights, and extracts state.
        """
        if not self.is_running or not traci.isLoaded():
            return {'running': False}

        try:
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            active_ids = traci.vehicle.getIDList()
            loaded = traci.simulation.getLoadedNumber()
            arrived = traci.simulation.getArrivedNumber()
            expected = traci.simulation.getMinExpectedNumber()

            # Extract edge state and update dynamic graph weights
            traffic_state = self.extractor.extract_state()
            self.dt_graph.update_weights(traffic_state)

            return {
                'running': True,
                'sim_time': current_time,
                'active_count': len(active_ids),
                'loaded_count': loaded,
                'arrived_count': arrived,
                'expected_count': expected,
                'traffic_state': traffic_state
            }
        except Exception as e:
            print(f"SUMO simulation step ended or encountered error: {e}")
            self.is_running = False
            return {'running': False, 'error': str(e)}

    def get_vehicle_states(self, max_vehicles: int = 200) -> List[Dict[str, Any]]:
        """
        Retrieves real-time position, speed, and road for active vehicles.
        """
        if not self.is_running or not traci.isLoaded():
            return []

        active_ids = traci.vehicle.getIDList()
        vehicles = []

        for veh_id in active_ids[:max_vehicles]:
            try:
                x, y = traci.vehicle.getPosition(veh_id)
                speed = traci.vehicle.getSpeed(veh_id)
                road_id = traci.vehicle.getRoadID(veh_id)
                veh_type = traci.vehicle.getTypeID(veh_id)

                vehicles.append({
                    'id': veh_id,
                    'type': veh_type,
                    'x': round(x, 2),
                    'y': round(y, 2),
                    'speed': round(speed, 2),
                    'road_id': road_id
                })
            except traci.TraCIException:
                continue

        return vehicles

    def get_snapshot(self) -> Dict[str, Any]:
        """Return a read-only snapshot for dashboards when WebSocket delivery is unavailable."""
        snapshot = {
            'running': self.is_running,
            'paused': self.is_paused,
            'sim_time': 0.0,
            'active_count': 0,
            'loaded_count': 0,
            'arrived_count': 0,
            'expected_count': 0,
        }
        if not self.is_running or not traci.isLoaded():
            return snapshot

        try:
            snapshot.update({
                'sim_time': traci.simulation.getTime(),
                'active_count': len(traci.vehicle.getIDList()),
                'loaded_count': traci.simulation.getLoadedNumber(),
                'arrived_count': traci.simulation.getArrivedNumber(),
                'expected_count': traci.simulation.getMinExpectedNumber(),
            })
        except traci.TraCIException:
            snapshot['running'] = False
        return snapshot

    def inject_incident(self, edge_id: str, speed_factor: float = 0.05) -> bool:
        """
        Simulates an incident on edge_id by reducing max speed in SUMO and increasing graph weight.
        """
        if not self.is_running or not traci.isLoaded():
            return False

        if edge_id in self.dt_graph.edge_metadata:
            orig_speed = self.dt_graph.edge_metadata[edge_id]['speed_limit']
            self.active_incidents[edge_id] = orig_speed

            reduced_speed = max(0.1, orig_speed * speed_factor)

            try:
                # Reduce lane max speed in SUMO
                lane_count = self.dt_graph.edge_metadata[edge_id]['lane_count']
                for index in range(lane_count):
                    traci.lane.setMaxSpeed(f"{edge_id}_{index}", reduced_speed)
            except Exception as e:
                print(f"Warning setting lane max speed for {edge_id}: {e}")

            # Mark incident on graph
            self.dt_graph.set_incident(edge_id, penalty_multiplier=50.0)
            return True

        return False

    def clear_incident(self, edge_id: str) -> bool:
        """
        Clears an incident on edge_id and restores normal speed.
        """
        if not self.is_running or not traci.isLoaded():
            return False

        if edge_id in self.active_incidents:
            orig_speed = self.active_incidents[edge_id]
            del self.active_incidents[edge_id]

            try:
                lane_count = self.dt_graph.edge_metadata[edge_id]['lane_count']
                for index in range(lane_count):
                    traci.lane.setMaxSpeed(f"{edge_id}_{index}", orig_speed)
            except Exception as e:
                print(f"Warning restoring lane max speed for {edge_id}: {e}")

            self.dt_graph.clear_incident(edge_id)
            return True

        return False

    def reroute_vehicle(self, veh_id: str, edge_path: List[str]) -> bool:
        """
        Applies a new route edge sequence to a specific vehicle in SUMO.
        """
        if not self.is_running or not traci.isLoaded():
            return False

        try:
            current_road = traci.vehicle.getRoadID(veh_id)
            if current_road in edge_path:
                idx = edge_path.index(current_road)
                valid_path = edge_path[idx:]
            else:
                valid_path = edge_path

            if len(valid_path) > 1:
                traci.vehicle.setRoute(veh_id, valid_path)
                return True
        except Exception as e:
            print(f"Error rerouting vehicle {veh_id}: {e}")

        return False

    def stop(self) -> None:
        """
        Closes TraCI session gracefully.
        """
        if traci.isLoaded():
            try:
                traci.close()
            except Exception:
                pass
        self.is_running = False
        self.is_paused = False
