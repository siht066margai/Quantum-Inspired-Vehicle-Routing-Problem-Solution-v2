from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.graph.dynamic_graph import DynamicTrafficGraph


@dataclass
class Order:
    order_id: str
    node_id: str
    x: float = 0.0
    y: float = 0.0
    demand: float = 1.0
    priority: int = 1
    ready_time: float = 0.0
    due_time: float = 86400.0
    service_time: float = 0.0
    status: str = "pending"  # pending, assigned, in_transit, delivered

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "node_id": self.node_id,
            "x": self.x,
            "y": self.y,
            "demand": self.demand,
            "priority": self.priority,
            "ready_time": self.ready_time,
            "due_time": self.due_time,
            "service_time": self.service_time,
            "status": self.status,
        }


@dataclass
class VehicleConfig:
    vehicle_id: str
    capacity: float = 100.0
    start_node: Optional[str] = None
    color: str = "#3b82f6"
    is_available: bool = True
    status: str = "available"  # "available" or "unavailable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "capacity": self.capacity,
            "start_node": self.start_node,
            "color": self.color,
            "is_available": self.is_available,
            "status": self.status,
        }


@dataclass
class FleetVehicleRoute:
    vehicle_id: str
    capacity: float
    assigned_orders: List[str]  # Order IDs or customer node IDs
    visit_sequence: List[str]  # e.g., ['Origin', 'C1', 'C4', 'Origin']
    node_path: List[str]
    edge_path: List[str]
    geometry: List[Tuple[float, float]]
    total_cost: float
    total_travel_time: float
    total_distance: float
    load_used: float
    capacity_utilization_pct: float = 0.0
    initial_load: float = 0.0
    delivered_load: float = 0.0
    remaining_load: float = 0.0
    operational_steps: List[Dict[str, Any]] = field(default_factory=list)
    segment_calculations: List[Dict[str, Any]] = field(default_factory=list)
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    status: str = "active"  # "active" or "standby"
    color: str = "#3b82f6"
    legs: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def assigned_customers(self) -> List[str]:
        return self.assigned_orders

    @assigned_customers.setter
    def assigned_customers(self, val: List[str]):
        self.assigned_orders = val

    @property
    def is_standby(self) -> bool:
        return not self.is_active or self.status == "standby"

    @is_standby.setter
    def is_standby(self, val: bool):
        self.is_active = not val
        self.status = "standby" if val else "active"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "capacity": self.capacity,
            "assigned_orders": self.assigned_orders,
            "assigned_customers": self.assigned_orders,
            "is_standby": self.is_standby,
            "visit_sequence": self.visit_sequence,
            "node_path": self.node_path,
            "edge_path": self.edge_path,
            "geometry": self.geometry,
            "total_cost": round(self.total_cost, 2),
            "total_travel_time": round(self.total_travel_time, 2),
            "total_distance": round(self.total_distance, 2),
            "load_used": round(self.load_used, 2),
            "capacity_utilization_pct": round(self.capacity_utilization_pct, 1),
            "initial_load": round(self.initial_load, 2),
            "delivered_load": round(self.delivered_load, 2),
            "remaining_load": round(self.remaining_load, 2),
            "operational_steps": self.operational_steps,
            "segment_calculations": self.segment_calculations,
            "bottlenecks": self.bottlenecks,
            "is_active": self.is_active,
            "status": self.status,
            "color": self.color,
            "legs": self.legs,
        }


@dataclass
class ProblemInstance:
    """
    Immutable representation of a dynamic VRP problem instance captured at simulation time t.
    Supports single-vehicle and multi-vehicle dynamic fleet routing.
    """
    origin: str
    destinations: List[str]
    graph: DynamicTrafficGraph
    vehicle_capacity: float = 100.0
    num_vehicles: int = 1
    vehicles: List[VehicleConfig] = field(default_factory=list)
    orders: List[Order] = field(default_factory=list)
    customer_demands: Dict[str, float] = field(default_factory=dict)
    objective_weights: Dict[str, float] = field(
        default_factory=lambda: {"alpha": 1.0, "beta": 0.0, "gamma": 0.0}
    )
    time_windows: Optional[Dict[str, Tuple[float, float]]] = None
    incidents: Dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0

    def __post_init__(self):
        if self.orders and not self.customer_demands:
            self.customer_demands = {o.node_id: o.demand for o in self.orders}
            if not self.destinations:
                self.destinations = [o.node_id for o in self.orders]
        elif not self.customer_demands and self.destinations:
            # Default demand of 1.0 package unit per customer if unspecified
            self.customer_demands = {dest: 1.0 for dest in self.destinations}
        elif self.customer_demands and not self.destinations:
            self.destinations = list(self.customer_demands.keys())

        if not self.vehicles and self.num_vehicles > 0:
            # Generate default vehicles based on num_vehicles and vehicle_capacity
            colors = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4", "#ef4444"]
            self.vehicles = [
                VehicleConfig(
                    vehicle_id=f"V{i+1}",
                    capacity=self.vehicle_capacity,
                    start_node=self.origin,
                    color=colors[i % len(colors)]
                )
                for i in range(self.num_vehicles)
            ]
        else:
            self.num_vehicles = len(self.vehicles)

        if not self.incidents and self.graph:
            self.incidents = dict(self.graph.incidents)

    def is_valid(self) -> Tuple[bool, str]:
        """
        Validates graph connectivity and mathematical feasibility of the VRP instance.
        Enforces Edge Cases 1, 2, 8, 9, 11, 12, 14, 24, 25, 26, 27, 30.
        """
        g = self.graph.graph
        if self.origin not in g:
            return False, f"Origin node '{self.origin}' does not exist / not present in SUMO network graph."
        if g.degree(self.origin) == 0:
            return False, f"Origin node '{self.origin}' is isolated with no road connections in SUMO network graph."

        for idx, dest in enumerate(self.destinations):
            if dest not in g:
                return False, f"Destination {idx + 1} node '{dest}' not present in SUMO network graph."
            if g.degree(dest) == 0:
                return False, f"Destination {idx + 1} node '{dest}' is isolated with no road connections in SUMO network graph."

        # Edge Case 14: Duplicate customer nodes in destination list
        if len(self.destinations) != len(set(self.destinations)):
            dups = [d for d in self.destinations if self.destinations.count(d) > 1]
            return False, f"Duplicate customer destination detected: {set(dups)}."

        # Edge Case 14: Duplicate customer order ID check
        if self.orders:
            order_ids = [o.order_id for o in self.orders]
            if len(order_ids) != len(set(order_ids)):
                dups = [oid for oid in order_ids if order_ids.count(oid) > 1]
                return False, f"Duplicate customer order identifier detected: {set(dups)}."

        # Edge Case 26 & 24: Non-positive or negative vehicle capacities
        for v in self.vehicles:
            if v.capacity < 0:
                return False, f"Vehicle {v.vehicle_id} has invalid negative capacity ({v.capacity})."
            if v.capacity == 0 and len(self.destinations) > 0:
                return False, f"Vehicle {v.vehicle_id} has zero capacity with positive customer demand. CVRP is mathematically infeasible."

        # Edge Case 1: Zero vehicles with positive destinations
        if (self.num_vehicles <= 0 or len(self.vehicles) == 0) and len(self.destinations) > 0:
            return False, "Number of vehicles cannot be zero for non-empty customer destinations. CVRP is mathematically infeasible."

        # Edge Case 1 & 21: Zero customers / Depot only
        if len(self.destinations) == 0:
            return False, "Customer count is zero. CVRP requires at least one customer destination."

        # Edge Case 14 & 15: Customer package demand check (reject negative demand, zero demand allowed)
        for dest in self.destinations:
            d_i = self.customer_demands.get(dest, 1.0)
            if d_i < 0:
                return False, f"Customer '{dest}' has invalid negative package demand ({d_i}). Demand must be non-negative."

        # Edge Case 30: Account only for available vehicles
        available_vehicles = [
            v for v in self.vehicles
            if getattr(v, "is_available", True) and getattr(v, "status", "available") != "unavailable"
        ]
        if not available_vehicles:
            return False, "All fleet vehicles are marked unavailable. CVRP is mathematically infeasible."

        # Strict Fleet Capacity Feasibility Checks (Invariants I04 and I19 / Edge Cases 8, 9, 11, 12)
        total_demand = sum(self.customer_demands.get(dest, 1.0) for dest in self.destinations)
        total_capacity = sum(v.capacity for v in available_vehicles)
        max_capacity = max((v.capacity for v in available_vehicles), default=0.0)

        if total_demand > total_capacity:
            return (
                False,
                f"Total Demand ({total_demand:.1f} packages) > Total Fleet Capacity ({total_capacity:.1f} packages across {len(available_vehicles)} available vehicles). CVRP is mathematically infeasible."
            )

        for dest in self.destinations:
            d_i = self.customer_demands.get(dest, 1.0)
            if d_i > max_capacity:
                return (
                    False,
                    f"Customer '{dest}' demand ({d_i:.1f} packages) exceeds maximum vehicle capacity ({max_capacity:.1f} packages). CVRP is mathematically infeasible."
                )

        return True, "Valid ProblemInstance"

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable problem instance metadata."""
        return {
            "origin": self.origin,
            "destinations": self.destinations,
            "num_vehicles": self.num_vehicles,
            "vehicles": [v.to_dict() for v in self.vehicles],
            "orders": [o.to_dict() for o in self.orders],
            "vehicle_capacity": self.vehicle_capacity,
            "customer_demands": self.customer_demands,
            "objective_weights": self.objective_weights,
            "incident_count": len(self.incidents),
            "timestamp": self.timestamp,
        }


@dataclass
class AlgorithmResult:
    """
    Standardized algorithm result schema returned by every VRP solver (Dijkstra, QPSO, QAOA, ALNS, HGS).
    Supports single-vehicle sequence as well as multi-vehicle fleet solutions.
    """
    algorithm: str
    status: str
    success: bool
    visit_sequence: List[str]  # Primary/combined sequence, e.g. ['O', 'C1', 'C2']
    node_path: List[str]
    edge_path: List[str]
    total_cost: float
    total_travel_time: float
    total_distance: float
    average_speed_ms: float
    bottleneck_count: int
    bottlenecks: List[Dict[str, Any]]
    segment_calculations: List[Dict[str, Any]]
    geometry: List[Tuple[float, float]]
    feasible: bool
    constraint_violations: List[str]
    computation_time_ms: float
    math_proof: Dict[str, Any]
    solver_details: Dict[str, Any] = field(default_factory=dict)
    fleet_routes: List[FleetVehicleRoute] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def vehicle_routes(self) -> List[FleetVehicleRoute]:
        return self.fleet_routes

    @vehicle_routes.setter
    def vehicle_routes(self, val: List[FleetVehicleRoute]):
        self.fleet_routes = val

    def to_dict(self) -> Dict[str, Any]:
        fr_dicts = [fr.to_dict() for fr in self.fleet_routes]
        return {
            "algorithm": self.algorithm,
            "status": self.status,
            "success": self.success,
            "visit_sequence": self.visit_sequence,
            "node_path": self.node_path,
            "edge_path": self.edge_path,
            "total_cost": round(self.total_cost, 2),
            "total_travel_time": round(self.total_travel_time, 2),
            "total_distance": round(self.total_distance, 2),
            "average_speed_ms": round(self.average_speed_ms, 2),
            "bottleneck_count": self.bottleneck_count,
            "bottlenecks": self.bottlenecks,
            "segment_calculations": self.segment_calculations,
            "geometry": self.geometry,
            "feasible": self.feasible,
            "constraint_violations": self.constraint_violations,
            "computation_time_ms": round(self.computation_time_ms, 3),
            "math_proof": self.math_proof,
            "solver_details": self.solver_details,
            "fleet_routes": fr_dicts,
            "vehicle_routes": fr_dicts,
            "error": self.error,
        }
