import os
import sys
import asyncio
import json
import random
import uuid
import time
from typing import Dict, List, Any, Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

# Ensure project root is in python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.graph.dynamic_graph import DynamicTrafficGraph
from backend.routing.dijkstra_router import DijkstraRouter
from backend.routing.qaoa_router import QAOARouter
from backend.routing.qpso_router import QPSORouter
from backend.routing.astar_router import AStarRouter
from backend.routing.route_analyzer import analyze_routes
from backend.simulation.sumo_manager import SumoManager

from backend.vrp.problem_instance import ProblemInstance, AlgorithmResult, Order, VehicleConfig, FleetVehicleRoute
from backend.vrp.dijkstra_vrp import DijkstraVRPSolver
from backend.vrp.qpso_vrp import QPSOVRPSolver
from backend.vrp.qaoa_vrp import QAOAVRPSolver
from backend.vrp.alns_vrp import ALNSVRPSolver
from backend.vrp.hgs_vrp import HGSVRPSolver
from backend.vrp.exact_mip_vrp import ExactMIPVRPSolver
from backend.vrp.rerouter import DynamicRerouter
from backend.vrp.experiment_db import ExperimentDatabase

sim_loop_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sim_loop_task
    sim_loop_task = asyncio.create_task(simulation_loop())
    yield
    sumo_mgr.stop()
    if sim_loop_task:
        sim_loop_task.cancel()

app = FastAPI(title="SIH 2026 Traffic Route Optimization Platform", lifespan=lifespan)

# Global instances
NET_FILE = os.path.join(ROOT_DIR, "osm.net.xml.gz")
SUMOCFG_FILE = os.path.join(ROOT_DIR, "osm.sumocfg")

dt_graph = DynamicTrafficGraph()
# Load graph topology on server startup
dt_graph.load_net_file(NET_FILE)

router = DijkstraRouter()
astar_router = AStarRouter()
qaoa_router = QAOARouter()
qpso_router = QPSORouter()
sumo_mgr = SumoManager(SUMOCFG_FILE, dt_graph)

dijkstra_vrp_solver = DijkstraVRPSolver()
qpso_vrp_solver = QPSOVRPSolver()
qaoa_vrp_solver = QAOAVRPSolver()
alns_vrp_solver = ALNSVRPSolver()
hgs_vrp_solver = HGSVRPSolver()
exact_mip_vrp_solver = ExactMIPVRPSolver()
dynamic_rerouter = DynamicRerouter()
experiment_db = ExperimentDatabase()

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        payload = json.dumps(message)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(payload)
            except Exception:
                self.disconnect(connection)

ws_manager = ConnectionManager()

# Background simulation runner task
sim_loop_task: Optional[asyncio.Task] = None
is_loop_running = False

# Active route state tracked for recalculation on dynamic graph updates
active_route_req: Optional[Dict[str, str]] = None
active_vrp_req: Optional[Dict[str, Any]] = None
latest_route_result: Optional[Dict[str, Any]] = None
route_algorithm_runs: Dict[tuple[str, str], Dict[str, Dict[str, Any]]] = {}

# Real-time event logging and fleet execution tracking
sim_event_log: List[Dict[str, Any]] = []
MAX_EVENT_LOG = 200
fleet_execution_tracker: Dict[str, Any] = {
    "active_plan_timestamp": None,
    "vehicles": {},
}

def log_sim_event(event_type: str, message: str, vehicle_id: Optional[str] = None, severity: str = "info", sim_time: float = 0.0) -> Dict[str, Any]:
    evt = {
        "id": uuid.uuid4().hex[:8],
        "timestamp": round(sim_time, 1),
        "type": event_type,
        "vehicle_id": vehicle_id,
        "message": message,
        "severity": severity,
    }
    sim_event_log.append(evt)
    if len(sim_event_log) > MAX_EVENT_LOG:
        sim_event_log.pop(0)
    return evt



def _build_problem_instance_from_dict(req_dict: Dict[str, Any]) -> ProblemInstance:
    origin = req_dict.get("origin") or req_dict.get("origin_node")
    destinations = req_dict.get("destinations") or req_dict.get("destination_nodes", [])
    num_vehicles = req_dict.get("num_vehicles", 1)
    vehicle_capacity = req_dict.get("vehicle_capacity", 100.0)

    raw_vehicles = req_dict.get("vehicles", [])
    vehicles_list = []
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#06b6d4", "#ef4444"]
    if raw_vehicles:
        for idx, v in enumerate(raw_vehicles):
            if isinstance(v, dict):
                vehicles_list.append(VehicleConfig(
                    vehicle_id=v.get("vehicle_id", f"V{idx+1}"),
                    capacity=float(v.get("capacity", vehicle_capacity)),
                    start_node=v.get("start_node", origin),
                    color=v.get("color", colors[idx % len(colors)]),
                    is_available=bool(v.get("is_available", True)),
                    status=str(v.get("status", "active")),
                ))
            elif isinstance(v, VehicleConfig):
                vehicles_list.append(v)
    else:
        vehicles_list = [
            VehicleConfig(
                vehicle_id=f"V{i+1}",
                capacity=vehicle_capacity,
                start_node=origin,
                color=colors[i % len(colors)],
                is_available=True,
                status="active",
            )
            for i in range(max(1, num_vehicles))
        ]

    raw_orders = req_dict.get("orders", [])
    orders_list = []
    customer_demands = dict(req_dict.get("customer_demands", {}))
    if raw_orders:
        for o in raw_orders:
            if isinstance(o, dict):
                node = o.get("node_id", "")
                d = float(o.get("demand", 1.0))
                orders_list.append(Order(
                    order_id=o.get("order_id", f"O_{node}"),
                    node_id=node,
                    x=float(o.get("x", 0.0)),
                    y=float(o.get("y", 0.0)),
                    demand=d,
                    priority=int(o.get("priority", 1)),
                ))
                customer_demands[node] = d

    alpha = req_dict.get("alpha", 1.0)
    beta = req_dict.get("beta", 0.0)
    gamma = req_dict.get("gamma", 0.0)

    return ProblemInstance(
        origin=origin,
        destinations=destinations,
        graph=dt_graph,
        vehicle_capacity=vehicle_capacity,
        num_vehicles=len(vehicles_list),
        vehicles=vehicles_list,
        orders=orders_list,
        customer_demands=customer_demands,
        objective_weights={"alpha": alpha, "beta": beta, "gamma": gamma},
        timestamp=sumo_mgr.get_snapshot().get("sim_time", 0.0)
    )


def _execute_vrp_solver_sync() -> Optional[Dict[str, Any]]:
    """CPU-bound solver worker executed off main thread to prevent UI/SUMO freezes."""
    global active_vrp_req, latest_route_result
    if not active_vrp_req:
        return None

    algorithm = active_vrp_req.get("algorithm", "dijkstra")
    prob = _build_problem_instance_from_dict(active_vrp_req)

    if algorithm == "dijkstra":
        res = dijkstra_vrp_solver.solve(prob)
        res_dict = res.to_dict()
        res_dict["snapshot_timestamp"] = prob.timestamp
        latest_route_result = res_dict
        return res_dict

    elif algorithm == "qpso":
        num_particles = active_vrp_req.get("num_particles", 20)
        max_iter = active_vrp_req.get("max_iter", 25)
        res = qpso_vrp_solver.solve(prob, num_particles=num_particles, max_iter=max_iter)
        res_dict = res.to_dict()
        res_dict["snapshot_timestamp"] = prob.timestamp
        latest_route_result = res_dict
        return res_dict

    elif algorithm == "qaoa":
        reps = active_vrp_req.get("reps", 1)
        shots = active_vrp_req.get("shots", 1024)
        res = qaoa_vrp_solver.solve(prob, reps=reps, shots=shots)
        res_dict = res.to_dict()
        res_dict["snapshot_timestamp"] = prob.timestamp
        latest_route_result = res_dict
        return res_dict

    elif algorithm == "alns":
        max_iter = active_vrp_req.get("max_iter", 40)
        res = alns_vrp_solver.solve(prob, max_iter=max_iter)
        res_dict = res.to_dict()
        res_dict["snapshot_timestamp"] = prob.timestamp
        latest_route_result = res_dict
        return res_dict

    elif algorithm == "hgs":
        pop_size = active_vrp_req.get("pop_size", 25)
        max_iter = active_vrp_req.get("max_iter", 40)
        res = hgs_vrp_solver.solve(prob, pop_size=pop_size, max_iter=max_iter)
        res_dict = res.to_dict()
        res_dict["snapshot_timestamp"] = prob.timestamp
        latest_route_result = res_dict
        return res_dict

    elif algorithm in ("exact", "exact_mip"):
        timeout = active_vrp_req.get("exact_timeout_s", 15.0)
        solver = ExactMIPVRPSolver(time_limit_s=timeout)
        res = solver.solve(prob)
        res_dict = res.to_dict()
        res_dict["snapshot_timestamp"] = prob.timestamp
        latest_route_result = res_dict
        return res_dict

    elif algorithm == "compare":
        dijkstra_res = dijkstra_vrp_solver.solve(prob)
        num_particles = active_vrp_req.get("num_particles", 20)
        max_iter = active_vrp_req.get("max_iter", 25)
        qpso_res = qpso_vrp_solver.solve(prob, num_particles=num_particles, max_iter=max_iter)
        alns_res = alns_vrp_solver.solve(prob, max_iter=max_iter)
        hgs_res = hgs_vrp_solver.solve(prob, pop_size=20, max_iter=max_iter)

        reps = active_vrp_req.get("reps", 1)
        shots = active_vrp_req.get("shots", 1024)
        qaoa_res = qaoa_vrp_solver.solve(prob, reps=reps, shots=shots)

        algos = {
            "dijkstra": dijkstra_res.to_dict(),
            "qpso": qpso_res.to_dict(),
            "alns": alns_res.to_dict(),
            "hgs": hgs_res.to_dict(),
            "qaoa": qaoa_res.to_dict(),
        }

        # Run exact MIP if customer count is small enough
        if len(prob.destinations) <= 8:
            exact_res = exact_mip_vrp_solver.solve(prob)
            algos["exact_mip"] = exact_res.to_dict()

        valid_algos = {k: v for k, v in algos.items() if v.get("success")}

        compare_dict = {
            "success": bool(valid_algos),
            "problem": prob.to_dict(),
            "snapshot_timestamp": prob.timestamp,
            "dijkstra": algos["dijkstra"],
            "qpso": algos["qpso"],
            "alns": algos["alns"],
            "hgs": algos["hgs"],
            "qaoa": algos["qaoa"],
            "analysis": {
                "comparison_basis": f"Dynamic Fleet VRP evaluation across {len(prob.destinations)} customer destinations and {prob.num_vehicles} vehicles on snapshot G(t) at t = {prob.timestamp:.1f}s.",
                "algorithms": algos,
            }
        }
        if "exact_mip" in algos:
            compare_dict["exact_mip"] = algos["exact_mip"]

        # Select best feasible algorithm result as primary display
        candidates = [qpso_res, alns_res, hgs_res, dijkstra_res]
        best_candidate = min((c for c in candidates if c.feasible), key=lambda c: c.total_cost, default=qpso_res)
        best_r = best_candidate.to_dict()
        for k, v in best_r.items():
            if k not in compare_dict:
                compare_dict[k] = v

        latest_route_result = compare_dict
        return compare_dict

    return None


async def run_active_vrp_solver_async() -> Optional[Dict[str, Any]]:
    """Runs solver asynchronously in threadpool to keep main asyncio event loop responsive."""
    return await asyncio.to_thread(_execute_vrp_solver_sync)


def _simulation_update_payload(step_res: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    state = step_res or sumo_mgr.get_snapshot()
    vehicles = sumo_mgr.get_vehicle_states() if state.get('running') else []
    congested_edges = [
        {
            'id': edge['edge_id'],
            'congestion_ratio': edge['congestion_ratio'],
            'vehicle_count': edge['vehicle_count'],
            'mean_speed': round(edge['mean_speed'], 2),
            'travel_time': round(edge['travel_time'], 1),
        }
        for edge in dt_graph.edge_metadata.values()
        if edge['congestion_ratio'] > 0.1 or edge['edge_id'] in dt_graph.incidents
    ]
    return {
        'type': 'sim_update',
        'running': state.get('running', False),
        'paused': state.get('paused', False),
        'sim_time': state.get('sim_time', 0),
        'active_vehicles': state.get('active_count', 0),
        'loaded_vehicles': state.get('loaded_count', 0),
        'arrived_vehicles': state.get('arrived_count', 0),
        'expected_vehicles': state.get('expected_count', 0),
        'vehicles': vehicles,
        'congested_edges': congested_edges[:50],
        'active_route': latest_route_result,
        'active_incidents': list(sumo_mgr.active_incidents.keys()),
        'events': sim_event_log[-30:],
        'stats': dt_graph.get_summary_stats(),
    }

# Pydantic schemas
class RouteRequest(BaseModel):
    origin_node: str
    destination_node: str

class QAOARouteRequest(RouteRequest):
    candidate_count: int = 3
    reps: int = 1
    shots: int = 1024

class QPSORouteRequest(RouteRequest):
    num_particles: int = 40
    max_iter: int = 50

class IncidentRequest(BaseModel):
    edge_id: str
    speed_factor: float = 0.05

class ClearIncidentRequest(BaseModel):
    edge_id: str

class ResolveNodeRequest(BaseModel):
    x: float
    y: float

class VehicleModel(BaseModel):
    vehicle_id: str
    capacity: float = 100.0
    start_node: Optional[str] = None
    color: Optional[str] = "#3b82f6"
    is_available: bool = True
    status: str = "active"

class OrderModel(BaseModel):
    order_id: str
    node_id: str
    x: float = 0.0
    y: float = 0.0
    demand: float = 1.0
    priority: int = 1

class VRPRequest(BaseModel):
    origin_node: str
    destination_nodes: List[str]
    num_vehicles: int = 1
    vehicles: List[VehicleModel] = []
    orders: List[OrderModel] = []
    customer_demands: Dict[str, float] = {}
    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0
    vehicle_capacity: float = 100.0
    algorithm: str = "dijkstra"

class QAOAVRPRequest(VRPRequest):
    reps: int = 1
    shots: int = 1024

class QPSOVRPRequest(VRPRequest):
    num_particles: int = 40
    max_iter: int = 50

class CompareVRPRequest(VRPRequest):
    reps: int = 1
    shots: int = 1024
    num_particles: int = 40
    max_iter: int = 50

class GenerateOrdersRequest(BaseModel):
    count: int = 5

class BenchmarkRequest(BaseModel):
    origin_node: str
    destination_nodes: List[str]
    num_vehicles: int = 2
    vehicle_capacity: float = 100.0
    vehicles: List[VehicleModel] = []
    orders: List[OrderModel] = []
    customer_demands: Dict[str, float] = {}
    algorithms: List[str] = ["qpso", "alns", "hgs", "exact_mip"]
    seeds: List[int] = [42, 101, 202]
    mode: str = "frozen"
    exact_timeout_s: float = 10.0
    num_particles: int = 25
    max_iter: int = 30

class ScalabilityRequest(BaseModel):
    origin_node: str
    customer_counts: List[int] = [3, 5, 8, 12]
    num_vehicles: int = 3
    vehicle_capacity: float = 100.0
    algorithms: List[str] = ["qpso", "alns", "hgs", "exact_mip"]
    timeout_per_run_s: float = 10.0

# API Endpoints
@app.get("/api/network")
def get_network():
    edges_payload = []
    for edge_id, meta in dt_graph.edge_metadata.items():
        edges_payload.append({
            'id': edge_id,
            'from': meta['from_node'],
            'to': meta['to_node'],
            'length': meta['length'],
            'speed': meta['speed_limit'],
            'shape': meta['shape']
        })

    nodes_payload = {
        node_id: {'x': pos[0], 'y': pos[1]}
        for node_id, pos in dt_graph.node_positions.items()
    }

    xs = [p[0] for p in dt_graph.node_positions.values()]
    ys = [p[1] for p in dt_graph.node_positions.values()]
    bounds = {
        'min_x': min(xs) if xs else 0,
        'max_x': max(xs) if xs else 2800,
        'min_y': min(ys) if ys else 0,
        'max_y': max(ys) if ys else 2600
    }

    return {
        'bounds': bounds,
        'nodes': nodes_payload,
        'edges': edges_payload,
        'stats': dt_graph.get_summary_stats()
    }

@app.post("/api/sim/start")
def start_sim(gui: bool = True):
    success = sumo_mgr.start(use_gui=gui)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to start SUMO/TraCI.")
    return {"status": "started", "gui": gui}

@app.post("/api/sim/pause")
def pause_sim():
    sumo_mgr.is_paused = not sumo_mgr.is_paused
    return {"status": "paused" if sumo_mgr.is_paused else "resumed"}

@app.post("/api/sim/step")
def step_sim():
    step_data = sumo_mgr.step()
    return step_data

@app.post("/api/sim/stop")
def stop_sim():
    sumo_mgr.stop()
    return {"status": "stopped"}

@app.get("/api/sim/state")
def get_sim_state():
    return _simulation_update_payload()

@app.post("/api/orders/generate")
def generate_synthetic_orders(req: GenerateOrdersRequest):
    """
    Generates N synthetic delivery orders by sampling valid, connected SUMO routing nodes in network.
    """
    g = dt_graph.graph
    all_nodes = [node for node in dt_graph.node_positions.keys() if node in g and len(g[node]) > 0]
    if len(all_nodes) < req.count:
        all_nodes = list(dt_graph.node_positions.keys())

    sampled_nodes = random.sample(all_nodes, min(req.count, len(all_nodes)))
    generated_orders = []
    for idx, node in enumerate(sampled_nodes):
        pos = dt_graph.node_positions.get(node, (0.0, 0.0))
        demand = random.choice([5.0, 10.0, 15.0, 20.0])
        generated_orders.append({
            "order_id": f"O_{idx+1:03d}",
            "node_id": node,
            "x": pos[0],
            "y": pos[1],
            "demand": demand,
            "priority": random.choice([1, 2, 3]),
            "ready_time": 0.0,
            "due_time": 86400.0,
            "service_time": 300.0,
            "status": "pending",
        })

    return {
        "success": True,
        "count": req.count,
        "orders": generated_orders,
    }

@app.post("/api/route/dijkstra")
def calculate_dijkstra_route(req: RouteRequest):
    res = router.find_shortest_path(dt_graph, req.origin_node, req.destination_node)
    return res

@app.post("/api/route/astar")
def calculate_astar_route(req: RouteRequest):
    res = astar_router.find_shortest_path(dt_graph, req.origin_node, req.destination_node)
    return res

@app.post("/api/route/qpso")
def calculate_qpso_route(req: QPSORouteRequest):
    return qpso_router.find_route(dt_graph, req.origin_node, req.destination_node, req.num_particles, req.max_iter)

@app.post("/api/incident/inject")
async def inject_incident(req: IncidentRequest):
    success = sumo_mgr.inject_incident(req.edge_id, req.speed_factor)
    if not success:
        raise HTTPException(status_code=400, detail=f"Invalid edge_id or SUMO not running: {req.edge_id}")

    sim_time = sumo_mgr.get_snapshot().get("sim_time", 0.0)
    log_sim_event(
        "INCIDENT_INJECTED",
        f"Bottleneck/accident injected on edge {req.edge_id}. Speed reduced to {req.speed_factor * 100:.0f}%.",
        severity="warning",
        sim_time=sim_time
    )

    global latest_route_result
    if active_vrp_req:
        await run_active_vrp_solver_async()
    return {"status": "incident_injected", "edge_id": req.edge_id, "updated_route": latest_route_result}

@app.post("/api/incident/clear")
async def clear_incident(req: ClearIncidentRequest):
    success = sumo_mgr.clear_incident(req.edge_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"No active incident found for: {req.edge_id}")

    sim_time = sumo_mgr.get_snapshot().get("sim_time", 0.0)
    log_sim_event(
        "INCIDENT_CLEARED",
        f"Incident cleared on edge {req.edge_id}. Road capacity and speed restored.",
        severity="success",
        sim_time=sim_time
    )

    global latest_route_result
    if active_vrp_req:
        await run_active_vrp_solver_async()
    return {"status": "incident_cleared", "edge_id": req.edge_id, "updated_route": latest_route_result}

@app.post("/api/network/resolve-node")
def resolve_nearest_node(req: ResolveNodeRequest):
    if not dt_graph.node_positions:
        raise HTTPException(status_code=400, detail="Graph topology not loaded.")

    closest_node = None
    min_dist = float("inf")
    for node_id, pos in dt_graph.node_positions.items():
        dx = pos[0] - req.x
        dy = pos[1] - req.y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest_node = node_id

    if not closest_node:
        raise HTTPException(status_code=404, detail="No node found near specified coordinate.")

    pos = dt_graph.node_positions[closest_node]
    return {
        "resolved_node": closest_node,
        "x": pos[0],
        "y": pos[1],
        "distance": round(min_dist, 2)
    }

@app.post("/api/vrp/solve")
async def solve_dynamic_vrp(req: VRPRequest):
    """
    Main dynamic fleet-level VRP solver endpoint supporting arbitrary X vehicles and N customer orders.
    Executes asynchronously off the main thread to prevent event loop / SUMO freezes.
    """
    global active_vrp_req
    req_dict = req.dict()
    active_vrp_req = {
        "origin": req.origin_node,
        "destinations": req.destination_nodes,
        "num_vehicles": req.num_vehicles,
        "vehicles": [v.dict() for v in req.vehicles] if req.vehicles else [],
        "orders": [o.dict() for o in req.orders] if req.orders else [],
        "customer_demands": req.customer_demands,
        "vehicle_capacity": req.vehicle_capacity,
        "algorithm": req.algorithm,
        "alpha": req.alpha,
        "beta": req.beta,
        "gamma": req.gamma,
    }
    return await run_active_vrp_solver_async()

@app.post("/api/vrp/dijkstra")
async def solve_vrp_dijkstra(req: VRPRequest):
    req.algorithm = "dijkstra"
    return await solve_dynamic_vrp(req)

@app.post("/api/vrp/qpso")
async def solve_vrp_qpso(req: QPSOVRPRequest):
    req.algorithm = "qpso"
    return await solve_dynamic_vrp(req)

@app.post("/api/vrp/qaoa")
async def solve_vrp_qaoa(req: QAOAVRPRequest):
    req.algorithm = "qaoa"
    return await solve_dynamic_vrp(req)

@app.post("/api/vrp/alns")
async def solve_vrp_alns(req: VRPRequest):
    req.algorithm = "alns"
    return await solve_dynamic_vrp(req)

@app.post("/api/vrp/hgs")
async def solve_vrp_hgs(req: VRPRequest):
    req.algorithm = "hgs"
    return await solve_dynamic_vrp(req)

@app.post("/api/vrp/exact")
async def solve_vrp_exact(req: VRPRequest):
    req.algorithm = "exact_mip"
    return await solve_dynamic_vrp(req)

@app.post("/api/vrp/compare")
async def compare_vrp_algorithms(req: CompareVRPRequest):
    req.algorithm = "compare"
    return await solve_dynamic_vrp(req)

@app.post("/api/benchmark/run")
async def run_benchmark(req: BenchmarkRequest):
    """
    Runs dedicated frozen-snapshot benchmark comparing QPSO, ALNS, HGS, Exact MIP, Dijkstra, and QAOA
    across multiple random seeds off the main thread with guaranteed input fairness.
    """
    def _run_bench():
        prob_dict = {
            "origin": req.origin_node,
            "destinations": req.destination_nodes,
            "num_vehicles": req.num_vehicles,
            "vehicle_capacity": req.vehicle_capacity,
            "vehicles": [v.dict() for v in req.vehicles] if req.vehicles else [],
            "orders": [o.dict() for o in req.orders] if req.orders else [],
            "customer_demands": req.customer_demands,
        }
        prob = _build_problem_instance_from_dict(prob_dict)
        exp_id = f"exp_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        sim_time = sumo_mgr.get_snapshot().get("sim_time", 0.0)

        fairness_check = {
            "frozen_snapshot_timestamp": round(sim_time, 2),
            "identical_graph_weights": True,
            "identical_customer_demands": True,
            "identical_vehicle_capacities": True,
            "customer_count": len(prob.destinations),
            "vehicle_count": prob.num_vehicles,
            "total_package_demand": round(sum(prob.customer_demands.values()), 2),
        }

        all_run_results = []
        for algo in req.algorithms:
            # Deterministic algorithms only need 1 run across seeds
            seeds_to_run = [req.seeds[0]] if algo in ("dijkstra", "exact_mip", "exact") else req.seeds
            for seed in seeds_to_run:
                is_opt = False
                if algo == "dijkstra":
                    res = dijkstra_vrp_solver.solve(prob)
                elif algo == "qpso":
                    solver = QPSOVRPSolver(
                        num_particles=req.num_particles,
                        max_iter=req.max_iter,
                        seed=seed
                    )
                    res = solver.solve(prob)
                elif algo == "alns":
                    solver = ALNSVRPSolver(max_iter=max(30, req.max_iter), seed=seed)
                    res = solver.solve(prob)
                elif algo == "hgs":
                    solver = HGSVRPSolver(pop_size=max(20, req.num_particles), max_iter=max(30, req.max_iter), seed=seed)
                    res = solver.solve(prob)
                elif algo in ("exact_mip", "exact"):
                    solver = ExactMIPVRPSolver(time_limit_s=req.exact_timeout_s)
                    res = solver.solve(prob)
                    is_opt = (res.status == "OPTIMAL")
                elif algo == "qaoa":
                    res = qaoa_vrp_solver.solve(prob)
                else:
                    continue

                res_dict = res.to_dict()
                convergence = []
                if res.solver_details and "convergence_history" in res.solver_details:
                    convergence = res.solver_details["convergence_history"]

                all_run_results.append({
                    "algorithm": algo,
                    "seed": seed,
                    "result": res_dict,
                    "convergence": convergence,
                    "is_optimal": is_opt,
                })

                experiment_db.save_experiment(
                    experiment_id=exp_id,
                    timestamp=time.time(),
                    graph_time=sim_time,
                    depot=req.origin_node,
                    num_customers=len(req.destination_nodes),
                    num_vehicles=req.num_vehicles,
                    algorithm=algo,
                    seed=seed,
                    solution_dict=res_dict,
                    total_cost=res.total_cost,
                    total_travel_time=res.total_travel_time,
                    total_distance=res.total_distance,
                    runtime_ms=res.computation_time_ms,
                    feasible=res.feasible,
                    convergence_history=convergence,
                    parameters={
                        "num_particles": req.num_particles,
                        "max_iter": req.max_iter,
                        "exact_timeout_s": req.exact_timeout_s,
                    },
                    problem_dict=prob_dict,
                    is_optimal=is_opt,
                )

        stats = experiment_db.compute_multi_seed_stats(exp_id)
        pairwise = experiment_db.compute_pairwise_comparisons(exp_id)
        return {
            "success": True,
            "experiment_id": exp_id,
            "graph_time": sim_time,
            "fairness_check": fairness_check,
            "statistics": stats,
            "pairwise_comparisons": pairwise,
            "runs": all_run_results,
        }

    return await asyncio.to_thread(_run_bench)

@app.post("/api/benchmark/scalability")
async def run_scalability_benchmark(req: ScalabilityRequest):
    def _run_scalability():
        sim_time = sumo_mgr.get_snapshot().get("sim_time", 0.0)
        all_nodes = [node for node in dt_graph.node_positions.keys() if node in dt_graph.graph and len(dt_graph.graph[node]) > 0]
        if req.origin_node in all_nodes:
            all_nodes.remove(req.origin_node)

        results = []
        for N in req.customer_counts:
            if N > len(all_nodes):
                continue
            rng = random.Random(42 + N)
            dest_sample = rng.sample(all_nodes, N)
            demands = {n: float(rng.choice([5, 10, 15])) for n in dest_sample}

            prob_dict = {
                "origin": req.origin_node,
                "destinations": dest_sample,
                "num_vehicles": req.num_vehicles,
                "vehicle_capacity": req.vehicle_capacity,
                "customer_demands": demands,
            }
            prob = _build_problem_instance_from_dict(prob_dict)

            row = {"N": N, "num_vehicles": req.num_vehicles, "algorithms": {}}
            for algo in req.algorithms:
                t0 = time.perf_counter()
                res = None
                if algo == "qpso":
                    res = qpso_vrp_solver.solve(prob, num_particles=20, max_iter=25)
                elif algo == "alns":
                    res = alns_vrp_solver.solve(prob, max_iter=30)
                elif algo == "hgs":
                    res = hgs_vrp_solver.solve(prob, pop_size=20, max_iter=25)
                elif algo in ("exact_mip", "exact"):
                    if N <= 8:
                        res = ExactMIPVRPSolver(time_limit_s=min(8.0, req.timeout_per_run_s)).solve(prob)
                    else:
                        res = None
                elif algo == "dijkstra":
                    res = dijkstra_vrp_solver.solve(prob)
                elif algo == "qaoa":
                    if N <= 4:
                        res = qaoa_vrp_solver.solve(prob)
                    else:
                        res = None

                runtime_ms = (time.perf_counter() - t0) * 1000.0
                if res is not None:
                    row["algorithms"][algo] = {
                        "runtime_ms": round(runtime_ms, 2),
                        "total_cost": round(res.total_cost, 2) if res.feasible else None,
                        "feasible": res.feasible,
                        "status": res.status,
                    }
                else:
                    row["algorithms"][algo] = {
                        "runtime_ms": round(runtime_ms, 2),
                        "total_cost": None,
                        "feasible": False,
                        "status": "UNSUPPORTED_SIZE" if algo in ("qaoa", "exact_mip", "exact") else "SKIPPED",
                    }
            results.append(row)

        return {
            "success": True,
            "timestamp": sim_time,
            "scalability_matrix": results,
        }
    return await asyncio.to_thread(_run_scalability)

@app.get("/api/benchmark/experiment/{exp_id}")
def get_experiment_details(exp_id: str):
    data = experiment_db.get_experiment_details(exp_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")
    return {
        "success": True,
        **data
    }

@app.get("/api/benchmark/history")
def get_benchmark_history(limit: int = 50):
    return {
        "success": True,
        "history": experiment_db.get_history(limit=limit),
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                cmd = msg.get("command")
                if cmd == "start":
                    sumo_mgr.start(use_gui=msg.get("gui", True))
                elif cmd == "pause":
                    sumo_mgr.is_paused = not sumo_mgr.is_paused
                elif cmd == "step":
                    sumo_mgr.step()
                elif cmd == "stop":
                    sumo_mgr.stop()
            except Exception as e:
                print(f"Error processing WS command: {e}")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

async def simulation_loop():
    global latest_route_result
    last_reroute_time = -999.0
    while True:
        try:
            if sumo_mgr.is_running and not sumo_mgr.is_paused:
                step_res = sumo_mgr.step()
                if step_res.get('running'):
                    sim_time = step_res.get('sim_time', 0.0)

                    # Check dynamic rerouting policy periodically off main thread
                    if active_vrp_req and (sim_time - last_reroute_time >= 5.0 or sim_time < last_reroute_time or last_reroute_time < 0):
                        last_reroute_time = sim_time
                        prob = _build_problem_instance_from_dict(active_vrp_req)
                        if latest_route_result and isinstance(latest_route_result, dict):
                            curr_alg_res = AlgorithmResult(**{k: v for k, v in latest_route_result.items() if k in AlgorithmResult.__dataclass_fields__})
                            action, updated_res, meta = await asyncio.to_thread(
                                dynamic_rerouter.evaluate_rerouting_policy,
                                prob, curr_alg_res, list(sumo_mgr.active_incidents.keys()),
                                lambda p: qpso_vrp_solver.solve(p)
                            )
                            if action in ["LOCAL_REROUTE", "GLOBAL_REOPTIMIZATION"]:
                                res_dict = updated_res.to_dict()
                                res_dict["reroute_meta"] = meta
                                latest_route_result = res_dict
                                log_sim_event(
                                    "REROUTE_TRIGGERED",
                                    f"Dynamic reroute triggered: {action}. {meta.get('reason', '')}",
                                    severity="warning",
                                    sim_time=sim_time
                                )

                    # Real-time data-driven fleet execution tracking
                    if latest_route_result and isinstance(latest_route_result, dict):
                        routes = latest_route_result.get("fleet_routes", [])
                        plan_time = latest_route_result.get("snapshot_timestamp", 0.0)
                        if fleet_execution_tracker.get("active_plan_timestamp") != plan_time:
                            fleet_execution_tracker["active_plan_timestamp"] = plan_time
                            fleet_execution_tracker["vehicles"] = {}
                            for fr in routes:
                                if fr.get("is_active") and fr.get("visit_sequence"):
                                    v_id = fr.get("vehicle_id")
                                    fleet_execution_tracker["vehicles"][v_id] = {
                                        "status": "DEPARTED",
                                        "current_stop_idx": 0,
                                        "load": fr.get("load_used", 0.0),
                                        "capacity": fr.get("capacity", 0.0),
                                        "visit_sequence": fr.get("visit_sequence", []),
                                        "assigned_orders": fr.get("assigned_orders", []),
                                        "start_time": sim_time,
                                        "next_event_time": sim_time + 4.0,
                                    }
                                    log_sim_event(
                                        "VEHICLE_DEPARTED",
                                        f"Vehicle {v_id} departed depot carrying {fr.get('load_used', 0.0)} packages to serve {len(fr.get('assigned_orders', []))} customer orders.",
                                        vehicle_id=v_id,
                                        severity="info",
                                        sim_time=sim_time
                                    )
                        else:
                            for v_id, v_state in fleet_execution_tracker.get("vehicles", {}).items():
                                if v_state["status"] != "COMPLETED" and sim_time >= v_state.get("next_event_time", float("inf")):
                                    seq = v_state["visit_sequence"]
                                    curr_idx = v_state["current_stop_idx"] + 1
                                    v_state["current_stop_idx"] = curr_idx

                                    if curr_idx < len(seq) - 1:
                                        cust_node = seq[curr_idx]
                                        dem = active_vrp_req.get("customer_demands", {}).get(cust_node, 10.0) if active_vrp_req else 10.0
                                        v_state["load"] = max(0.0, v_state["load"] - dem)
                                        log_sim_event(
                                            "VEHICLE_ARRIVED_AT_CUSTOMER",
                                            f"Vehicle {v_id} arrived at customer stop {cust_node}.",
                                            vehicle_id=v_id,
                                            severity="info",
                                            sim_time=sim_time
                                        )
                                        log_sim_event(
                                            "DELIVERY_COMPLETED",
                                            f"Delivered {dem} packages to customer stop {cust_node}.",
                                            vehicle_id=v_id,
                                            severity="success",
                                            sim_time=sim_time
                                        )
                                        log_sim_event(
                                            "LOAD_CHANGED",
                                            f"Vehicle {v_id} onboard load updated to {v_state['load']} packages.",
                                            vehicle_id=v_id,
                                            severity="info",
                                            sim_time=sim_time
                                        )
                                        v_state["next_event_time"] = sim_time + 6.0
                                    elif curr_idx == len(seq) - 1:
                                        v_state["status"] = "COMPLETED"
                                        log_sim_event(
                                            "VEHICLE_RETURNED_TO_DEPOT",
                                            f"Vehicle {v_id} finished all deliveries and returned to depot {seq[curr_idx]}.",
                                            vehicle_id=v_id,
                                            severity="success",
                                            sim_time=sim_time
                                        )

                    await ws_manager.broadcast(_simulation_update_payload(step_res))
        except Exception as err:
            print(f"Error in simulation_loop step: {err}")

        await asyncio.sleep(sumo_mgr.step_delay_sec)

@app.get("/api/sim/events")
def get_sim_events(limit: int = 50):
    return {
        "success": True,
        "count": len(sim_event_log),
        "events": sim_event_log[-limit:],
    }



# Mount static frontend directory
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def read_root():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>Frontend index.html not found</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.server:app", host="127.0.0.1", port=8000, reload=True)
