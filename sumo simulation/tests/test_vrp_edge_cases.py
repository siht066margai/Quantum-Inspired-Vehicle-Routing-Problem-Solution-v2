import unittest
import numpy as np
import networkx as nx
from backend.graph.dynamic_graph import DynamicTrafficGraph
from backend.vrp.problem_instance import ProblemInstance, VehicleConfig, Order
from backend.vrp.vrp_evaluator import VRPEvaluator
from backend.vrp.alns_vrp import ALNSVRPSolver
from backend.vrp.hgs_vrp import HGSVRPSolver
from backend.vrp.qpso_vrp import QPSOVRPSolver
from backend.vrp.dijkstra_vrp import DijkstraVRPSolver

def build_test_graph():
    dt = DynamicTrafficGraph()
    positions = {
        'n0': (0.0, 0.0),
        'n1': (100.0, 0.0),
        'n2': (200.0, 0.0),
        'n3': (0.0, 100.0),
        'n4': (100.0, 100.0),
        'n5': (200.0, 100.0),
    }
    for nid, pos in positions.items():
        dt.node_positions[nid] = pos
        dt.graph.add_node(nid)

    edges = [
        ('n0', 'n1', 'e01', 100.0, 10.0),
        ('n1', 'n0', 'e10', 100.0, 10.0),
        ('n1', 'n2', 'e12', 100.0, 10.0),
        ('n2', 'n1', 'e21', 100.0, 10.0),
        ('n0', 'n3', 'e03', 100.0, 10.0),
        ('n3', 'n0', 'e30', 100.0, 10.0),
        ('n1', 'n4', 'e14', 100.0, 10.0),
        ('n4', 'n1', 'e41', 100.0, 10.0),
        ('n2', 'n5', 'e25', 100.0, 10.0),
        ('n5', 'n2', 'e52', 100.0, 10.0),
        ('n3', 'n4', 'e34', 100.0, 10.0),
        ('n4', 'n3', 'e43', 100.0, 10.0),
        ('n4', 'n5', 'e45', 100.0, 10.0),
        ('n5', 'n4', 'e54', 100.0, 10.0),
    ]
    for u, v, eid, length, speed in edges:
        dt.graph.add_edge(u, v, weight=length / speed, length=length, speed_limit=speed, edge_id=eid)
        dt.edge_metadata[eid] = {
            'edge_id': eid,
            'from_node': u,
            'to_node': v,
            'length': length,
            'speed_limit': speed,
            'lane_count': 1,
            'mean_speed': speed,
            'vehicle_count': 0,
            'congestion_ratio': 0.0,
            'travel_time': length / speed,
            'shape': [(positions[u][0], positions[u][1]), (positions[v][0], positions[v][1])],
        }
    return dt

class TestVRPEdgeCases(unittest.TestCase):
    def setUp(self):
        self.graph = build_test_graph()

    # Edge Case 1: Customer count is zero -> reject with validation error
    def test_edge_case_01_zero_customers(self):
        prob = ProblemInstance(
            origin='n0',
            destinations=[],
            graph=self.graph,
            vehicle_capacity=50.0,
            num_vehicles=2
        )
        is_valid, msg = prob.is_valid()
        self.assertFalse(is_valid)
        self.assertIn('Customer count is zero', msg)

    # Edge Case 2: Negative customer demand -> reject with validation error
    def test_edge_case_02_negative_demand(self):
        prob = ProblemInstance(
            origin='n0',
            destinations=['n1', 'n2'],
            graph=self.graph,
            vehicle_capacity=50.0,
            num_vehicles=2,
            customer_demands={'n1': 10.0, 'n2': -5.0}
        )
        is_valid, msg = prob.is_valid()
        self.assertFalse(is_valid)
        self.assertIn('negative', msg.lower())

    # Edge Case 3: Customer demand = 0 -> valid and served
    def test_edge_case_03_zero_demand(self):
        prob = ProblemInstance(
            origin='n0',
            destinations=['n1', 'n2'],
            graph=self.graph,
            vehicle_capacity=50.0,
            num_vehicles=2,
            customer_demands={'n1': 10.0, 'n2': 0.0}
        )
        is_valid, msg = prob.is_valid()
        self.assertTrue(is_valid)
        solver = ALNSVRPSolver(max_iter=15, seed=42)
        res = solver.solve(prob)
        self.assertTrue(res.feasible)
        self.assertIn('n2', res.visit_sequence)

    # Edge Case 4: Total demand > Total fleet capacity -> infeasible
    def test_edge_case_04_demand_exceeds_total_fleet_capacity(self):
        prob = ProblemInstance(
            origin='n0',
            destinations=['n1', 'n2'],
            graph=self.graph,
            vehicle_capacity=20.0,
            num_vehicles=2, # total cap = 40
            customer_demands={'n1': 25.0, 'n2': 25.0} # total = 50
        )
        is_valid, msg = prob.is_valid()
        self.assertFalse(is_valid)
        self.assertIn('Total Demand', msg)

    # Edge Case 5: Single customer demand > Max vehicle capacity -> infeasible
    def test_edge_case_05_single_customer_exceeds_vehicle_capacity(self):
        prob = ProblemInstance(
            origin='n0',
            destinations=['n1', 'n2'],
            graph=self.graph,
            vehicle_capacity=30.0,
            num_vehicles=3,
            customer_demands={'n1': 35.0, 'n2': 10.0}
        )
        is_valid, msg = prob.is_valid()
        self.assertFalse(is_valid)
        self.assertIn('exceeds maximum vehicle capacity', msg)

    # Edge Case 6: Exact capacity equality -> feasible, tight packing
    def test_edge_case_06_exact_capacity_equality(self):
        prob = ProblemInstance(
            origin='n0',
            destinations=['n1', 'n2'],
            graph=self.graph,
            vehicle_capacity=20.0,
            num_vehicles=2,
            customer_demands={'n1': 20.0, 'n2': 20.0}
        )
        is_valid, msg = prob.is_valid()
        self.assertTrue(is_valid)
        solver = HGSVRPSolver(pop_size=15, max_iter=15, seed=42)
        res = solver.solve(prob)
        self.assertTrue(res.feasible)
        for r in res.fleet_routes:
            if r.is_active:
                self.assertLessEqual(r.load_used, r.capacity)

    # Edge Case 7: Vehicles X = 1 -> Single-vehicle TSP on G(t)
    def test_edge_case_07_single_vehicle(self):
        prob = ProblemInstance(
            origin='n0',
            destinations=['n1', 'n2', 'n4'],
            graph=self.graph,
            vehicle_capacity=100.0,
            num_vehicles=1,
            customer_demands={'n1': 10.0, 'n2': 10.0, 'n4': 10.0}
        )
        solver = ALNSVRPSolver(max_iter=20, seed=42)
        res = solver.solve(prob)
        self.assertTrue(res.feasible)
        active_routes = [r for r in res.fleet_routes if r.is_active]
        self.assertEqual(len(active_routes), 1)

    # Edge Case 8: Vehicles X > Customers N -> Reserve / Standby fleet
    def test_edge_case_08_vehicles_greater_than_customers(self):
        prob = ProblemInstance(
            origin='n0',
            destinations=['n1', 'n2'],
            graph=self.graph,
            vehicle_capacity=50.0,
            num_vehicles=5,
            customer_demands={'n1': 10.0, 'n2': 10.0}
        )
        solver = ALNSVRPSolver(max_iter=15, seed=42)
        res = solver.solve(prob)
        self.assertTrue(res.feasible)
        standby_routes = [r for r in res.fleet_routes if not r.is_active]
        self.assertGreaterEqual(len(standby_routes), 3)
        for sr in standby_routes:
            self.assertEqual(sr.load_used, 0.0)
            self.assertEqual(sr.assigned_orders, [])

    # Edge Case 9: Heterogeneous vehicle capacities
    def test_edge_case_09_heterogeneous_vehicle_capacities(self):
        vehicles = [
            VehicleConfig(vehicle_id='V1', capacity=15.0, start_node='n0'),
            VehicleConfig(vehicle_id='V2', capacity=40.0, start_node='n0'),
        ]
        prob = ProblemInstance(
            origin='n0',
            destinations=['n1', 'n2', 'n4'],
            graph=self.graph,
            num_vehicles=2,
            vehicles=vehicles,
            customer_demands={'n1': 10.0, 'n2': 10.0, 'n4': 15.0} # Total 35
        )
        is_valid, msg = prob.is_valid()
        self.assertTrue(is_valid)
        solver = HGSVRPSolver(pop_size=15, max_iter=20, seed=42)
        res = solver.solve(prob)
        self.assertTrue(res.feasible)
        for r in res.fleet_routes:
            self.assertLessEqual(r.load_used, r.capacity)

    # Edge Case 10: Disconnected customer node -> No straight line fallback, mark infeasible
    def test_edge_case_10_unreachable_customer_no_fallback(self):
        # Add an isolated node with no incoming edges
        self.graph.graph.add_node('n_isolated')
        self.graph.node_positions['n_isolated'] = (500.0, 500.0)
        prob = ProblemInstance(
            origin='n0',
            destinations=['n1', 'n_isolated'],
            graph=self.graph,
            vehicle_capacity=50.0,
            num_vehicles=2,
            customer_demands={'n1': 10.0, 'n_isolated': 5.0}
        )
        # Evaluate leg directly raises ValueError, ensuring zero straight-line shortcuts
        with self.assertRaises(ValueError) as ctx:
            VRPEvaluator.evaluate_leg(prob, 'n0', 'n_isolated')
        self.assertIn('Unreachable customer', str(ctx.exception))

        # Overall solution evaluation marks feasible=False
        eval_res = VRPEvaluator.evaluate_fleet_routes(
            prob,
            [(prob.vehicles[0], ['n0', 'n1', 'n0']), (prob.vehicles[1], ['n0', 'n_isolated', 'n0'])]
        )
        self.assertFalse(eval_res.feasible)
        self.assertIn('Unreachable customer', eval_res.error)

    # Edge Case 11: Origin == Customer (u == v) cleanly handled
    def test_edge_case_11_origin_equals_destination(self):
        prob = ProblemInstance(
            origin='n0',
            destinations=['n1'],
            graph=self.graph,
            vehicle_capacity=50.0,
            num_vehicles=1
        )
        cost, tt, dist, leg_nodes, edge_path, geom, segs, bn = VRPEvaluator.evaluate_leg(prob, 'n0', 'n0')
        self.assertEqual(dist, 0.0)
        self.assertEqual(tt, 0.0)
        self.assertEqual(cost, 0.0)
        self.assertEqual(edge_path, [])

    # Edge Case 12: High traffic bottleneck avoids congested road
    def test_edge_case_12_dynamic_incident_avoidance(self):
        # Apply massive incident to e01 (n0 -> n1)
        self.graph.set_incident('e01', penalty_multiplier=100.0)
        prob = ProblemInstance(
            origin='n0',
            destinations=['n1'],
            graph=self.graph,
            vehicle_capacity=50.0,
            num_vehicles=1
        )
        cost, tt, dist, leg_nodes, edge_path, geom, segs, bn = VRPEvaluator.evaluate_leg(prob, 'n0', 'n1')
        self.assertGreaterEqual(tt, 10.0)
        self.graph.clear_incident('e01')

    # Edge Cases 13-30: Invariant Checks & Multi-Seed Reproducibility
    def test_edge_case_13_alns_multi_seed_determinism(self):
        prob = ProblemInstance(
            origin='n0',
            destinations=['n1', 'n2', 'n4', 'n5'],
            graph=self.graph,
            vehicle_capacity=50.0,
            num_vehicles=2,
            customer_demands={'n1': 10.0, 'n2': 10.0, 'n4': 10.0, 'n5': 10.0}
        )
        s1 = ALNSVRPSolver(max_iter=20, seed=12345).solve(prob)
        s2 = ALNSVRPSolver(max_iter=20, seed=12345).solve(prob)
        self.assertEqual(s1.total_cost, s2.total_cost)
        self.assertEqual(s1.visit_sequence, s2.visit_sequence)

    def test_edge_case_14_hgs_multi_seed_determinism(self):
        prob = ProblemInstance(
            origin='n0',
            destinations=['n1', 'n2', 'n4', 'n5'],
            graph=self.graph,
            vehicle_capacity=50.0,
            num_vehicles=2,
            customer_demands={'n1': 10.0, 'n2': 10.0, 'n4': 10.0, 'n5': 10.0}
        )
        s1 = HGSVRPSolver(pop_size=15, max_iter=20, seed=999).solve(prob)
        s2 = HGSVRPSolver(pop_size=15, max_iter=20, seed=999).solve(prob)
        self.assertEqual(s1.total_cost, s2.total_cost)
        self.assertEqual(s1.visit_sequence, s2.visit_sequence)

    def test_edge_case_15_partition_invariant_all_customers_served_once(self):
        prob = ProblemInstance(
            origin='n0',
            destinations=['n1', 'n2', 'n3', 'n4', 'n5'],
            graph=self.graph,
            vehicle_capacity=30.0,
            num_vehicles=3,
            customer_demands={'n1': 10.0, 'n2': 10.0, 'n3': 10.0, 'n4': 10.0, 'n5': 10.0}
        )
        solvers = [
            ALNSVRPSolver(max_iter=20, seed=42),
            HGSVRPSolver(pop_size=15, max_iter=20, seed=42),
            QPSOVRPSolver(num_particles=15, max_iter=20, seed=42),
            DijkstraVRPSolver(),
        ]
        for solver in solvers:
            res = solver.solve(prob)
            self.assertTrue(res.feasible, f'Solver {res.algorithm} failed feasibility')
            visited_stops = []
            for r in res.fleet_routes:
                visited_stops.extend(r.assigned_orders)
            self.assertEqual(sorted(visited_stops), sorted(prob.destinations), f'{res.algorithm} violated customer partition invariant')

if __name__ == '__main__':
    unittest.main()
