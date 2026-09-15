import unittest
import numpy as np
import tempfile
import os
import asyncio

from backend.graph.dynamic_graph import DynamicTrafficGraph
from backend.vrp.problem_instance import ProblemInstance, VehicleConfig
from backend.vrp.alns_vrp import ALNSVRPSolver
from backend.vrp.hgs_vrp import HGSVRPSolver
from backend.vrp.experiment_db import ExperimentDatabase
from backend.api.server import (
    solve_vrp_alns,
    solve_vrp_hgs,
    get_sim_events,
    get_network,
    VRPRequest
)

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

class TestSolversAndBenchmarks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = build_test_graph()
        cls.prob = ProblemInstance(
            origin='n0',
            destinations=['n1', 'n2', 'n3', 'n4', 'n5'],
            graph=cls.graph,
            vehicle_capacity=30.0,
            num_vehicles=3,
            customer_demands={'n1': 10.0, 'n2': 10.0, 'n3': 10.0, 'n4': 10.0, 'n5': 10.0}
        )

    # 1. ALNS Internal Mechanics
    def test_alns_solver_mechanics(self):
        solver = ALNSVRPSolver(max_iter=30, seed=42)
        res = solver.solve(self.prob)
        self.assertTrue(res.feasible)
        self.assertGreater(res.total_cost, 0.0)

        sd = res.solver_details
        self.assertIsNotNone(sd)
        self.assertIn('final_destroy_weights', sd)
        self.assertIn('final_repair_weights', sd)
        self.assertIn('convergence_history', sd)
        self.assertGreaterEqual(len(sd['convergence_history']), 30)

        # Check weights remain strictly positive
        for op, w in sd['final_destroy_weights'].items():
            self.assertGreater(w, 0.0)
        for op, w in sd['final_repair_weights'].items():
            self.assertGreater(w, 0.0)

    # 2. HGS Internal Mechanics
    def test_hgs_solver_mechanics(self):
        solver = HGSVRPSolver(pop_size=15, max_iter=25, seed=42)
        res = solver.solve(self.prob)
        self.assertTrue(res.feasible)
        self.assertGreater(res.total_cost, 0.0)

        sd = res.solver_details
        self.assertIsNotNone(sd)
        self.assertIn('final_omega_cap', sd)
        self.assertIn('average_diversity', sd)
        self.assertIn('feasible_solutions_count', sd)
        self.assertGreaterEqual(sd['feasible_solutions_count'], 1)

    # 3. Solver Independence (No copying, distinct routes or search paths)
    def test_solver_independence(self):
        alns = ALNSVRPSolver(max_iter=25, seed=777)
        hgs = HGSVRPSolver(pop_size=15, max_iter=25, seed=777)

        res_alns = alns.solve(self.prob)
        res_hgs = hgs.solve(self.prob)

        self.assertIn('ALNS', res_alns.algorithm)
        self.assertIn('HGS', res_hgs.algorithm)
        # Both must find valid solutions independently
        self.assertTrue(res_alns.feasible)
        self.assertTrue(res_hgs.feasible)

    # 4. Wilcoxon Statistical Significance Engine
    def test_wilcoxon_paired_test(self):
        # Use a temporary SQLite database
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, 'test_experiments.db')

        try:
            db = ExperimentDatabase(db_path=db_path)
            exp_id = 'test_exp_wilcoxon_01'

            # Seed n=6 paired runs where HGS strictly dominates ALNS across all seeds
            # (Minimum n=6 required for two-sided Wilcoxon to achieve p < 0.05: p = 2 * (1/2)^6 = 0.03125)
            for s in [10, 20, 30, 40, 50, 60]:
                db.save_experiment(
                    experiment_id=exp_id,
                    timestamp=100.0,
                    graph_time=0.0,
                    depot='n0',
                    num_customers=5,
                    num_vehicles=3,
                    algorithm='alns',
                    seed=s,
                    solution_dict={'test': True},
                    total_cost=100.0 + s * 0.5,
                    total_travel_time=100.0,
                    total_distance=1000.0,
                    runtime_ms=15.0,
                    feasible=True,
                )
                db.save_experiment(
                    experiment_id=exp_id,
                    timestamp=100.0,
                    graph_time=0.0,
                    depot='n0',
                    num_customers=5,
                    num_vehicles=3,
                    algorithm='hgs',
                    seed=s,
                    solution_dict={'test': True},
                    total_cost=85.0 + s * 0.4, # HGS strictly lower
                    total_travel_time=85.0,
                    total_distance=850.0,
                    runtime_ms=25.0,
                    feasible=True,
                )

            comp = db.compute_wilcoxon_test(exp_id, 'alns', 'hgs')
            self.assertEqual(comp['n_pairs'], 6)
            self.assertIsNotNone(comp['p_value'])
            self.assertLess(comp['p_value'], 0.05)
            self.assertTrue(comp['is_significant'])
            self.assertEqual(comp['winner'], 'hgs')
            self.assertIn('achieves statistically significantly lower objective cost', comp['conclusion'])

            # Test all-zero differences edge case -> tie with p=1.0
            exp_id_tie = 'test_exp_tie_02'
            for s in [1, 2, 3]:
                for algo in ['alns', 'qpso']:
                    db.save_experiment(
                        experiment_id=exp_id_tie,
                        timestamp=100.0,
                        graph_time=0.0,
                        depot='n0',
                        num_customers=5,
                        num_vehicles=3,
                        algorithm=algo,
                        seed=s,
                        solution_dict={'test': True},
                        total_cost=120.0, # identical
                        total_travel_time=120.0,
                        total_distance=1200.0,
                        runtime_ms=10.0,
                        feasible=True,
                    )
            comp_tie = db.compute_wilcoxon_test(exp_id_tie, 'alns', 'qpso')
            self.assertEqual(comp_tie['p_value'], 1.0)
            self.assertEqual(comp_tie['winner'], 'tie')
            self.assertFalse(comp_tie['is_significant'])
            self.assertIn('No statistically significant difference', comp_tie['conclusion'])
        finally:
            import gc
            gc.collect()
            try:
                if os.path.exists(db_path):
                    os.remove(db_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except Exception:
                pass

    # 5. REST API Endpoints Verification
    def test_api_vrp_endpoints(self):
        net_data = get_network()
        all_nodes = list(net_data['nodes'].keys())
        self.assertGreater(len(all_nodes), 5)

        origin = all_nodes[0]
        destinations = all_nodes[1:4]

        # Test solve_vrp_alns
        req_alns = VRPRequest(
            origin_node=origin,
            destination_nodes=destinations,
            num_vehicles=2,
            vehicle_capacity=50.0,
            customer_demands={d: 10.0 for d in destinations},
            algorithm='alns'
        )
        data_alns = asyncio.run(solve_vrp_alns(req_alns))
        self.assertIn('ALNS', data_alns['algorithm'])
        self.assertIn('fleet_routes', data_alns)

        # Test solve_vrp_hgs
        req_hgs = VRPRequest(
            origin_node=origin,
            destination_nodes=destinations,
            num_vehicles=2,
            vehicle_capacity=50.0,
            customer_demands={d: 10.0 for d in destinations},
            algorithm='hgs'
        )
        data_hgs = asyncio.run(solve_vrp_hgs(req_hgs))
        self.assertIn('HGS', data_hgs['algorithm'])
        self.assertIn('fleet_routes', data_hgs)

        # Test get_sim_events
        events_data = get_sim_events()
        self.assertTrue(events_data['success'])
        self.assertIsInstance(events_data['events'], list)

if __name__ == '__main__':
    unittest.main()
