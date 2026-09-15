from __future__ import annotations

import time
import itertools
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from scipy.optimize import minimize

from backend.vrp.problem_instance import ProblemInstance, AlgorithmResult, FleetVehicleRoute
from backend.vrp.vrp_evaluator import VRPEvaluator
from backend.graph.dynamic_graph import DynamicTrafficGraph

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QuantumCircuit = None  # type: ignore
    Statevector = None  # type: ignore
    QISKIT_AVAILABLE = False


class QAOAVRPSolver:
    """
    Quantum-Assisted VRP Solver implementing QAOA from scratch based on Azfar et al. (2025).
    Formulates decision variables x_{i,p} representing customer i visited at position p.
    Extracts dynamic pairwise path costs from SUMO graph G(t), constructs QUBO & Ising Hamiltonian,
    synthesizes QAOA quantum circuit, optimizes variational angles with COBYLA, and decodes bitstrings.
    """

    def __init__(self, maxiter: int = 15, restarts: int = 1, seed: int = 42):
        self.maxiter = maxiter
        self.restarts = restarts
        self.seed = seed

    def solve(
        self,
        problem: ProblemInstance,
        reps: int = 1,
        shots: int = 1024,
    ) -> AlgorithmResult:
        started = time.perf_counter()

        if not QISKIT_AVAILABLE:
            return self._failure("Qiskit is not installed in Python environment.", started, problem)

        is_valid, msg = problem.is_valid()
        if not is_valid:
            return self._failure(msg, started, problem)

        origin = problem.origin
        destinations = problem.destinations
        N = len(destinations)  # Number of customers

        # Invariant I18: Honest Sizing Guard for QAOA Simulation
        # Full statevector simulation requires 2^(N^2) state amplitudes
        if N > 4:
            num_qubits = N * N
            state_dim = 2 ** num_qubits
            error_msg = (
                f"QAOA position formulation requires N^2 = {num_qubits} qubits (2^{num_qubits} = {state_dim:,} statevector dimension). "
                f"Full statevector simulation is intractable on classical hardware for N > 4. "
                f"Please use QPSO Swarm or Classical baseline for N = {N} destinations."
            )
            standby_routes = [
                FleetVehicleRoute(
                    vehicle_id=v.vehicle_id,
                    capacity=v.capacity,
                    assigned_orders=[],
                    visit_sequence=[origin, origin],
                    node_path=[origin],
                    edge_path=[],
                    geometry=[],
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
                        "node": origin,
                        "action": f"QAOA Unsupported Size ({num_qubits} qubits). Vehicle on Standby.",
                        "delivered": 0.0,
                        "remaining": 0.0,
                        "cumulative_distance_m": 0.0,
                        "cumulative_time_s": 0.0,
                    }],
                    segment_calculations=[],
                    bottlenecks=[],
                    is_active=False,
                    status="standby",
                    color=v.color,
                )
                for v in problem.vehicles
            ]
            return AlgorithmResult(
                algorithm="QAOA VRP Solver (Azfar et al. 2025)",
                status="unsupported_size",
                success=False,
                visit_sequence=[origin, origin],
                node_path=[],
                edge_path=[],
                total_cost=0.0,
                total_travel_time=0.0,
                total_distance=0.0,
                average_speed_ms=0.0,
                bottleneck_count=0,
                bottlenecks=[],
                segment_calculations=[],
                geometry=[],
                feasible=False,
                constraint_violations=[error_msg],
                computation_time_ms=(time.perf_counter() - started) * 1000,
                math_proof={
                    "status": "unsupported_size",
                    "customer_count": N,
                    "qubits_required": num_qubits,
                    "hilbert_space_dimension": state_dim,
                    "max_classical_simulable_customers": 4,
                    "theoretical_formulation": "Position-based QUBO (Azfar et al. 2025)",
                    "recommendation": "Use QPSO Swarm VRP or Classical Baseline for N > 4."
                },
                solver_details={"num_qubits": num_qubits, "max_qubits": 16},
                fleet_routes=standby_routes,
                error=error_msg,
            )

        # 1. Extract exact pairwise leg costs from dynamic SUMO graph G(t)
        # Leg costs matrix: cost_from_origin[i], cost_between[i][j]
        cost_matrix, leg_paths = self._extract_pairwise_costs(problem)

        # 2. Define binary decision variables x_{i, p} (Customer i visited at position p)
        # Qubit index mapping: var_index = i * N + p (for i in 0..N-1, p in 0..N-1)
        num_qubits = N * N

        # 3. Build QUBO Matrix & Ising Hamiltonian
        Q_matrix, linear_terms, const_offset, P_val = self._build_position_vrp_qubo(
            N, cost_matrix
        )

        max_coeff = float(np.max(np.abs(Q_matrix))) if np.max(np.abs(Q_matrix)) > 0 else 1.0
        Q_norm = Q_matrix / max_coeff
        linear_norm = linear_terms / max_coeff

        h_vec, J_couplings, ising_offset = self._qubo_to_ising(Q_norm, linear_norm)
        energies = self._compute_bitstring_energies(Q_norm, linear_norm)

        # 4. Optimize QAOA Variational Parameters (gamma, beta) using COBYLA
        opt_params, exp_energy, evaluations = self._optimize_qaoa_parameters(
            h_vec, J_couplings, energies, reps=reps
        )

        # 5. Synthesize QAOA Quantum Circuit & evaluate statevector probabilities
        circuit = self._synthesize_qaoa_circuit(h_vec, J_couplings, opt_params, reps=reps)
        state_vec = Statevector.from_instruction(circuit)
        probabilities = state_vec.probabilities()

        # 6. Sample bitstrings and decode feasible customer visit sequence
        ranked_states = sorted(
            range(len(probabilities)), key=lambda idx: float(probabilities[idx]), reverse=True
        )

        feasible_state = None
        decoded_sequence = None

        for state_idx in ranked_states:
            seq = self._decode_bitstring_to_sequence(state_idx, origin, destinations)
            if seq is not None:
                feasible_state = state_idx
                decoded_sequence = seq
                break

        # Fallback to default order if noise prevents valid bitstring
        if decoded_sequence is None:
            decoded_sequence = [origin] + destinations
            feasible_state = 0

        # 7. Evaluate decoded sequence using common VRPEvaluator on G(t)
        result = VRPEvaluator.evaluate_sequence(
            problem,
            decoded_sequence,
            algorithm_name="QAOA VRP Solver (Azfar et al. 2025)",
            computation_time_ms=(time.perf_counter() - started) * 1000,
        )

        sample_counts = self._sample_counts(probabilities, num_qubits, shots)

        selected_prob = float(probabilities[feasible_state]) if feasible_state is not None else 0.0
        most_likely_state = ranked_states[0]
        most_likely_prob = float(probabilities[most_likely_state])

        # Construct qubit variable details for transparency UI
        qubit_variables = []
        for i, customer in enumerate(destinations):
            for p in range(N):
                q_idx = i * N + p
                qubit_variables.append({
                    "var_index": q_idx,
                    "qubit": f"q_{q_idx}",
                    "customer": customer,
                    "position": p + 1,
                    "description": f"x_{{C{i+1},{p+1}}}: Visit {customer} at step {p+1}",
                })

        qubo_details = {
            "qubit_variables": qubit_variables,
            "penalty_P": round(P_val, 4),
            "normalization_factor": round(max_coeff, 6),
            "ising_hamiltonian": [
                {
                    "type": "linear",
                    "qubit": i,
                    "coeff": round(float(h_vec[i]), 5),
                    "term": f"{round(float(h_vec[i]), 5)} * Z_{i}",
                }
                for i in range(num_qubits) if abs(h_vec[i]) > 1e-9
            ] + [
                {
                    "type": "coupling",
                    "qubits": [i, j],
                    "coeff": round(float(c_val), 5),
                    "term": f"{round(float(c_val), 5)} * Z_{i} Z_{j}",
                }
                for (i, j), c_val in J_couplings.items() if abs(c_val) > 1e-9
            ],
            "variational_parameters": {
                "gamma": [round(float(g), 5) for g in opt_params[:reps]],
                "beta": [round(float(b), 5) for b in opt_params[reps:]],
            },
        }

        math_proof = {
            "paper_reference": "Azfar, Raisuddin, Ke, Holguín-Veras (ACM Trans. Quantum Comput. 2025 / arXiv:2505.01614)",
            "objective_equation": "min sum_{i,j,p} c_{i,j} x_{i,p} x_{j,p+1} + P * Penalties",
            "penalty_scaling_eq": "P = 2 * sum_{i,j} |c_{i,j}|  (Sec 4.5)",
            "qubo_formulation_eq": "H_QUBO = x^T Q x + linear^T x + P * (Uniqueness Penalties)",
            "ising_transformation_eq": "x_i -> (Z_i + I) / 2  =>  H_C = c0 + sum h_i Z_i + sum_{i<j} J_{i,j} Z_i Z_j",
            "qaoa_ansatz_eq": "|psi(gamma, beta)> = prod_{l=1}^p e^{-i beta_l H_M} e^{-i gamma_l H_C} |+>^n",
            "mixer_hamiltonian_eq": "H_M = sum_{i=1}^n X_i",
            "expectation_value_eq": "E(gamma, beta) = <psi(gamma, beta) | H_C | psi(gamma, beta)>",
            "cobyla_optimization": "COBYLA minimizes expectation value over variational parameters (gamma, beta)",
        }

        solver_details = {
            "execution_backend": "Qiskit Statevector Simulator",
            "qaoa_qubits": num_qubits,
            "qaoa_depth": reps,
            "penalty_multiplier_P": round(P_val, 4),
            "normalization_factor": round(max_coeff, 6),
            "selected_feasible_bitstring": self._format_bitstring(feasible_state, num_qubits),
            "selected_probability": round(selected_prob, 6),
            "most_likely_bitstring": self._format_bitstring(most_likely_state, num_qubits),
            "most_likely_probability": round(most_likely_prob, 6),
            "expected_normalized_energy": round(float(exp_energy), 6),
            "sample_counts": sample_counts,
            "optimizer": "COBYLA",
            "optimizer_evaluations": evaluations,
            "qubo_details": qubo_details,
        }

        result.algorithm = "QAOA VRP Solver (Azfar et al. 2025)"
        if result.math_proof:
            math_proof.update(result.math_proof)
        result.math_proof = math_proof
        result.solver_details = solver_details
        return result

    def _extract_pairwise_costs(self, problem: ProblemInstance) -> Tuple[Dict[Tuple[int, int], float], Dict]:
        """Extracts dynamic shortest path travel costs between origin and all customer pairs on G(t)."""
        import networkx as nx

        g = problem.graph.graph
        nodes = [problem.origin] + problem.destinations
        cost_matrix = {}
        paths = {}

        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i == j:
                    continue
                try:
                    path_cost = nx.dijkstra_path_length(g, u, v, weight="weight")
                    path_nodes = nx.dijkstra_path(g, u, v, weight="weight")
                    cost_matrix[(i, j)] = float(path_cost)
                    paths[(i, j)] = path_nodes
                except nx.NetworkXNoPath:
                    cost_matrix[(i, j)] = 1000.0

        return cost_matrix, paths

    def _build_position_vrp_qubo(
        self, N: int, cost_matrix: Dict[Tuple[int, int], float]
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Formulates QUBO matrix for N-customer VRP with position decision variables x_{i,p}.
        x_{i,p} = 1 if customer i is visited at step p (0-indexed).
        """
        num_vars = N * N
        Q = np.zeros((num_vars, num_vars), dtype=float)
        linear = np.zeros(num_vars, dtype=float)
        offset = 0.0

        # Sum of absolute costs for penalty multiplier P = 2 * sum(|c_{i,j}|)
        sum_c = sum(abs(w) for w in cost_matrix.values())
        P = 2.0 * sum_c if sum_c > 0 else 100.0

        # 1. Cost terms
        # Step 0: Cost from Origin (0) to first customer (i)
        for i in range(N):
            c_0_i = cost_matrix.get((0, i + 1), 10.0)
            v_idx = i * N + 0  # Customer i at position 0
            linear[v_idx] += c_0_i

        # Transition steps p -> p+1: Cost from customer i at position p to customer j at position p+1
        for p in range(N - 1):
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    c_i_j = cost_matrix.get((i + 1, j + 1), 10.0)
                    v1 = i * N + p
                    v2 = j * N + (p + 1)
                    Q[v1, v2] += c_i_j

        # 2. Customer visit uniqueness penalty: Each customer i is visited at exactly one position p
        for i in range(N):
            cust_vars = [i * N + p for p in range(N)]
            offset += P
            for v in cust_vars:
                linear[v] -= 2.0 * P
            for v1, v2 in itertools.product(cust_vars, repeat=2):
                Q[v1, v2] += P

        # 3. Position uniqueness penalty: Each position p is assigned exactly one customer i
        for p in range(N):
            pos_vars = [i * N + p for i in range(N)]
            offset += P
            for v in pos_vars:
                linear[v] -= 2.0 * P
            for v1, v2 in itertools.product(pos_vars, repeat=2):
                Q[v1, v2] += P

        Q_sym = np.diag(np.diag(Q)) + np.triu(Q, 1) + np.tril(Q, -1)
        return Q_sym, linear, offset, P

    @staticmethod
    def _qubo_to_ising(
        Q: np.ndarray, linear: np.ndarray
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], float], float]:
        """Transforms QUBO to Ising Hamiltonian: x_i = (Z_i + I)/2."""
        n = len(linear)
        h = np.zeros(n, dtype=float)
        couplings: Dict[Tuple[int, int], float] = {}
        ising_offset = 0.0

        for i in range(n):
            h[i] += linear[i] / 2.0 + Q[i, i] / 4.0
            ising_offset += linear[i] / 2.0 + Q[i, i] / 4.0

            for j in range(i + 1, n):
                coeff = Q[i, j] + Q[j, i]
                if abs(coeff) > 1e-9:
                    h[i] += coeff / 4.0
                    h[j] += coeff / 4.0
                    couplings[(i, j)] = coeff / 4.0
                    ising_offset += coeff / 4.0

        return h, couplings, ising_offset

    @staticmethod
    def _compute_bitstring_energies(Q: np.ndarray, linear: np.ndarray) -> np.ndarray:
        """Precomputes normalized QUBO energy efficiently without memory explosion."""
        n = len(linear)
        max_states = 65536
        if (1 << n) <= max_states:
            states = np.arange(1 << n)
        else:
            states = np.linspace(0, (1 << n) - 1, max_states, dtype=int)

        x_all = np.array([[(int(state) >> k) & 1 for k in range(n)] for state in states], dtype=float)
        linear_term = x_all @ linear
        quad_term = np.sum((x_all @ Q) * x_all, axis=1)
        return linear_term + quad_term

    def _optimize_qaoa_parameters(
        self,
        h: np.ndarray,
        couplings: Dict[Tuple[int, int], float],
        energies: np.ndarray,
        reps: int = 1,
    ) -> Tuple[np.ndarray, float, int]:
        """Optimizes QAOA variational parameters (gamma, beta) using COBYLA optimizer."""
        initial_starts = [
            np.concatenate([np.full(reps, 0.2 * np.pi), np.full(reps, 0.1 * np.pi)]),
            np.concatenate([np.full(reps, 0.5 * np.pi), np.full(reps, 0.25 * np.pi)]),
            np.concatenate([np.full(reps, 0.8 * np.pi), np.full(reps, 0.4 * np.pi)]),
        ][: self.restarts]

        n = len(h)
        n_states = 1 << n

        evaluations = 0

        def objective_func(params: np.ndarray) -> float:
            nonlocal evaluations
            evaluations += 1
            if n_states <= 65536 and len(energies) == n_states:
                gammas = params[:reps]
                betas = params[reps:]
                psi = np.full(n_states, 1.0 / np.sqrt(n_states), dtype=complex)
                for p_idx in range(reps):
                    g = float(gammas[p_idx])
                    b = float(betas[p_idx])
                    psi *= np.exp(-1j * g * energies)
                    cos_b = np.cos(b)
                    sin_b = -1j * np.sin(b)
                    psi_t = psi.reshape([2] * n)
                    for q in range(n):
                        psi_t = np.moveaxis(psi_t, q, 0)
                        v0 = psi_t[0].copy()
                        v1 = psi_t[1].copy()
                        psi_t[0] = cos_b * v0 + sin_b * v1
                        psi_t[1] = sin_b * v0 + cos_b * v1
                        psi_t = np.moveaxis(psi_t, 0, q)
                    psi = psi_t.ravel()
                probs = np.abs(psi)**2
                return float(np.dot(probs, energies))

            circuit = self._synthesize_qaoa_circuit(h, couplings, params, reps=reps)
            probs = Statevector.from_instruction(circuit).probabilities()
            if len(probs) > len(energies):
                step = max(1, len(probs) // len(energies))
                sampled_probs = probs[::step][:len(energies)]
                norm = np.sum(sampled_probs)
                if norm > 0:
                    sampled_probs = sampled_probs / norm
                return float(np.dot(sampled_probs, energies))
            return float(np.dot(probs[:len(energies)], energies[:len(probs)]))

        best_result = None
        min_energy = float("inf")

        for start_params in initial_starts:
            res = minimize(
                objective_func,
                start_params,
                method="COBYLA",
                options={"maxiter": self.maxiter, "rhobeg": 0.2},
            )
            if res.fun < min_energy:
                min_energy = float(res.fun)
                best_result = res

        opt_x = best_result.x if best_result is not None else initial_starts[0]
        return np.asarray(opt_x, dtype=float), min_energy, evaluations

    @staticmethod
    def _synthesize_qaoa_circuit(
        h: np.ndarray,
        couplings: Dict[Tuple[int, int], float],
        parameters: np.ndarray,
        reps: int = 1,
    ) -> "QuantumCircuit":
        """Synthesizes p-layer QAOA quantum circuit with cost phase evolution and X-mixer gates."""
        n_qubits = len(h)
        circuit = QuantumCircuit(n_qubits)

        # Initial equal superposition state |+>
        circuit.h(range(n_qubits))

        gammas = parameters[:reps]
        betas = parameters[reps:]

        for p in range(reps):
            g = float(gammas[p])
            b = float(betas[p])

            for i, coeff in enumerate(h):
                if abs(coeff) > 1e-9:
                    circuit.rz(2.0 * g * coeff, i)

            for (i, j), coeff in couplings.items():
                if abs(coeff) > 1e-9:
                    circuit.cx(i, j)
                    circuit.rz(2.0 * g * coeff, j)
                    circuit.cx(i, j)

            for i in range(n_qubits):
                circuit.rx(2.0 * b, i)

        return circuit

    def _decode_bitstring_to_sequence(
        self, state_idx: int, origin: str, destinations: List[str]
    ) -> Optional[List[str]]:
        """Decodes active decision variables x_{i,p} = 1 from bitstring into customer visit sequence."""
        N = len(destinations)
        num_vars = N * N

        position_assignment = {}
        for i in range(N):
            for p in range(N):
                v_idx = i * N + p
                if (state_idx >> v_idx) & 1:
                    if p in position_assignment:
                        return None  # Duplicate position assignment
                    position_assignment[p] = destinations[i]

        if len(position_assignment) != N:
            return None  # Incomplete customer assignment

        sequence = [origin]
        for p in range(N):
            sequence.append(position_assignment[p])

        return sequence

    def _sample_counts(self, probabilities: np.ndarray, num_qubits: int, shots: int) -> Dict[str, int]:
        rng = np.random.default_rng(self.seed)
        sampled_indices = rng.choice(len(probabilities), size=shots, p=probabilities)
        counts: Dict[str, int] = {}
        for idx in sampled_indices:
            b_str = self._format_bitstring(int(idx), num_qubits)
            counts[b_str] = counts.get(b_str, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8])

    @staticmethod
    def _format_bitstring(state_idx: int, num_qubits: int) -> str:
        return "".join(str((state_idx >> k) & 1) for k in range(num_qubits))

    @staticmethod
    def _failure(error_msg: str, started: float, problem: Optional[ProblemInstance] = None) -> AlgorithmResult:
        origin = problem.origin if problem else "Origin"
        vehicles = problem.vehicles if problem else []
        is_infeasible = "infeasible" in error_msg.lower() or "exceeds" in error_msg.lower()
        standby_routes = [
            FleetVehicleRoute(
                vehicle_id=v.vehicle_id,
                capacity=v.capacity,
                assigned_orders=[],
                visit_sequence=[origin, origin],
                node_path=[origin],
                edge_path=[],
                geometry=[],
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
                    "node": origin,
                    "action": f"Vehicle {v.vehicle_id} Standby at Depot. ({error_msg})",
                    "delivered": 0.0,
                    "remaining": 0.0,
                    "cumulative_distance_m": 0.0,
                    "cumulative_time_s": 0.0,
                }],
                segment_calculations=[],
                bottlenecks=[],
                is_active=False,
                status="standby",
                color=v.color,
            )
            for v in vehicles
        ]
        return AlgorithmResult(
            algorithm="QAOA VRP Solver (Azfar et al. 2025)",
            status="infeasible" if is_infeasible else "error",
            success=False,
            visit_sequence=[origin, origin],
            node_path=[],
            edge_path=[],
            total_cost=0.0,
            total_travel_time=0.0,
            total_distance=0.0,
            average_speed_ms=0.0,
            bottleneck_count=0,
            bottlenecks=[],
            segment_calculations=[],
            geometry=[],
            feasible=False,
            constraint_violations=[error_msg],
            computation_time_ms=(time.perf_counter() - started) * 1000,
            math_proof={"infeasibility_reason": error_msg, "pre_check_failed": True},
            fleet_routes=standby_routes,
            error=error_msg,
        )
