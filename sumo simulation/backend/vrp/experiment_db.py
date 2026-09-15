from __future__ import annotations

import os
import json
import sqlite3
import numpy as np
from typing import Any, Dict, List, Optional


class ExperimentDatabase:
    """
    SQLite persistence database for benchmark experiment storage and multi-seed statistical analysis.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(backend_dir, "experiments.db")
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    timestamp REAL,
                    graph_time REAL,
                    depot TEXT,
                    num_customers INTEGER,
                    num_vehicles INTEGER,
                    algorithm TEXT,
                    seed INTEGER,
                    solution_json TEXT,
                    total_cost REAL,
                    total_travel_time REAL,
                    total_distance REAL,
                    runtime_ms REAL,
                    feasible INTEGER,
                    convergence_json TEXT,
                    parameters_json TEXT,
                    problem_json TEXT,
                    is_optimal INTEGER DEFAULT 0,
                    optimality_gap REAL
                )
            """)
            conn.commit()

            # Ensure columns exist if table was created in an older schema
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(experiments)")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            needed_cols = {
                "convergence_json": "TEXT",
                "parameters_json": "TEXT",
                "problem_json": "TEXT",
                "is_optimal": "INTEGER DEFAULT 0",
                "optimality_gap": "REAL",
            }
            for col_name, col_type in needed_cols.items():
                if col_name not in existing_cols:
                    try:
                        conn.execute(f"ALTER TABLE experiments ADD COLUMN {col_name} {col_type}")
                    except Exception:
                        pass
            conn.commit()

    def save_experiment(
        self,
        experiment_id: str,
        timestamp: float,
        graph_time: float,
        depot: str,
        num_customers: int,
        num_vehicles: int,
        algorithm: str,
        seed: int,
        solution_dict: Dict[str, Any],
        total_cost: float,
        total_travel_time: float,
        total_distance: float,
        runtime_ms: float,
        feasible: bool,
        convergence_history: Optional[List[float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        problem_dict: Optional[Dict[str, Any]] = None,
        is_optimal: bool = False,
        optimality_gap: Optional[float] = None,
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (
                    experiment_id, timestamp, graph_time, depot, num_customers, num_vehicles,
                    algorithm, seed, solution_json, total_cost, total_travel_time,
                    total_distance, runtime_ms, feasible, convergence_json, parameters_json,
                    problem_json, is_optimal, optimality_gap
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id, timestamp, graph_time, depot, num_customers, num_vehicles,
                algorithm, seed, json.dumps(solution_dict), total_cost, total_travel_time,
                total_distance, runtime_ms, 1 if feasible else 0,
                json.dumps(convergence_history or []),
                json.dumps(parameters or {}),
                json.dumps(problem_dict or {}),
                1 if is_optimal else 0,
                optimality_gap,
            ))
            conn.commit()
            return cursor.lastrowid or 0

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            results = []
            for row in rows:
                item = dict(row)
                item["solution"] = json.loads(item["solution_json"]) if item.get("solution_json") else {}
                item["convergence"] = json.loads(item["convergence_json"]) if item.get("convergence_json") else []
                item["parameters"] = json.loads(item["parameters_json"]) if item.get("parameters_json") else {}
                item["problem"] = json.loads(item["problem_json"]) if item.get("problem_json") else {}
                if "solution_json" in item:
                    del item["solution_json"]
                if "convergence_json" in item:
                    del item["convergence_json"]
                if "parameters_json" in item:
                    del item["parameters_json"]
                if "problem_json" in item:
                    del item["problem_json"]
                results.append(item)
            return results

    def get_experiment_details(self, experiment_id: str) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments WHERE experiment_id = ? ORDER BY id ASC", (experiment_id,))
            rows = cursor.fetchall()
            if not rows:
                return {}
            
            runs = []
            for row in rows:
                item = dict(row)
                item["solution"] = json.loads(item["solution_json"]) if item.get("solution_json") else {}
                item["convergence"] = json.loads(item["convergence_json"]) if item.get("convergence_json") else []
                item["parameters"] = json.loads(item["parameters_json"]) if item.get("parameters_json") else {}
                item["problem"] = json.loads(item["problem_json"]) if item.get("problem_json") else {}
                del item["solution_json"]
                del item["convergence_json"]
                del item["parameters_json"]
                del item["problem_json"]
                runs.append(item)
            
            stats = self.compute_multi_seed_stats(experiment_id)
            pairwise = self.compute_pairwise_comparisons(experiment_id)
            return {
                "experiment_id": experiment_id,
                "runs": runs,
                "stats": stats,
                "pairwise_comparisons": pairwise,
            }

    def compute_wilcoxon_test(
        self,
        experiment_id: str,
        algo_a: str,
        algo_b: str,
        metric: str = "total_cost",
    ) -> Dict[str, Any]:
        """
        Compute paired Wilcoxon signed-rank test between two algorithms on matched seeds for an experiment.
        """
        import scipy.stats as stats

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT seed, algorithm, total_cost, runtime_ms, feasible
                FROM experiments
                WHERE experiment_id = ? AND algorithm IN (?, ?)
                ORDER BY seed ASC
            """, (experiment_id, algo_a, algo_b))
            rows = cursor.fetchall()

        by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for row in rows:
            s = row["seed"]
            a = row["algorithm"]
            by_seed.setdefault(s, {})[a] = dict(row)

        paired_a: List[float] = []
        paired_b: List[float] = []
        paired_runtimes_a: List[float] = []
        paired_runtimes_b: List[float] = []
        matched_seeds: List[int] = []

        for s, alg_map in sorted(by_seed.items()):
            if algo_a in alg_map and algo_b in alg_map:
                ra = alg_map[algo_a]
                rb = alg_map[algo_b]
                if ra["feasible"] and rb["feasible"] and ra["total_cost"] > 0 and rb["total_cost"] > 0:
                    matched_seeds.append(s)
                    paired_a.append(ra["total_cost"])
                    paired_b.append(rb["total_cost"])
                    paired_runtimes_a.append(ra["runtime_ms"])
                    paired_runtimes_b.append(rb["runtime_ms"])

        n = len(matched_seeds)
        if n == 0:
            return {
                "algo_a": algo_a,
                "algo_b": algo_b,
                "metric": metric,
                "n_pairs": 0,
                "status": "insufficient_data",
                "message": f"No matched feasible seeds between {algo_a} and {algo_b}",
                "p_value": None,
                "statistic": None,
                "is_significant": False,
                "conclusion": "Insufficient matched feasible runs to conduct statistical testing.",
            }

        arr_a = np.array(paired_a)
        arr_b = np.array(paired_b)
        mean_a = float(np.mean(arr_a))
        mean_b = float(np.mean(arr_b))
        gap_pct = round(((mean_a - mean_b) / mean_b) * 100.0, 2) if mean_b > 0 else 0.0

        rt_a = float(np.mean(paired_runtimes_a)) if paired_runtimes_a else 0.0
        rt_b = float(np.mean(paired_runtimes_b)) if paired_runtimes_b else 0.0
        time_ratio = round(rt_a / rt_b, 2) if rt_b > 0 else 1.0

        diff = arr_a - arr_b
        if np.all(diff == 0.0):
            statistic = 0.0
            p_value = 1.0
        else:
            try:
                res = stats.wilcoxon(arr_a, arr_b, alternative="two-sided")
                statistic = float(res.statistic)
                p_value = float(res.pvalue)
            except Exception:
                statistic = 0.0
                p_value = 1.0

        is_significant = bool(p_value < 0.05)
        if is_significant:
            if mean_a < mean_b:
                winner = algo_a
                conclusion = (
                    f"{algo_a} achieves statistically significantly lower objective cost than {algo_b} "
                    f"(p = {p_value:.4f} < 0.05, Wilcoxon W = {statistic:.1f}, mean cost gap: {gap_pct:+.2f}%)."
                )
            else:
                winner = algo_b
                conclusion = (
                    f"{algo_b} achieves statistically significantly lower objective cost than {algo_a} "
                    f"(p = {p_value:.4f} < 0.05, Wilcoxon W = {statistic:.1f}, mean cost gap: {gap_pct:+.2f}%)."
                )
        else:
            winner = "tie"
            conclusion = (
                f"No statistically significant difference detected between {algo_a} and {algo_b} "
                f"under the selected benchmark methodology (p = {p_value:.4f} >= 0.05, alpha = 0.05)."
            )

        return {
            "algo_a": algo_a,
            "algo_b": algo_b,
            "metric": metric,
            "n_pairs": n,
            "matched_seeds": matched_seeds,
            "mean_a": round(mean_a, 2),
            "mean_b": round(mean_b, 2),
            "gap_pct": gap_pct,
            "mean_runtime_ms_a": round(rt_a, 2),
            "mean_runtime_ms_b": round(rt_b, 2),
            "time_ratio": time_ratio,
            "statistic": round(statistic, 2),
            "p_value": round(p_value, 4),
            "alpha": 0.05,
            "is_significant": is_significant,
            "winner": winner,
            "conclusion": conclusion,
            "sample_size_note": (
                f"Paired sample size n = {n}."
                + (" Warning: n < 5 provides limited power for non-parametric rank tests." if n < 5 else "")
            ),
        }

    def compute_pairwise_comparisons(self, experiment_id: str) -> List[Dict[str, Any]]:
        """
        Compute paired Wilcoxon tests across all algorithm pairs in the experiment.
        Standard pairs prioritized: (QPSO, ALNS), (QPSO, HGS), (ALNS, HGS), (QPSO, QAOA).
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT algorithm FROM experiments WHERE experiment_id = ?", (experiment_id,))
            algos = [r["algorithm"] for r in cursor.fetchall()]

        comparisons = []
        for i in range(len(algos)):
            for j in range(i + 1, len(algos)):
                comp = self.compute_wilcoxon_test(experiment_id, algos[i], algos[j])
                comparisons.append(comp)
        return comparisons

    def compute_multi_seed_stats(self, experiment_id: str) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT algorithm, seed, total_cost, total_travel_time, total_distance, runtime_ms, feasible, is_optimal
                FROM experiments
                WHERE experiment_id = ?
            """, (experiment_id,))
            rows = cursor.fetchall()

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            algo = row["algorithm"]
            grouped.setdefault(algo, []).append(dict(row))

        # First find exact optimal or best overall cost for gap calculations
        optimal_cost: Optional[float] = None
        best_known_cost: float = float("inf")

        for items in grouped.values():
            for it in items:
                if it["feasible"] and it["total_cost"] > 0:
                    if it["total_cost"] < best_known_cost:
                        best_known_cost = it["total_cost"]
                    if it.get("is_optimal"):
                        optimal_cost = it["total_cost"]

        reference_cost = optimal_cost if optimal_cost is not None else (best_known_cost if best_known_cost < float("inf") else None)

        stats: Dict[str, Any] = {}
        for algo, items in grouped.items():
            feasible_items = [it for it in items if it["feasible"] and it["total_cost"] > 0]
            costs = [it["total_cost"] for it in feasible_items]
            times = [it["total_travel_time"] for it in feasible_items]
            dists = [it["total_distance"] for it in feasible_items]
            runtimes = [it["runtime_ms"] for it in items]

            total_runs = len(items)
            feasible_runs = len(feasible_items)
            feasibility_rate = round((feasible_runs / total_runs * 100.0), 1) if total_runs > 0 else 0.0

            if costs:
                mean_cost = float(np.mean(costs))
                opt_gap = None
                if reference_cost and reference_cost > 0:
                    opt_gap = round(((mean_cost - reference_cost) / reference_cost) * 100.0, 2)

                stats[algo] = {
                    "runs_count": total_runs,
                    "feasible_runs": feasible_runs,
                    "feasibility_rate_pct": feasibility_rate,
                    "optimality_gap_pct": opt_gap,
                    "cost": {
                        "best": round(float(np.min(costs)), 2),
                        "worst": round(float(np.max(costs)), 2),
                        "mean": round(mean_cost, 2),
                        "median": round(float(np.median(costs)), 2),
                        "std": round(float(np.std(costs)), 2),
                        "variance": round(float(np.var(costs)), 2),
                    },
                    "travel_time": {
                        "best": round(float(np.min(times)), 2),
                        "worst": round(float(np.max(times)), 2),
                        "mean": round(float(np.mean(times)), 2),
                        "median": round(float(np.median(times)), 2),
                        "std": round(float(np.std(times)), 2),
                        "variance": round(float(np.var(times)), 2),
                    },
                    "distance": {
                        "best": round(float(np.min(dists)), 2),
                        "worst": round(float(np.max(dists)), 2),
                        "mean": round(float(np.mean(dists)), 2),
                        "median": round(float(np.median(dists)), 2),
                        "std": round(float(np.std(dists)), 2),
                        "variance": round(float(np.var(dists)), 2),
                    },
                    "runtime_ms": {
                        "best": round(float(np.min(runtimes)), 2),
                        "worst": round(float(np.max(runtimes)), 2),
                        "mean": round(float(np.mean(runtimes)), 2),
                        "median": round(float(np.median(runtimes)), 2),
                        "std": round(float(np.std(runtimes)), 2),
                    }
                }
            else:
                stats[algo] = {
                    "runs_count": total_runs,
                    "feasible_runs": 0,
                    "feasibility_rate_pct": 0.0,
                    "optimality_gap_pct": None,
                    "error": "No feasible solutions found across seeds"
                }

        return stats
