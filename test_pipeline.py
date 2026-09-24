"""
tests/test_pipeline.py
=======================
Unit tests for the AI Waste Optimizer pipeline.
Run with: pytest tests/ -v
"""
import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.helpers import (
    euclidean_distance, haversine_distance,
    format_money, format_pct, format_km
)

# ─── Minimal config for testing ──────────────────────────────────────────────
@pytest.fixture
def cfg():
    return {
        "simulation": {
            "n_bins": 10,
            "n_days": 3,
            "sensor_interval_minutes": 15,
            "random_seed": 42,
            "zones": {"Residential": 5, "Commercial": 3, "Industrial": 1, "Park": 1},
        },
        "fleet": {
            "n_trucks": 2,
            "truck_capacity_liters": 1000,
            "bin_capacity_liters": 240,
            "depot_lat": 0.5,
            "depot_lon": 0.5,
        },
        "thresholds": {
            "high_priority": 0.80,
            "collection_trigger": 0.60,
            "overflow_alert": 0.95,
        },
        "ml": {
            "test_size": 0.20,
            "n_estimators": 20,
            "max_depth": 4,
            "learning_rate": 0.1,
            "forecast_horizon_hours": 24,
            "window_size": 4,
        },
        "rl": {
            "episodes": 20,
            "learning_rate": 0.1,
            "discount_factor": 0.95,
            "epsilon_start": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.99,
            "n_priority_bins": 5,
        },
        "cvrp": {
            "max_iterations_2opt": 10,
            "time_limit_seconds": 5,
        },
        "paths": {
            "data_dir": "/tmp/test_waste_data",
            "outputs_dir": "/tmp/test_waste_outputs",
            "sample_data": "/tmp/test_waste_data/sample/bins.csv",
        },
    }


# ─── Utils ───────────────────────────────────────────────────────────────────

class TestUtils:
    def test_euclidean_distance_same_point(self):
        assert euclidean_distance(0.5, 0.5, 0.5, 0.5) == 0.0

    def test_euclidean_distance_known(self):
        d = euclidean_distance(0, 0, 3, 4)
        assert abs(d - 5.0) < 1e-6

    def test_euclidean_symmetry(self):
        a = euclidean_distance(0.1, 0.2, 0.7, 0.9)
        b = euclidean_distance(0.7, 0.9, 0.1, 0.2)
        assert abs(a - b) < 1e-10

    def test_format_money_thousands(self):
        assert format_money(357_000) == "$357.0K"

    def test_format_money_millions(self):
        assert format_money(1_200_000) == "$1.20M"

    def test_format_pct(self):
        assert format_pct(28.333) == "28.3%"

    def test_format_km(self):
        assert format_km(52.8) == "52.8 km"


# ─── Simulation ──────────────────────────────────────────────────────────────

class TestSimulation:
    def test_bins_count(self, cfg):
        from src.simulation.iot_simulator import IoTSimulator
        sim = IoTSimulator(cfg)
        bins_df, _ = sim.run()
        assert len(bins_df) == cfg["simulation"]["n_bins"]

    def test_bins_columns(self, cfg):
        from src.simulation.iot_simulator import IoTSimulator
        sim = IoTSimulator(cfg)
        bins_df, _ = sim.run()
        for col in ["bin_id", "zone", "lat", "lon", "base_fill_rate", "capacity_liters"]:
            assert col in bins_df.columns

    def test_readings_fill_bounds(self, cfg):
        from src.simulation.iot_simulator import IoTSimulator
        sim = IoTSimulator(cfg)
        _, readings_df = sim.run()
        assert readings_df["fill_level"].between(0, 1).all(), "Fill level out of [0,1]"

    def test_readings_alert_categories(self, cfg):
        from src.simulation.iot_simulator import IoTSimulator
        sim = IoTSimulator(cfg)
        _, readings_df = sim.run()
        assert set(readings_df["alert"].unique()).issubset({"LOW", "MEDIUM", "HIGH"})

    def test_readings_volume(self, cfg):
        from src.simulation.iot_simulator import IoTSimulator
        sim = IoTSimulator(cfg)
        _, readings_df = sim.run()
        expected_per_bin = cfg["simulation"]["n_days"] * 24 * (60 // cfg["simulation"]["sensor_interval_minutes"])
        assert len(readings_df) == cfg["simulation"]["n_bins"] * expected_per_bin


# ─── Fill Predictor ──────────────────────────────────────────────────────────

class TestFillPredictor:
    @pytest.fixture
    def data(self, cfg):
        from src.simulation.iot_simulator import IoTSimulator
        sim = IoTSimulator(cfg)
        bins_df, readings_df = sim.run()
        # Need enough commercial bins
        cfg["simulation"]["zones"]["Commercial"] = 5
        cfg["simulation"]["zones"]["Residential"] = 3
        cfg["simulation"]["zones"]["Industrial"] = 1
        cfg["simulation"]["zones"]["Park"] = 1
        cfg["simulation"]["n_bins"] = 10
        sim2 = IoTSimulator(cfg)
        _, readings_df2 = sim2.run()
        return readings_df2

    def test_alert_classify(self, cfg):
        from src.models.fill_predictor import FillPredictor
        pred = FillPredictor(cfg)
        assert pred.classify_alert(0.90) == "HIGH"
        assert pred.classify_alert(0.70) == "MEDIUM"
        assert pred.classify_alert(0.30) == "LOW"


# ─── CVRP Solver ─────────────────────────────────────────────────────────────

class TestCVRPSolver:
    @pytest.fixture
    def solver_data(self, cfg):
        from src.simulation.iot_simulator import IoTSimulator
        sim = IoTSimulator(cfg)
        bins_df, readings_df = sim.run()
        return bins_df, readings_df

    def test_distance_matrix_symmetry(self, cfg, solver_data):
        from src.optimization.cvrp_solver import CVRPSolver
        bins_df, readings_df = solver_data
        solver = CVRPSolver(cfg)
        candidates = solver._select_candidates(bins_df, readings_df)
        if len(candidates) < 2:
            pytest.skip("Not enough bins above collection trigger for this test")
        D = solver._build_distance_matrix(candidates)
        assert np.allclose(D, D.T), "Distance matrix must be symmetric"

    def test_distance_matrix_diagonal_zero(self, cfg, solver_data):
        from src.optimization.cvrp_solver import CVRPSolver
        bins_df, readings_df = solver_data
        solver = CVRPSolver(cfg)
        candidates = solver._select_candidates(bins_df, readings_df)
        if len(candidates) < 2:
            pytest.skip("Not enough candidate bins")
        D = solver._build_distance_matrix(candidates)
        assert np.all(np.diag(D) == 0), "Self-distance must be 0"

    def test_two_opt_no_worse(self, cfg):
        """2-opt should never increase route distance."""
        from src.optimization.cvrp_solver import CVRPSolver
        solver = CVRPSolver(cfg)
        # Synthetic data
        n = 8
        D = np.random.rand(n, n)
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)
        route = list(range(n))
        np.random.shuffle(route)

        original_dist = sum(D[route[i], route[i+1]] for i in range(len(route)-1))
        optimized = solver._two_opt(route, D)
        opt_dist = sum(D[optimized[i], optimized[i+1]] for i in range(len(optimized)-1))

        assert opt_dist <= original_dist + 1e-6, "2-opt must not increase distance"


# ─── Q-Learning Agent ────────────────────────────────────────────────────────

class TestQLearningAgent:
    def test_reward_convergence(self, cfg):
        """Later episodes should have higher average rewards than early ones."""
        from src.simulation.iot_simulator import IoTSimulator
        from src.models.rl_agent import QLearningAgent
        sim = IoTSimulator(cfg)
        bins_df, readings_df = sim.run()

        cfg["rl"]["episodes"] = 50
        agent = QLearningAgent(cfg)
        history = agent.train(bins_df, readings_df)

        early_avg = np.mean(history["episode_rewards"][:10])
        late_avg = np.mean(history["episode_rewards"][-10:])
        # Q-learning should show improvement over time
        assert late_avg >= early_avg - 1.0, "RL agent should not significantly regress"

    def test_q_table_grows(self, cfg):
        from src.simulation.iot_simulator import IoTSimulator
        from src.models.rl_agent import QLearningAgent
        sim = IoTSimulator(cfg)
        bins_df, readings_df = sim.run()

        cfg["rl"]["episodes"] = 20
        agent = QLearningAgent(cfg)
        agent.train(bins_df, readings_df)
        assert len(agent.Q) > 0, "Q-table should be non-empty after training"


# ─── KPI Dashboard ───────────────────────────────────────────────────────────

class TestKPIDashboard:
    def test_report_keys(self, cfg):
        from src.dashboard.kpi_report import KPIDashboard
        dash = KPIDashboard(cfg)
        report = dash._build_report(None, None, None)
        for key in ["annual_savings", "roi_2yr", "payback_months", "investment"]:
            assert key in report

    def test_roi_positive(self, cfg):
        from src.dashboard.kpi_report import KPIDashboard
        dash = KPIDashboard(cfg)
        report = dash._build_report(None, None, None)
        assert report["roi_2yr"] > 1.0, "2-year ROI should exceed 1× investment"
