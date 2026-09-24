#!/usr/bin/env python3
"""
main.py
=======
CLI entry point for the AI Waste Optimizer pipeline.

Usage:
    python main.py --run all          # Full pipeline
    python main.py --run simulate     # Generate IoT data
    python main.py --run eda          # Exploratory analysis
    python main.py --run predict      # Train ML model
    python main.py --run optimize     # Solve CVRP
    python main.py --run rl           # Train RL agent
    python main.py --run dashboard    # Generate KPI dashboard
    python main.py --help
"""
import argparse
import sys
import os
import time
import pandas as pd

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(__file__))

from src.utils.helpers import load_config, get_logger, ensure_dir

logger = get_logger("main")


def run_simulate(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.simulation.iot_simulator import IoTSimulator
    sim = IoTSimulator(cfg)
    bins_df, readings_df = sim.run()
    sim.plot(bins_df, readings_df)
    return bins_df, readings_df


def run_eda(cfg, bins_df, readings_df) -> dict:
    from src.eda.analyzer import EDAAnalyzer
    eda = EDAAnalyzer(cfg)
    return eda.run(bins_df, readings_df)


def run_predict(cfg, readings_df) -> tuple:
    from src.models.fill_predictor import FillPredictor
    predictor = FillPredictor(cfg)
    model, metrics = predictor.run(readings_df)
    return model, metrics


def run_optimize(cfg, bins_df, readings_df) -> tuple[list, dict]:
    from src.optimization.cvrp_solver import CVRPSolver
    solver = CVRPSolver(cfg)
    routes, stats = solver.run(bins_df, readings_df)
    return routes, stats


def run_rl(cfg, bins_df, readings_df) -> dict:
    from src.models.rl_agent import QLearningAgent, RandomForestDemandForecaster
    # Random Forest
    rf = RandomForestDemandForecaster(cfg)
    rf_metrics = rf.run(readings_df)
    logger.info(f"RF — MAE: {rf_metrics['mae']:.4f} | R²: {rf_metrics['r2']:.4f}")

    # Q-Learning RL
    agent = QLearningAgent(cfg)
    rl_history = agent.train(bins_df, readings_df)
    return rl_history


def run_dashboard(cfg, cvrp_stats=None, ml_metrics=None, rl_history=None) -> dict:
    from src.dashboard.kpi_report import KPIDashboard
    dash = KPIDashboard(cfg)
    return dash.run(cvrp_stats, ml_metrics, rl_history)


def load_data(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-existing data, or regenerate if missing."""
    data_dir = cfg["paths"]["data_dir"]
    bins_path = f"{data_dir}/bins_metadata.csv"
    readings_path = f"{data_dir}/sensor_readings.csv"

    if os.path.exists(bins_path) and os.path.exists(readings_path):
        logger.info("Loading existing data from disk...")
        bins_df = pd.read_csv(bins_path)
        readings_df = pd.read_csv(readings_path)
        return bins_df, readings_df
    else:
        logger.info("No data found — running simulation first...")
        return run_simulate(cfg)


def main():
    parser = argparse.ArgumentParser(
        description="AI-Driven Waste Collection & Route Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  simulate    Generate IoT sensor data for 50 bins × 30 days
  eda         Exploratory data analysis and visualisation
  predict     Train GBM fill-level prediction model
  optimize    Solve CVRP route optimization (NN + 2-opt)
  rl          Train Q-Learning dynamic routing agent
  dashboard   Generate KPI comparison dashboard
  all         Run entire pipeline end-to-end

Examples:
  python main.py --run all
  python main.py --run simulate --config config.yaml
  python main.py --run optimize
        """,
    )
    parser.add_argument(
        "--run",
        choices=["all", "simulate", "eda", "predict", "optimize", "rl", "dashboard"],
        default="all",
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    # ── Load configuration ────────────────────────────────────────────────────
    cfg = load_config(args.config)
    ensure_dir(cfg["paths"]["outputs_dir"])
    ensure_dir(cfg["paths"]["data_dir"])

    print("\n" + "=" * 65)
    print("  🗑️  AI-Driven Waste Collection & Route Optimization")
    print("  BIT Mesra | Civil Engineering | Sanchit Sanyam")
    print("=" * 65)

    start = time.time()
    cvrp_stats = ml_metrics = rl_history = bins_df = readings_df = None

    stage = args.run

    # ── Execute requested stage(s) ────────────────────────────────────────────
    if stage in ("all", "simulate"):
        print("\n[1/6] 📡 IoT Data Simulation...")
        bins_df, readings_df = run_simulate(cfg)
        print(f"      ✓ {len(readings_df):,} sensor readings generated")

    if stage in ("all", "eda", "predict", "optimize", "rl", "dashboard"):
        if bins_df is None:
            bins_df, readings_df = load_data(cfg)

    if stage in ("all", "eda"):
        print("\n[2/6] 🔍 Exploratory Data Analysis...")
        eda_results = run_eda(cfg, bins_df, readings_df)
        overflow_bins = len(eda_results["overflow_stats"][
            eda_results["overflow_stats"]["overflow_rate"] > 0.10])
        print(f"      ✓ {overflow_bins} bins with >10% overflow risk identified")

    if stage in ("all", "predict"):
        print("\n[3/6] 🤖 ML Fill-Level Prediction (GBM)...")
        _, ml_metrics = run_predict(cfg, readings_df)
        print(f"      ✓ MAE: {ml_metrics['mae']:.4f} | R²: {ml_metrics['r2']:.4f}")

    if stage in ("all", "optimize"):
        print("\n[4/6] 🗺️  CVRP Route Optimization...")
        _, cvrp_stats = run_optimize(cfg, bins_df, readings_df)
        print(f"      ✓ {cvrp_stats['improvement_pct']:.1f}% route improvement | "
              f"CO2 saved: {cvrp_stats['co2_saved_kg']:.2f} kg")

    if stage in ("all", "rl"):
        print("\n[5/6] 🎮 Reinforcement Learning Agent...")
        rl_history = run_rl(cfg, bins_df, readings_df)
        final_reward = sum(rl_history["episode_rewards"][-50:]) / 50
        print(f"      ✓ Q-Learning converged | Avg reward (last 50 ep): {final_reward:.3f}")

    if stage in ("all", "dashboard"):
        print("\n[6/6] 📊 KPI Dashboard...")
        run_dashboard(cfg, cvrp_stats, ml_metrics, rl_history)

    elapsed = time.time() - start
    print(f"\n✅ Pipeline complete in {elapsed:.1f}s")
    print(f"   Outputs saved to: {cfg['paths']['outputs_dir']}/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
