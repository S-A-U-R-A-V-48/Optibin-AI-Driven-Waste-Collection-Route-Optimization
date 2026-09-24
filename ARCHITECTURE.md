# System Architecture

## Overview

The AI Waste Optimizer is a five-layer system that takes raw IoT sensor readings
and converts them into optimized truck routes with quantified environmental impact.

## Layer 1 — IoT Simulation (`src/simulation/`)

- **50 smart bins** across 4 zones: Residential, Commercial, Industrial, Park
- Each bin has a zone-specific fill-rate profile with peak hours and weekend factors
- **Sensor interval**: 15 minutes → 144,000 readings over 30 days
- Outputs: `bins_metadata.csv`, `sensor_readings.csv`

## Layer 2 — Feature Engineering & EDA (`src/eda/`)

- Overflow risk analysis (% of readings above 80% threshold)
- Day-of-week × hour heatmaps
- Zone-level statistical summaries
- Identifies top bins for priority collection

## Layer 3 — Machine Learning (`src/models/`)

### Fill Level Predictor (`fill_predictor.py`)
- **Algorithm**: Gradient Boosting Regressor (sklearn)
- **Features**: Lag values (t-1, t-2, t-4, t-8), rolling mean/std, temporal features, zone
- **Target**: Fill level at next reading interval
- **Evaluated**: MAE, RMSE, R² on held-out 20% test split

### Random Forest Demand Forecaster (`rl_agent.py`)
- **Algorithm**: RandomForestRegressor with 5-fold CV
- **Scope**: Zone-level aggregate demand forecasting

### Q-Learning Agent (`rl_agent.py`)
- **Algorithm**: Tabular Q-Learning (from scratch)
- **State**: Encoded (truck load bucket, bins visited count)
- **Action**: Which priority bin to collect next
- **Reward**: +fill_collected - 0.1 × distance
- **Update rule**: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]
- Trains over 600 episodes with ε-greedy exploration and exponential ε-decay

## Layer 4 — Route Optimization (`src/optimization/`)

### CVRP Solver (`cvrp_solver.py`)
- **Problem**: Capacitated Vehicle Routing Problem (CVRP)
- **Step 1**: Nearest-Neighbour construction heuristic (greedy baseline)
- **Step 2**: 2-opt local search (iterative edge reversal)
- **Constraints**: Truck capacity (1,000 L), depot start/end
- **Metrics**: Total km, CO2 saved, fuel cost saved

## Layer 5 — KPI Dashboard (`src/dashboard/`)

- Traditional vs AI-optimized comparison (trips, fuel, CO2, cost)
- ROI curve with payback period
- Annual CO2 savings by month
- Cost waterfall chart
- On-time collection rate (donut chart)

## Data Flow

```
config.yaml
    │
    ▼
IoTSimulator.run()
    │ bins_metadata.csv
    │ sensor_readings.csv
    │
    ├──► EDAAnalyzer.run()          → 02_eda_analysis.png
    │
    ├──► FillPredictor.run()        → 03_lstm_prediction.png
    │
    ├──► CVRPSolver.run()           → 04_route_optimization.png
    │
    ├──► QLearningAgent.train()     → 05_rf_rl.png
    │
    └──► KPIDashboard.run()         → 06_kpi_dashboard.png
```

## Configuration

All parameters are in `config.yaml`. Key tunables:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_bins` | 50 | Scale of city simulation |
| `n_days` | 30 | Length of sensor history |
| `n_trucks` | 5 | Fleet size for CVRP |
| `high_priority` | 0.80 | Fill threshold for HIGH alert |
| `rl.episodes` | 600 | Q-Learning training length |
| `ml.n_estimators` | 100 | GBM/RF tree count |
