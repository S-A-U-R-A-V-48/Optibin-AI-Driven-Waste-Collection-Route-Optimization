# 🗑️ AI-Driven Waste Collection & Route Optimization

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![ML](https://img.shields.io/badge/Machine%20Learning-GBM%20%7C%20RF%20%7C%20RL-orange?logo=scikit-learn)
![Optimization](https://img.shields.io/badge/Optimization-CVRP%20%7C%202--opt-green)
![IoT](https://img.shields.io/badge/IoT-Sensor%20Simulation-teal)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**End-to-end AI system for smart city waste collection — from IoT sensor simulation to optimized truck routes and KPI dashboards.**

*Academic Prototype | BIT Mesra, Ranchi | Civil Engineering — Sem VIII*

</div>

---

##  Project Overview

Traditional waste collection operates on **fixed schedules**, ignoring real-time bin fill levels. This results in:
- 30–50% unnecessary trips and fuel consumption
- Overflowing bins causing public health hazards
-  $200B+ global annual waste management cost

This project implements an **AI-driven pipeline** that:
1. **Simulates** 50 IoT-enabled smart bins across 4 city zones
2. **Predicts** fill levels 24 hours ahead using ML (Gradient Boosting / LSTM proxy)
3. **Optimizes** truck routes using Capacitated Vehicle Routing Problem (CVRP)
4. **Learns** dynamic priority routing via Reinforcement Learning (Q-Learning)
5. **Quantifies** impact: cost, CO₂ savings, and ROI

---

##  Key Results

| KPI | Traditional | AI-Optimized | Δ |
|-----|-------------|--------------|---|
| Collection Trips | 100% | 65–77% | **↓ 23–35%** |
| Fuel Consumption | 100% | 72% | **↓ 28%** |
| CO₂ Emissions | 100% | 70% | **↓ 30%** |
| On-Time Rate | 72% | 97% | **↑ +25pp** |
| Overflow Incidents | 100% | 15% | **↓ 85%** |
| Annual Savings | — | $357K/yr | **3.2× ROI** |

---

##  Architecture

```
IoT Sensors (50 bins, 15-min intervals)
        │
        ▼
┌─────────────────────────────────────────────┐
│            Data Pipeline                    │
│   Simulation → Feature Engineering → Store  │
└─────────────────────────────────────────────┘
        │                    │
        ▼                    ▼
┌─────────────┐    ┌──────────────────────┐
│  ML Engine  │    │   OR Optimizer       │
│  GBM / RF   │    │   CVRP + 2-opt       │
│  Q-Learning │    │   Time Windows       │
└─────────────┘    └──────────────────────┘
        │                    │
        └─────────┬──────────┘
                  ▼
        ┌─────────────────┐
        │  KPI Dashboard  │
        │  ROI · CO₂ · SLA│
        └─────────────────┘
```

---

##  Project Structure

```
ai-waste-optimizer/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 setup.py
├── 📄 config.yaml               ← All parameters in one place
├──  main.py                   ← CLI entry point (run everything)
│
├──  src/
│   ├── simulation/
│   │   └── iot_simulator.py     ← 50-bin city grid, 30-day sensor data
│   ├── eda/
│   │   └── analyzer.py          ← EDA plots, overflow risk, heatmaps
│   ├── models/
│   │   ├── fill_predictor.py    ← GBM fill-level prediction + alerts
│   │   └── rl_agent.py          ← Q-Learning dynamic routing agent
│   ├── optimization/
│   │   └── cvrp_solver.py       ← CVRP nearest-neighbour + 2-opt
│   ├── dashboard/
│   │   └── kpi_report.py        ← KPI comparison + financial model
│   └── utils/
│       └── helpers.py           ← Shared utilities
│
├──  notebooks/                ← Step-by-step Jupyter analysis
│   ├── 01_iot_data_simulation.ipynb
│   ├── 02_eda_waste_patterns.ipynb
│   ├── 03_ml_lstm_fill_prediction.ipynb
│   ├── 04_cvrp_route_optimization.ipynb
│   ├── 05_random_forest_rl.ipynb
│   └── 06_impact_dashboard.ipynb
│
├── data/sample/              ← Sample CSV (full data generated on run)
├──  outputs/                  ← Generated figures saved here
├──  tests/                    ← Unit tests
└── docs/                     ← Architecture and API docs
```

---

##  Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/ai-waste-optimizer.git
cd ai-waste-optimizer

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
# Run entire pipeline end-to-end
python main.py --run all

# Run individual stages
python main.py --run simulate     # Generate IoT sensor data
python main.py --run eda          # Exploratory data analysis
python main.py --run predict      # Train & evaluate ML model
python main.py --run optimize     # Solve CVRP route optimization
python main.py --run rl           # Train RL routing agent
python main.py --run dashboard    # Generate KPI dashboard
```

### 3. Explore Notebooks

```bash
jupyter notebook notebooks/
```

Open notebooks in order (01 → 06) for step-by-step walkthrough.

---

## 📓 Notebook Guide

| # | Notebook | What it does |
|---|----------|-------------|
| 01 | `01_iot_data_simulation.ipynb` | Creates city grid, simulates 144,000 sensor readings over 30 days |
| 02 | `02_eda_waste_patterns.ipynb` | Overflow risk, temporal heatmaps, zone distributions |
| 03 | `03_ml_lstm_fill_prediction.ipynb` | Time-series feature engineering, GBM model, 24h fill forecast |
| 04 | `04_cvrp_route_optimization.ipynb` | Distance matrix, nearest-neighbour CVRP, 2-opt improvement |
| 05 | `05_random_forest_rl.ipynb` | Random Forest demand forecast + Q-Learning routing agent |
| 06 | `06_impact_dashboard.ipynb` | Full KPI dashboard, ROI model, CO₂ savings visualization |

---

## ⚙️ Configuration

All system parameters are in `config.yaml`:

```yaml
simulation:
  n_bins: 50
  n_days: 30
  sensor_interval_minutes: 15

fleet:
  n_trucks: 5
  truck_capacity_liters: 1000
  bin_capacity_liters: 240

thresholds:
  high_priority: 0.80      # ≥80% fill → HIGH priority
  collection_trigger: 0.60  # ≥60% fill → schedule collection

ml:
  test_size: 0.20
  n_estimators: 100
  forecast_horizon_hours: 24

rl:
  episodes: 600
  learning_rate: 0.1
  discount_factor: 0.95
  epsilon: 1.0
  epsilon_decay: 0.995
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| Data | NumPy, Pandas |
| ML | Scikit-learn (GBM, Random Forest) |
| Optimization | Custom CVRP (Nearest-Neighbour + 2-opt) |
| RL | Q-Learning (from scratch) |
| Visualization | Matplotlib, Seaborn |
| Notebooks | Jupyter |
| Testing | Pytest |

---

##  Running Tests

```bash
pytest tests/ -v
```

---

##  Sample Outputs

All figures are saved to `outputs/` after running the pipeline:

- `01_iot_visualization.png` — Fill time-series, bin map, zone distributions
- `02_eda_analysis.png` — Overflow risk, day-hour heatmap
- `03_lstm_prediction.png` — Actual vs predicted fill levels
- `04_route_optimization.png` — Baseline vs optimized truck routes
- `05_rf_rl.png` — Random Forest accuracy + RL reward convergence
- `06_kpi_dashboard.png` — Full KPI comparison dashboard

---

##  Real-World Validation

| City | Population | Result |
|------|-----------|--------|
| Seoul, South Korea | 9.7M | 30% route reduction, $8.5M/yr savings |
| Amsterdam, Netherlands | 900K | 32% CO₂ reduction, €3.2M/yr savings |
| Songdo Smart City | Full city | 90% cost reduction with AI + pneumatic |

---

## References

1. Dantzig & Ramser (1959). *The Truck Dispatching Problem*. Management Science.
2. Clarke & Wright (1964). *Scheduling of Vehicles from a Central Depot*. Operations Research.
3. Du & Wibowo (2022). *Forecasting Waste Status Using Deep Learning*. IJERPH.
4. Sutton & Barto (2018). *Reinforcement Learning: An Introduction*. MIT Press.
5. Hochreiter & Schmidhuber (1997). *Long Short-Term Memory*. Neural Computation.

---



---

##  License

MIT License — feel free to use, adapt, and cite this work.
