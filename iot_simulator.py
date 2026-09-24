"""
simulation/iot_simulator.py
============================
Simulates a city-wide network of smart waste bins with realistic
fill-level dynamics, zone-specific behaviour, and temporal patterns.

Usage:
    from src.simulation.iot_simulator import IoTSimulator
    sim = IoTSimulator(cfg)
    bins_df, readings_df = sim.run()
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from src.utils.helpers import get_logger, ensure_dir, output_path

logger = get_logger("IoTSimulator")


# ─── Zone fill-rate profiles ──────────────────────────────────────────────────
ZONE_PROFILES = {
    "Residential": {
        "base_rate": 0.035,   # % per 15 min
        "peak_hours": [7, 8, 19, 20, 21],
        "peak_factor": 2.2,
        "weekend_factor": 1.4,
        "noise_std": 0.005,
    },
    "Commercial": {
        "base_rate": 0.055,
        "peak_hours": [11, 12, 13, 17, 18],
        "peak_factor": 3.0,
        "weekend_factor": 0.5,
        "noise_std": 0.008,
    },
    "Industrial": {
        "base_rate": 0.025,
        "peak_hours": [8, 9, 10, 14, 15],
        "peak_factor": 2.0,
        "weekend_factor": 0.2,
        "noise_std": 0.003,
    },
    "Park": {
        "base_rate": 0.020,
        "peak_hours": [11, 12, 15, 16, 17],
        "peak_factor": 2.5,
        "weekend_factor": 2.0,
        "noise_std": 0.004,
    },
}


class IoTSimulator:
    """
    Simulates 50 smart bins across a city grid over N days.

    Attributes:
        cfg (dict): Loaded config.yaml
        n_bins (int): Number of bins
        n_days (int): Simulation duration
        interval (int): Sensor reading interval in minutes
        rng (np.random.Generator): Seeded RNG for reproducibility
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        sim_cfg = cfg["simulation"]
        self.n_bins = sim_cfg["n_bins"]
        self.n_days = sim_cfg["n_days"]
        self.interval = sim_cfg["sensor_interval_minutes"]
        self.seed = sim_cfg.get("random_seed", 42)
        self.rng = np.random.default_rng(self.seed)
        self.bin_capacity = cfg["fleet"]["bin_capacity_liters"]
        self.thresholds = cfg["thresholds"]
        self.out_dir = cfg["paths"]["outputs_dir"]
        ensure_dir(self.out_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run full simulation.

        Returns:
            bins_df: DataFrame with bin metadata (1 row per bin)
            readings_df: DataFrame with sensor readings (time-series)
        """
        logger.info(f"Starting simulation: {self.n_bins} bins × {self.n_days} days")
        bins_df = self._generate_bins()
        readings_df = self._simulate_readings(bins_df)
        self._save_data(bins_df, readings_df)
        logger.info(f"Simulation complete. {len(readings_df):,} readings generated.")
        return bins_df, readings_df

    def plot(self, bins_df: pd.DataFrame, readings_df: pd.DataFrame) -> None:
        """Generate and save IoT visualisation figure."""
        logger.info("Generating IoT visualisation...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("IoT Sensor Simulation — City Waste Management System",
                     fontsize=16, fontweight="bold", color="#0D1B2A")

        self._plot_fill_timeseries(axes[0, 0], readings_df)
        self._plot_hourly_patterns(axes[0, 1], readings_df)
        self._plot_zone_distributions(axes[1, 0], readings_df, bins_df)
        self._plot_bin_map(axes[1, 1], bins_df)

        plt.tight_layout()
        fig.savefig(output_path("01_iot_visualization.png", self.cfg),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure saved: 01_iot_visualization.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_bins(self) -> pd.DataFrame:
        """Create 50 bins with metadata (zone, location, fill-rate params)."""
        zone_cfg = self.cfg["simulation"]["zones"]
        zones, lats, lons, fill_rates = [], [], [], []

        for zone, count in zone_cfg.items():
            profile = ZONE_PROFILES[zone]
            for _ in range(count):
                zones.append(zone)
                lats.append(round(self.rng.uniform(0.05, 0.95), 4))
                lons.append(round(self.rng.uniform(0.05, 0.95), 4))
                fill_rates.append(round(
                    profile["base_rate"] * self.rng.uniform(0.8, 1.2), 5))

        return pd.DataFrame({
            "bin_id": [f"BIN_{i:03d}" for i in range(self.n_bins)],
            "zone": zones,
            "lat": lats,
            "lon": lons,
            "base_fill_rate": fill_rates,
            "capacity_liters": self.bin_capacity,
        })

    def _simulate_readings(self, bins_df: pd.DataFrame) -> pd.DataFrame:
        """Generate time-series fill-level readings for all bins."""
        timestamps = pd.date_range(
            start="2024-01-01",
            periods=self.n_days * 24 * (60 // self.interval),
            freq=f"{self.interval}min",
        )
        records = []

        for _, bin_row in bins_df.iterrows():
            profile = ZONE_PROFILES[bin_row["zone"]]
            fill = self.rng.uniform(0.05, 0.25)  # random initial fill

            for ts in timestamps:
                hour = ts.hour
                is_weekend = ts.dayofweek >= 5

                # Dynamic fill rate
                rate = bin_row["base_fill_rate"]
                if hour in profile["peak_hours"]:
                    rate *= profile["peak_factor"]
                if is_weekend:
                    rate *= profile["weekend_factor"]
                noise = self.rng.normal(0, profile["noise_std"])
                rate = max(0, rate + noise)

                fill = min(1.0, fill + rate)

                # Simulate collection event
                if fill >= self.thresholds["collection_trigger"]:
                    if self.rng.random() < 0.15:   # 15% chance collected each interval
                        fill = self.rng.uniform(0.02, 0.10)

                records.append({
                    "timestamp": ts,
                    "bin_id": bin_row["bin_id"],
                    "zone": bin_row["zone"],
                    "fill_level": round(fill, 4),
                    "fill_liters": round(fill * self.bin_capacity, 1),
                    "alert": (
                        "HIGH" if fill >= self.thresholds["high_priority"]
                        else "MEDIUM" if fill >= self.thresholds["collection_trigger"]
                        else "LOW"
                    ),
                })

        return pd.DataFrame(records)

    def _save_data(self, bins_df: pd.DataFrame, readings_df: pd.DataFrame) -> None:
        data_dir = self.cfg["paths"]["data_dir"]
        ensure_dir(data_dir)
        bins_df.to_csv(f"{data_dir}/bins_metadata.csv", index=False)
        readings_df.to_csv(f"{data_dir}/sensor_readings.csv", index=False)
        logger.info(f"Data saved → {data_dir}/")

    # ─── Plot helpers ─────────────────────────────────────────────────────────

    def _plot_fill_timeseries(self, ax, readings_df):
        sample_bins = readings_df["bin_id"].unique()[:6]
        colors = plt.cm.tab10.colors
        for i, bid in enumerate(sample_bins):
            data = readings_df[readings_df["bin_id"] == bid].head(200)
            ax.plot(data["timestamp"], data["fill_level"],
                    alpha=0.8, linewidth=1.2, color=colors[i], label=bid)
        ax.axhline(self.thresholds["high_priority"], color="red",
                   linestyle="--", linewidth=1, label="High Priority (80%)")
        ax.axhline(self.thresholds["collection_trigger"], color="orange",
                   linestyle="--", linewidth=1, label="Collection Trigger (60%)")
        ax.set_title("Fill Level Time-Series (Sample Bins)", fontweight="bold")
        ax.set_ylabel("Fill Level (0–1)")
        ax.legend(fontsize=7, ncol=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    def _plot_hourly_patterns(self, ax, readings_df):
        hourly = readings_df.copy()
        hourly["hour"] = pd.to_datetime(hourly["timestamp"]).dt.hour
        avg = hourly.groupby(["hour", "zone"])["fill_level"].mean().reset_index()
        colors = {"Residential": "#2196F3", "Commercial": "#FF9800",
                  "Industrial": "#9C27B0", "Park": "#4CAF50"}
        for zone in avg["zone"].unique():
            z = avg[avg["zone"] == zone]
            ax.plot(z["hour"], z["fill_level"], marker="o", markersize=4,
                    linewidth=2, label=zone, color=colors.get(zone, "gray"))
        ax.set_title("Average Fill Level by Hour & Zone", fontweight="bold")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Mean Fill Level")
        ax.set_xticks(range(0, 24, 2))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    def _plot_zone_distributions(self, ax, readings_df, bins_df):
        import seaborn as sns
        zone_fills = readings_df.groupby("zone")["fill_level"].mean()
        colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]
        bars = ax.bar(zone_fills.index, zone_fills.values,
                      color=colors[:len(zone_fills)], alpha=0.8, edgecolor="white")
        ax.axhline(self.thresholds["high_priority"], color="red",
                   linestyle="--", linewidth=1.5, label="High Priority Threshold")
        for bar, val in zip(bars, zone_fills.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title("Average Fill Level by Zone", fontweight="bold")
        ax.set_ylabel("Mean Fill Level")
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    def _plot_bin_map(self, ax, bins_df):
        colors_map = {"Residential": "#2196F3", "Commercial": "#FF9800",
                      "Industrial": "#9C27B0", "Park": "#4CAF50"}
        for zone, group in bins_df.groupby("zone"):
            ax.scatter(group["lon"], group["lat"], c=colors_map.get(zone, "gray"),
                       label=zone, s=80, alpha=0.8, edgecolors="white", linewidth=0.5)
        depot_cfg = self.cfg["fleet"]
        ax.scatter([depot_cfg["depot_lon"]], [depot_cfg["depot_lat"]],
                   marker="*", s=300, color="red", label="Depot", zorder=5)
        ax.set_title("City Bin Map (50 Smart Bins)", fontweight="bold")
        ax.set_xlabel("Longitude (normalized)")
        ax.set_ylabel("Latitude (normalized)")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
