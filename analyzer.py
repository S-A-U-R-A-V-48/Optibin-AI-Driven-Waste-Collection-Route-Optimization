"""
eda/analyzer.py
================
Exploratory Data Analysis for waste sensor readings.
Generates overflow risk analysis, temporal heatmaps, and zone statistics.

Usage:
    from src.eda.analyzer import EDAAnalyzer
    eda = EDAAnalyzer(cfg)
    eda.run(bins_df, readings_df)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
from src.utils.helpers import get_logger, output_path

logger = get_logger("EDAAnalyzer")


class EDAAnalyzer:
    """
    Performs exploratory analysis on sensor readings.

    Key analyses:
        - Overflow risk per bin
        - Day-of-week × hour heatmap
        - Fill level distributions by zone
        - Statistical summaries
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.thresholds = cfg["thresholds"]

    def run(self, bins_df: pd.DataFrame, readings_df: pd.DataFrame) -> dict:
        """
        Run full EDA and return summary statistics.

        Returns:
            dict: overflow_stats, zone_stats, peak_hours
        """
        logger.info("Running EDA...")
        readings_df = readings_df.copy()
        readings_df["timestamp"] = pd.to_datetime(readings_df["timestamp"])
        readings_df["hour"] = readings_df["timestamp"].dt.hour
        readings_df["day_of_week"] = readings_df["timestamp"].dt.dayofweek
        readings_df["day_name"] = readings_df["timestamp"].dt.day_name()

        overflow_stats = self._overflow_risk(readings_df)
        zone_stats = self._zone_statistics(readings_df)
        peak_hours = self._peak_hours(readings_df)

        self.plot(readings_df, overflow_stats)

        logger.info(f"EDA complete. Bins at overflow risk: {len(overflow_stats[overflow_stats['overflow_rate']>0.1])}")
        return {
            "overflow_stats": overflow_stats,
            "zone_stats": zone_stats,
            "peak_hours": peak_hours,
        }

    def plot(self, readings_df: pd.DataFrame, overflow_stats: pd.DataFrame) -> None:
        """Generate and save EDA figure."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Exploratory Data Analysis — Waste Pattern Insights",
                     fontsize=15, fontweight="bold")

        self._plot_overflow_risk(axes[0], overflow_stats)
        self._plot_heatmap(axes[1], readings_df)
        self._plot_daily_trend(axes[2], readings_df)

        plt.tight_layout()
        fig.savefig(output_path("02_eda_analysis.png", self.cfg),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure saved: 02_eda_analysis.png")

    # ─── Analysis methods ────────────────────────────────────────────────────

    def _overflow_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute overflow risk (% of readings ≥ high_priority threshold) per bin."""
        threshold = self.thresholds["high_priority"]
        risk = df.groupby("bin_id").apply(
            lambda g: pd.Series({
                "overflow_rate": (g["fill_level"] >= threshold).mean(),
                "mean_fill": g["fill_level"].mean(),
                "max_fill": g["fill_level"].max(),
                "zone": g["zone"].iloc[0],
            })
        ).reset_index()
        return risk.sort_values("overflow_rate", ascending=False)

    def _zone_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-zone descriptive statistics."""
        return df.groupby("zone")["fill_level"].agg(
            ["mean", "std", "min", "max", "median"]
        ).round(4)

    def _peak_hours(self, df: pd.DataFrame) -> pd.Series:
        """Identify top-5 peak hours by average fill level."""
        return df.groupby("hour")["fill_level"].mean().nlargest(5)

    # ─── Plot helpers ─────────────────────────────────────────────────────────

    def _plot_overflow_risk(self, ax, overflow_stats):
        top = overflow_stats.head(20)
        colors_map = {"Residential": "#2196F3", "Commercial": "#FF9800",
                      "Industrial": "#9C27B0", "Park": "#4CAF50"}
        bar_colors = [colors_map.get(z, "gray") for z in top["zone"]]
        bars = ax.barh(top["bin_id"], top["overflow_rate"] * 100,
                       color=bar_colors, alpha=0.85, edgecolor="white")
        ax.axvline(10, color="red", linestyle="--", linewidth=1.5,
                   label="10% Risk Threshold")
        ax.set_title("Top 20 Bins — Overflow Risk %", fontweight="bold")
        ax.set_xlabel("Overflow Rate (%)")
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    def _plot_heatmap(self, ax, df):
        pivot = df.groupby(["day_of_week", "hour"])["fill_level"].mean().unstack()
        pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=False, fmt=".2f",
                    linewidths=0.2, cbar_kws={"label": "Avg Fill Level"})
        ax.set_title("Fill Level Heatmap (Day × Hour)", fontweight="bold")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Day of Week")

    def _plot_daily_trend(self, ax, df):
        df["date"] = df["timestamp"].dt.date
        daily = df.groupby("date")["fill_level"].agg(["mean", "max"]).reset_index()
        ax.fill_between(range(len(daily)), daily["mean"], alpha=0.3, color="#2196F3",
                        label="Mean Fill Level")
        ax.plot(range(len(daily)), daily["mean"], color="#2196F3", linewidth=2)
        ax.plot(range(len(daily)), daily["max"], color="#FF5252", linewidth=1.5,
                linestyle="--", label="Max Fill Level")
        ax.axhline(self.thresholds["high_priority"], color="red", linestyle=":",
                   linewidth=1.2, label=f"High Priority ({self.thresholds['high_priority']*100:.0f}%)")
        ax.set_title("Daily Fill Level Trend (30 Days)", fontweight="bold")
        ax.set_xlabel("Day")
        ax.set_ylabel("Fill Level")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
