"""
optimization/cvrp_solver.py
============================
Capacitated Vehicle Routing Problem (CVRP) solver.

Implements:
    1. Nearest-Neighbour (NN) construction heuristic — greedy baseline
    2. 2-opt local search improvement — iterative edge reversal
    3. CO2 emission modelling
    4. Multi-truck route construction

Mathematical formulation:
    Minimise:  Σ_{k} Σ_{(i,j)∈r_k} d_ij · x_ijk
    Subject to:
        Each bin visited exactly once
        Truck load ≤ capacity
        All routes begin and end at depot

Usage:
    from src.optimization.cvrp_solver import CVRPSolver
    solver = CVRPSolver(cfg)
    routes, stats = solver.run(bins_df, readings_df)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
from src.utils.helpers import get_logger, output_path, euclidean_distance

logger = get_logger("CVRPSolver")

CO2_PER_KM = 0.27  # kg CO2 per km (diesel truck average)
FUEL_COST_PER_KM = 0.18  # USD per km


class CVRPSolver:
    """
    Solves the CVRP with nearest-neighbour construction + 2-opt improvement.

    Attributes:
        n_trucks (int): Fleet size
        capacity (float): Truck capacity in litres
        bin_capacity (float): Bin volume in litres
        collection_trigger (float): Fill threshold for scheduling
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        fleet = cfg["fleet"]
        self.n_trucks = fleet["n_trucks"]
        self.truck_capacity = fleet["truck_capacity_liters"]
        self.bin_capacity = fleet["bin_capacity_liters"]
        self.depot = np.array([fleet["depot_lat"], fleet["depot_lon"]])
        self.collection_trigger = cfg["thresholds"]["collection_trigger"]
        self.max_iters_2opt = cfg["cvrp"]["max_iterations_2opt"]

    def run(self, bins_df: pd.DataFrame, readings_df: pd.DataFrame) -> tuple[list, dict]:
        """
        Full CVRP solve pipeline.

        Returns:
            routes: list of truck routes (list of bin_ids per truck)
            stats: distance, CO2, improvement metrics
        """
        logger.info("Selecting bins for collection...")
        candidate_bins = self._select_candidates(bins_df, readings_df)
        logger.info(f"{len(candidate_bins)} bins queued for collection")

        dist_matrix = self._build_distance_matrix(candidate_bins)

        logger.info("Solving CVRP — Nearest Neighbour heuristic...")
        nn_routes = self._nearest_neighbour(candidate_bins, dist_matrix)
        nn_dist = self._total_distance(nn_routes, dist_matrix)

        logger.info("Improving with 2-opt local search...")
        opt_routes = [self._two_opt(route, dist_matrix) for route in nn_routes]
        opt_dist = self._total_distance(opt_routes, dist_matrix)

        improvement = (nn_dist - opt_dist) / nn_dist * 100
        stats = {
            "n_bins": len(candidate_bins),
            "nn_distance_km": nn_dist,
            "optimized_distance_km": opt_dist,
            "improvement_pct": improvement,
            "co2_saved_kg": (nn_dist - opt_dist) * CO2_PER_KM,
            "fuel_saved_usd": (nn_dist - opt_dist) * FUEL_COST_PER_KM,
            "n_trucks_used": sum(1 for r in opt_routes if len(r) > 0),
        }

        logger.info(f"NN: {nn_dist:.1f} km → 2-opt: {opt_dist:.1f} km "
                    f"| Improvement: {improvement:.1f}%")

        self.plot(candidate_bins, nn_routes, opt_routes, stats)
        return opt_routes, stats

    # ─── Core algorithms ──────────────────────────────────────────────────────

    def _nearest_neighbour(self, bins_df: pd.DataFrame, dist_matrix: np.ndarray) -> list:
        """
        Nearest-Neighbour CVRP construction heuristic.
        Greedy: always visit the closest unvisited bin within capacity.
        """
        n = len(bins_df)
        depot_idx = n   # depot is virtual node at index n
        unvisited = set(range(n))
        routes = []

        for truck in range(self.n_trucks):
            if not unvisited:
                break
            route = []
            pos = depot_idx
            load = 0.0

            while unvisited:
                # Find nearest feasible bin
                candidates = [
                    b for b in unvisited
                    if load + bins_df.iloc[b]["demand_liters"] <= self.truck_capacity
                ]
                if not candidates:
                    break

                if pos == depot_idx:
                    distances = [self._depot_dist(bins_df.iloc[b]) for b in candidates]
                else:
                    distances = [dist_matrix[pos, b] for b in candidates]

                nearest = candidates[np.argmin(distances)]
                route.append(nearest)
                load += bins_df.iloc[nearest]["demand_liters"]
                unvisited.discard(nearest)
                pos = nearest

            routes.append(route)

        return routes

    def _two_opt(self, route: list, dist_matrix: np.ndarray) -> list:
        """
        2-opt local search: reverse sub-segments to eliminate crossing edges.
        Guarantees monotone improvement — O(n²) per iteration.
        """
        if len(route) < 4:
            return route

        best = route[:]
        improved = True
        iteration = 0

        while improved and iteration < self.max_iters_2opt:
            improved = False
            iteration += 1
            for i in range(len(best) - 1):
                for j in range(i + 2, len(best)):
                    d_before = dist_matrix[best[i], best[i+1]] + dist_matrix[best[j-1], best[j]]
                    d_after  = dist_matrix[best[i], best[j-1]] + dist_matrix[best[i+1], best[j]]
                    if d_after < d_before - 1e-6:
                        best[i+1:j] = reversed(best[i+1:j])
                        improved = True

        return best

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _select_candidates(self, bins_df, readings_df) -> pd.DataFrame:
        """Select bins whose current fill level exceeds collection trigger."""
        latest = (readings_df.sort_values("timestamp")
                              .groupby("bin_id")
                              .last()["fill_level"]
                              .reset_index())
        to_collect = latest[latest["fill_level"] >= self.collection_trigger]
        candidates = bins_df[bins_df["bin_id"].isin(to_collect["bin_id"])].copy()
        candidates = candidates.merge(to_collect, on="bin_id")
        candidates["demand_liters"] = candidates["fill_level"] * self.bin_capacity
        return candidates.reset_index(drop=True)

    def _build_distance_matrix(self, bins_df: pd.DataFrame) -> np.ndarray:
        n = len(bins_df)
        D = np.zeros((n, n))
        coords = bins_df[["lat", "lon"]].values
        for i in range(n):
            for j in range(i + 1, n):
                d = euclidean_distance(coords[i,0], coords[i,1], coords[j,0], coords[j,1])
                D[i,j] = D[j,i] = d
        return D

    def _depot_dist(self, bin_row) -> float:
        return euclidean_distance(self.depot[0], self.depot[1],
                                  bin_row["lat"], bin_row["lon"])

    def _total_distance(self, routes: list, dist_matrix: np.ndarray) -> float:
        total = 0.0
        for route in routes:
            if not route:
                continue
            # depot → first bin
            # (approximated as 0 for matrix-only distances in demo)
            for k in range(len(route) - 1):
                total += dist_matrix[route[k], route[k+1]]
        return total

    # ─── Plotting ─────────────────────────────────────────────────────────────

    def plot(self, bins_df, nn_routes, opt_routes, stats) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("CVRP Route Optimization — Nearest-Neighbour vs 2-opt",
                     fontsize=14, fontweight="bold")

        palette = ["#E74C3C", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
        zone_colors = {"Residential": "#2196F3", "Commercial": "#FF9800",
                       "Industrial": "#9C27B0", "Park": "#4CAF50"}

        for ax, routes, title, dist in [
            (axes[0], nn_routes, f"Nearest-Neighbour Baseline\n{stats['nn_distance_km']:.1f} km total", stats['nn_distance_km']),
            (axes[1], opt_routes, f"2-opt Optimized\n{stats['optimized_distance_km']:.1f} km total ({stats['improvement_pct']:.1f}% reduction)", stats['optimized_distance_km']),
        ]:
            # Scatter bins
            for zone, grp in bins_df.groupby("zone"):
                ax.scatter(grp["lon"], grp["lat"], c=zone_colors.get(zone, "gray"),
                           s=60, alpha=0.7, zorder=3, edgecolors="white", linewidth=0.4)

            # Depot
            ax.scatter([self.depot[1]], [self.depot[0]], marker="*",
                       s=250, color="red", zorder=5, label="Depot")

            # Routes
            for t_idx, route in enumerate(routes):
                color = palette[t_idx % len(palette)]
                coords = [self.depot] + [bins_df.iloc[b][["lat","lon"]].values for b in route] + [self.depot]
                xs = [c[1] for c in coords]
                ys = [c[0] for c in coords]
                ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.8,
                        label=f"Truck {t_idx+1}" if route else None)

            ax.set_title(title, fontweight="bold", fontsize=10)
            ax.set_xlabel("Longitude (normalized)")
            ax.set_ylabel("Latitude (normalized)")
            ax.legend(fontsize=7, loc="upper right")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path("04_route_optimization.png", self.cfg),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure saved: 04_route_optimization.png")
