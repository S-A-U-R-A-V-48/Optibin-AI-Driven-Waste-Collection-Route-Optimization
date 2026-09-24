"""
models/rl_agent.py
==================
Q-Learning Reinforcement Learning agent for dynamic bin priority routing.

The agent learns to select HIGH-priority bins for collection in an order
that minimises total travel distance while maximising waste collected.

State:  Encoded (bin fill levels, truck load, current position)
Action: Select next bin to visit (from priority candidate set)
Reward: +fill_collected - 0.1 * distance_travelled

Usage:
    from src.models.rl_agent import QLearningAgent
    agent = QLearningAgent(cfg)
    history = agent.train(bins_df, readings_df)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from src.utils.helpers import get_logger, output_path, euclidean_distance

logger = get_logger("QLearningAgent")


class QLearningAgent:
    """
    Tabular Q-Learning agent for bin collection routing.

    Q(s, a) ← Q(s, a) + α · [r + γ · max Q(s', a') − Q(s, a)]

    State space is discretized (fill levels bucketed into 5 levels).
    Action space is the set of available priority bins.
    """

    def __init__(self, cfg: dict):
        rl_cfg = cfg["rl"]
        self.episodes = rl_cfg["episodes"]
        self.alpha = rl_cfg["learning_rate"]
        self.gamma = rl_cfg["discount_factor"]
        self.epsilon = rl_cfg["epsilon_start"]
        self.epsilon_min = rl_cfg["epsilon_min"]
        self.epsilon_decay = rl_cfg["epsilon_decay"]
        self.n_priority_bins = rl_cfg["n_priority_bins"]
        self.truck_capacity = cfg["fleet"]["truck_capacity_liters"]
        self.bin_capacity = cfg["fleet"]["bin_capacity_liters"]
        self.cfg = cfg
        self.Q = {}  # Q-table: state_key → {action: value}

    def train(self, bins_df: pd.DataFrame, readings_df: pd.DataFrame) -> dict:
        """
        Train Q-Learning agent and return training history.

        Returns:
            dict with episode_rewards, episode_distances, epsilon_history
        """
        logger.info(f"Training Q-Learning agent for {self.episodes} episodes...")

        # Select high-priority bins as candidate actions
        high_fill = (
            readings_df[readings_df["fill_level"] >= 0.6]
            .groupby("bin_id")["fill_level"].mean()
            .nlargest(self.n_priority_bins)
            .index.tolist()
        )
        priority_bins = bins_df[bins_df["bin_id"].isin(high_fill)].reset_index(drop=True)

        rewards, distances, epsilons = [], [], []

        for ep in range(self.episodes):
            ep_reward, ep_dist = self._run_episode(priority_bins)
            rewards.append(ep_reward)
            distances.append(ep_dist)
            epsilons.append(self.epsilon)

            # Decay exploration
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if (ep + 1) % 100 == 0:
                logger.info(f"  Episode {ep+1}/{self.episodes} | "
                            f"Avg Reward (last 50): {np.mean(rewards[-50:]):.3f} | "
                            f"ε={self.epsilon:.3f}")

        logger.info("Q-Learning training complete.")
        history = {
            "episode_rewards": rewards,
            "episode_distances": distances,
            "epsilon_history": epsilons,
        }
        self.plot_rl(history)
        return history

    def _run_episode(self, priority_bins: pd.DataFrame) -> tuple[float, float]:
        """Simulate one collection episode with ε-greedy policy."""
        visited = set()
        pos = np.array([0.5, 0.5])   # depot position
        load = 0.0
        total_reward = 0.0
        total_dist = 0.0
        available = list(priority_bins.index)

        while available and load < self.truck_capacity:
            state = self._encode_state(priority_bins, visited, load)

            if np.random.random() < self.epsilon:
                action = np.random.choice(available)   # explore
            else:
                action = self._best_action(state, available)   # exploit

            # Collect selected bin
            bin_row = priority_bins.iloc[action]
            next_pos = np.array([bin_row["lat"], bin_row["lon"]])
            dist = euclidean_distance(pos[0], pos[1], next_pos[0], next_pos[1])
            fill = bin_row["base_fill_rate"] * 1000    # proxy liters
            collected = min(fill, self.truck_capacity - load)

            reward = collected / self.bin_capacity - 0.1 * dist
            load += collected
            total_reward += reward
            total_dist += dist
            visited.add(action)
            available.remove(action)
            pos = next_pos

            # TD Update
            next_state = self._encode_state(priority_bins, visited, load)
            self._update_q(state, action, reward, next_state, available)

        return total_reward, total_dist

    def _encode_state(self, bins_df, visited, load) -> str:
        fill_bucket = int(min(4, load / self.truck_capacity * 5))
        visited_count = len(visited)
        return f"{fill_bucket}_{visited_count}"

    def _best_action(self, state, available) -> int:
        q_vals = self.Q.get(state, {})
        best = max(available, key=lambda a: q_vals.get(a, 0.0))
        return best

    def _update_q(self, state, action, reward, next_state, available) -> None:
        if state not in self.Q:
            self.Q[state] = {}
        if next_state not in self.Q:
            self.Q[next_state] = {}

        next_q = max([self.Q[next_state].get(a, 0.0) for a in available], default=0.0)
        current_q = self.Q[state].get(action, 0.0)
        self.Q[state][action] = current_q + self.alpha * (
            reward + self.gamma * next_q - current_q
        )

    def plot_rl(self, history: dict) -> None:
        """Generate RL training convergence plot."""
        rewards = history["episode_rewards"]
        window = 30
        smoothed = pd.Series(rewards).rolling(window).mean()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Reinforcement Learning — Q-Learning Training",
                     fontsize=14, fontweight="bold")

        # Reward convergence
        axes[0].plot(rewards, alpha=0.3, color="#2196F3", linewidth=0.8)
        axes[0].plot(smoothed, color="#FF9800", linewidth=2.5,
                     label=f"Smoothed (window={window})")
        axes[0].set_title("Episode Reward Convergence", fontweight="bold")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Total Reward")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Distance trend
        dist = history["episode_distances"]
        axes[1].plot(pd.Series(dist).rolling(window).mean(),
                     color="#4CAF50", linewidth=2)
        axes[1].set_title("Collection Distance (Smoothed)", fontweight="bold")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Total Distance")
        axes[1].grid(alpha=0.3)

        # Epsilon decay
        axes[2].plot(history["epsilon_history"], color="#9C27B0", linewidth=2)
        axes[2].set_title("Exploration Rate (ε) Decay", fontweight="bold")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Epsilon (ε)")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path("05_rf_rl.png", self.cfg),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure saved: 05_rf_rl.png")


class RandomForestDemandForecaster:
    """
    Random Forest model for zone-level waste demand forecasting.
    Complements the bin-level GBM predictor with aggregate predictions.
    """

    def __init__(self, cfg: dict):
        ml_cfg = cfg["ml"]
        self.n_estimators = ml_cfg["n_estimators"]
        self.test_size = ml_cfg["test_size"]
        self.model = None
        self.cfg = cfg

    def run(self, readings_df: pd.DataFrame) -> dict:
        """Train RF demand model and return metrics."""
        logger.info("Training Random Forest demand forecaster...")
        df = self._prepare(readings_df)
        X_train, X_test, y_train, y_test = self._split(df)

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, n_jobs=-1, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }
        cv_scores = cross_val_score(self.model, X_train, y_train,
                                    cv=5, scoring="r2")
        metrics["cv_r2_mean"] = cv_scores.mean()
        metrics["cv_r2_std"] = cv_scores.std()
        logger.info(f"RF MAE={metrics['mae']:.4f} | R²={metrics['r2']:.4f}")
        return metrics

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["zone_enc"] = df["zone"].astype("category").cat.codes
        df["lag_1"] = df.groupby("bin_id")["fill_level"].shift(1)
        df["lag_4"] = df.groupby("bin_id")["fill_level"].shift(4)
        df["rolling_mean"] = df.groupby("bin_id")["fill_level"].transform(
            lambda x: x.shift(1).rolling(8).mean())
        return df.dropna()

    def _split(self, df):
        features = ["hour", "day_of_week", "is_weekend", "zone_enc",
                    "lag_1", "lag_4", "rolling_mean"]
        X = df[features]
        y = df["fill_level"]
        split = int(len(X) * (1 - self.test_size))
        return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
