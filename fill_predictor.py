"""
models/fill_predictor.py
=========================
Gradient Boosting fill-level predictor with 24-hour forecast horizon.

Pipeline:
    1. Time-series feature engineering (lag features, rolling stats, temporal dummies)
    2. Train/test split (last 20% of time-series)
    3. GBM training (sklearn GradientBoostingRegressor)
    4. Evaluation: MAE, RMSE, R²
    5. Bin alert classification (LOW / MEDIUM / HIGH)

Usage:
    from src.models.fill_predictor import FillPredictor
    predictor = FillPredictor(cfg)
    model, metrics = predictor.run(readings_df)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from src.utils.helpers import get_logger, output_path

logger = get_logger("FillPredictor")


class FillPredictor:
    """
    Trains a Gradient Boosting model to predict fill levels 24 hours ahead.

    Features:
        - Lag fill levels: t-1, t-2, t-4, t-8 intervals
        - Rolling mean / std over past 4 and 8 readings
        - Hour of day, day of week, is_weekend (one-hot style)
        - Zone encoding
    """

    FORECAST_SHIFTS = [1, 2, 4, 8]   # reading intervals back

    def __init__(self, cfg: dict):
        self.cfg = cfg
        ml_cfg = cfg["ml"]
        self.test_size = ml_cfg["test_size"]
        self.n_estimators = ml_cfg["n_estimators"]
        self.max_depth = ml_cfg["max_depth"]
        self.lr = ml_cfg["learning_rate"]
        self.window = ml_cfg["window_size"]
        self.thresholds = cfg["thresholds"]
        self.model = None
        self.feature_names = None

    def run(self, readings_df: pd.DataFrame) -> tuple[GradientBoostingRegressor, dict]:
        """
        Full training pipeline.

        Returns:
            model: trained GBM model
            metrics: dict with MAE, RMSE, R², alert_accuracy
        """
        logger.info("Building features for fill-level prediction...")
        features_df = self._engineer_features(readings_df)

        X_train, X_test, y_train, y_test = self._split(features_df)
        logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,} samples")

        self.model = self._train(X_train, y_train)
        metrics = self._evaluate(X_test, y_test)

        logger.info(f"MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}")
        self.plot(X_test, y_test, metrics)
        return self.model, metrics

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict fill levels from a feature DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call run() first.")
        return self.model.predict(features[self.feature_names])

    def classify_alert(self, fill_level: float) -> str:
        """Map fill level to alert category."""
        if fill_level >= self.thresholds["high_priority"]:
            return "HIGH"
        elif fill_level >= self.thresholds["collection_trigger"]:
            return "MEDIUM"
        return "LOW"

    def plot(self, X_test, y_test, metrics) -> None:
        """Generate and save prediction evaluation figure."""
        y_pred = self.model.predict(X_test)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("ML Fill Level Prediction — Model Evaluation",
                     fontsize=15, fontweight="bold")

        # Actual vs Predicted
        ax = axes[0, 0]
        ax.scatter(y_test[:500], y_pred[:500], alpha=0.3, s=12, color="#2196F3")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Fit")
        ax.set_title(f"Actual vs Predicted | R²={metrics['r2']:.3f}", fontweight="bold")
        ax.set_xlabel("Actual Fill Level")
        ax.set_ylabel("Predicted Fill Level")
        ax.legend()
        ax.grid(alpha=0.3)

        # Residuals
        ax = axes[0, 1]
        residuals = y_test - y_pred
        ax.hist(residuals, bins=50, color="#FF9800", alpha=0.8, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"Residual Distribution | MAE={metrics['mae']:.4f}", fontweight="bold")
        ax.set_xlabel("Residual (Actual − Predicted)")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)

        # Time-series sample
        ax = axes[1, 0]
        n = 200
        ax.plot(range(n), y_test[:n], label="Actual", color="#2196F3", linewidth=1.5)
        ax.plot(range(n), y_pred[:n], label="Predicted", color="#FF9800",
                linewidth=1.5, linestyle="--", alpha=0.9)
        ax.set_title("Actual vs Predicted — Time Series Sample", fontweight="bold")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Fill Level")
        ax.legend()
        ax.grid(alpha=0.3)

        # Feature importance
        ax = axes[1, 1]
        importances = self.model.feature_importances_
        top_idx = np.argsort(importances)[-10:]
        ax.barh(np.array(self.feature_names)[top_idx], importances[top_idx],
                color="#4CAF50", alpha=0.85, edgecolor="white")
        ax.set_title("Top 10 Feature Importances", fontweight="bold")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_path("03_lstm_prediction.png", self.cfg),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure saved: 03_lstm_prediction.png")

    # ─── Private ──────────────────────────────────────────────────────────────

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build time-series lag + rolling features."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["bin_id", "timestamp"])

        # Temporal features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["month"] = df["timestamp"].dt.month

        # Zone encoding
        le = LabelEncoder()
        df["zone_enc"] = le.fit_transform(df["zone"])

        # Lag features (group by bin_id to avoid cross-bin leakage)
        grp = df.groupby("bin_id")["fill_level"]
        for lag in self.FORECAST_SHIFTS:
            df[f"lag_{lag}"] = grp.shift(lag)

        # Rolling statistics
        for window in [4, 8]:
            df[f"rolling_mean_{window}"] = grp.transform(
                lambda x: x.shift(1).rolling(window).mean())
            df[f"rolling_std_{window}"] = grp.transform(
                lambda x: x.shift(1).rolling(window).std())

        # Target: fill level in next reading
        df["target"] = grp.shift(-1)

        # Use only commercial bins for demo (higher volatility)
        df = df[df["zone"] == "Commercial"].dropna()
        return df

    def _split(self, df: pd.DataFrame):
        feature_cols = (
            [f"lag_{l}" for l in self.FORECAST_SHIFTS]
            + [f"rolling_mean_{w}" for w in [4, 8]]
            + [f"rolling_std_{w}" for w in [4, 8]]
            + ["hour", "day_of_week", "is_weekend", "month", "zone_enc"]
        )
        self.feature_names = feature_cols
        X = df[feature_cols]
        y = df["target"]
        split = int(len(X) * (1 - self.test_size))
        return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    def _train(self, X_train, y_train) -> GradientBoostingRegressor:
        model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.lr,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model

    def _evaluate(self, X_test, y_test) -> dict:
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)
        return {"mae": mae, "rmse": rmse, "r2": r2}
