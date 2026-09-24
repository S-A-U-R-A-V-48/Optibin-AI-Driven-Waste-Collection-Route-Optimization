"""
utils/helpers.py
================
Shared utility functions used across all modules.
"""
import os
import yaml
import logging
from pathlib import Path

# ─── Logging ─────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with timestamp formatting."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        # Try searching up from current file
        path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Filesystem ──────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def output_path(filename: str, cfg: dict) -> str:
    """Resolve full output file path."""
    out_dir = cfg.get("paths", {}).get("outputs_dir", "outputs")
    ensure_dir(out_dir)
    return os.path.join(out_dir, filename)


# ─── Math / Stats ─────────────────────────────────────────────────────────────

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute Haversine distance in km between two lat/lon coordinates.
    For normalized grid coordinates (0–1), approximates city-scale distances.
    """
    import math
    R = 6371.0  # Earth radius km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance on a normalized grid."""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def format_km(km: float) -> str:
    return f"{km:.1f} km"


def format_pct(pct: float) -> str:
    return f"{pct:.1f}%"


def format_money(amount: float, symbol: str = "$") -> str:
    if amount >= 1_000_000:
        return f"{symbol}{amount/1_000_000:.2f}M"
    elif amount >= 1_000:
        return f"{symbol}{amount/1_000:.1f}K"
    return f"{symbol}{amount:.0f}"
