import json, time, statistics
from pathlib import Path
from collections import deque

LOG_FILE = Path("backend/logs/predictions.jsonl")
LOG_FILE.parent.mkdir(exist_ok=True)

# Training distribution bounds (update after each training run)
TRAIN_BOUNDS = {
    "band_gap":          (0.0,  10.0),
    "formation_energy":  (-5.5,  2.0),
    "energy_above_hull": (0.0,   3.0),
}

class PredictionMonitor:
    def __init__(self, window: int = 1000):
        self._window = deque(maxlen=window)
        self._total  = 0
        self._oods   = 0   # out-of-distribution count

    def log(self, result: dict, cif_hash: str):
        pred  = result.get("predictions", {})
        ood   = self._is_ood(pred)
        entry = {
            "ts":        time.time(),
            "hash":      cif_hash,
            "pred":      pred,
            "unc":       result.get("uncertainty"),
            "violations": result.get("violations"),
            "ood":       ood,
            "cache_hit": result.get("cache_hit", False),
        }
        self._window.append(entry)
        self._total += 1
        if ood: self._oods += 1
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _is_ood(self, pred: dict) -> bool:
        for prop, (lo, hi) in TRAIN_BOUNDS.items():
            v = pred.get(prop)
            if v is not None and not (lo <= v <= hi):
                return True
        return False

    def stats(self) -> dict:
        recent = list(self._window)
        if not recent: return {}
        bgs = [r["pred"].get("band_gap", 0) for r in recent if r["pred"]]
        return {
            "total_predictions": self._total,
            "ood_rate":          self._oods / max(self._total, 1),
            "recent_bg_mean":    statistics.mean(bgs) if bgs else None,
            "violation_rate":    sum(1 for r in recent if r["violations"]) / len(recent),
        }

monitor = PredictionMonitor()
