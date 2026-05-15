import torch, numpy as np

class ConformalPredictor:
    """
    Split conformal prediction for crystal property prediction.
    Calibrate on a held-out calibration set (distinct from test set).
    At inference, produce guaranteed coverage intervals.

    Usage:
        cal = ConformalPredictor()
        cal.calibrate(predictions, targets, alpha=0.1)   # 90% coverage
        lo, hi = cal.predict_interval(new_pred, new_std) # guaranteed interval
    """
    def __init__(self):
        self.quantiles = {}   # {property: q_hat}

    def calibrate(
        self,
        preds:   dict,   # {property_mean: (N,), property_std: (N,)}
        targets: dict,   # {property: (N,)}
        alpha:   float = 0.1,  # 1 - coverage (0.1 = 90% coverage)
    ):
        """
        Compute nonconformity scores on calibration set.
        Score = |y - ŷ| / σ  (standardised residual)
        q_hat = (1-alpha)-quantile of scores.
        """
        props = ["band_gap", "formation_energy", "energy_above_hull"]
        for prop in props:
            y_hat = preds[f"{prop}_mean"]
            sigma = preds[f"{prop}_std"] + 1e-6
            y     = targets[prop]
            scores = ((y - y_hat).abs() / sigma)       # nonconformity scores
            n      = len(scores)
            level  = np.ceil((n + 1) * (1 - alpha)) / n  # finite-sample adjustment
            self.quantiles[prop] = float(np.quantile(scores.numpy(), level))
            print(f"  {prop} q_hat = {self.quantiles[prop]:.3f} (α={alpha})")

    def interval(self, pred_mean: float, pred_std: float, prop: str) -> tuple[float, float]:
        """Return calibrated prediction interval [lo, hi] for a single prediction."""
        q = self.quantiles.get(prop, 2.0)
        w = q * (pred_std + 1e-6)
        return pred_mean - w, pred_mean + w

    def state_dict(self): return self.quantiles

    @classmethod
    def from_state_dict(cls, d):
        obj = cls(); obj.quantiles = d; return obj
