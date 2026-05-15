import torch

class Normalizer:
    """
    Normalizes the (3,) target vector to zero mean, unit std.
    Must be fitted on the TRAINING set only.
    Saved inside the checkpoint dict and loaded at inference.
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean   # shape (3,): [bg_mean, fe_mean, hull_mean]
        self.std  = std    # shape (3,): [bg_std,  fe_std,  hull_std]

    def norm(self, y: torch.Tensor) -> torch.Tensor:
        """Normalize targets. y shape: (B, 3)"""
        return (y - self.mean.to(y.device)) / (self.std.to(y.device) + 1e-8)

    def denorm(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize predictions back to physical units."""
        return y * (self.std.to(y.device) + 1e-8) + self.mean.to(y.device)

    def denorm_dict(self, d: dict) -> dict:
        """Denormalize a predictions dict {band_gap, formation_energy, energy_above_hull}."""
        keys = ["band_gap", "formation_energy", "energy_above_hull"]
        stacked = torch.stack([d[k] for k in keys], dim=-1)   # (B, 3)
        out     = self.denorm(stacked)
        return {k: out[..., i] for i, k in enumerate(keys)}

    @classmethod
    def fit(cls, data_list) -> "Normalizer":
        """Fit normalizer on a list of PyG Data objects."""
        targets = torch.cat([d.y for d in data_list], dim=0)  # (N, 3)
        return cls(targets.mean(dim=0), targets.std(dim=0))

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, d: dict) -> "Normalizer":
        return cls(d["mean"], d["std"])
