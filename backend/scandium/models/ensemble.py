import torch
from pathlib import Path
from .pignet_v2 import PIGNetV2
from ..physics.normalizer import Normalizer

class PIGNetEnsemble:
    """
    Deep ensemble of N independently trained PIGNet v2 models.
    Each model was trained with a different random seed.
    Returns mean prediction + standard deviation as uncertainty.
    """
    def __init__(self, ckpt_paths: list[str], device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models, self.normalizers = [], []
        for path in ckpt_paths:
            ckpt   = torch.load(path, map_location=self.device)
            cfg    = ckpt["config"]
            model  = PIGNetV2(cfg["hidden"], cfg["n_conv"], cfg["drop"]).to(self.device)
            # Load EMA shadow weights if present, else regular weights
            state  = {k.replace("module.",""): v for k,v in
                      (ckpt.get("ema_shadow") or ckpt["model_state"]).items()}
            model.load_state_dict(state, strict=False)
            model.eval()
            self.models.append(model)
            self.normalizers.append(Normalizer.from_state_dict(ckpt["normalizer"]))
        print(f"Ensemble: {len(self.models)} models loaded on {self.device}")

    @torch.no_grad()
    def predict(self, x, edge_index, edge_attr, batch) -> dict:
        """
        Returns: {
          band_gap_mean, band_gap_std,
          formation_energy_mean, formation_energy_std,
          energy_above_hull_mean, energy_above_hull_std,
        }
        All in physical units (eV / eV·atom⁻¹).
        """
        keys  = ["band_gap", "formation_energy", "energy_above_hull"]
        preds = {k: [] for k in keys}

        for model, norm in zip(self.models, self.normalizers):
            out  = model(x, edge_index, edge_attr, batch)
            phys = norm.denorm_dict(out)   # back to physical units
            for k in keys:
                preds[k].append(phys[k].cpu())

        result = {}
        for k in keys:
            stacked          = torch.stack(preds[k])   # (N_models, B)
            result[f"{k}_mean"] = stacked.mean(0)
            result[f"{k}_std"]  = stacked.std(0)
        return result
