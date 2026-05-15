import torch, asyncio
from io import StringIO
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from torch_geometric.data import Batch
from scandium.models.pignet_v2 import PIGNetV2
from scandium.physics.normalizer import Normalizer
from scandium.physics.constraints import check_violations
from scandium.physics.conformal import ConformalPredictor
from scandium.data.featurize_v2 import structure_to_graph_v2
from .cache import PredictionCache

CKPT = Path("backend/checkpoints/best_model_v2.pt")

class PredictorV2:
    """Production predictor. Singleton — load once at FastAPI startup."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None; self.normalizer = None; self.conformal = None
        self.cache = PredictionCache(maxsize=1024)
        self._load()

    def _load(self):
        ckpt            = torch.load(CKPT, map_location=self.device)
        cfg             = ckpt["config"]
        self.model      = PIGNetV2(cfg["hidden"], cfg["n_conv"], drop=0.0).to(self.device)
        state           = ckpt.get("ema_shadow") or ckpt["model_state"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.normalizer = Normalizer.from_state_dict(ckpt["normalizer"])
        # Load conformal quantiles if available
        if "conformal" in ckpt:
            self.conformal = ConformalPredictor.from_state_dict(ckpt["conformal"])
        meta = ckpt.get("metadata", {})
        print(f"PIGNet v2 loaded | version={meta.get('version','?')} | "
              f"val_bg_mae={meta.get('val_metrics',{}).get('val_bg_mae','?')}")

    async def predict_from_cif_async(self, cif: str) -> dict:
        """Async wrapper — returns cached result immediately if available."""
        cached = self.cache.get(cif)
        if cached: return {**cached, "cache_hit": True}
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._run_inference, cif)
        self.cache.set(cif, result)
        return {**result, "cache_hit": False}

    def _run_inference(self, cif: str) -> dict:
        try:
            structs = CifParser(StringIO(cif)).parse_structures(primitive=False)
            if not structs: raise ValueError("Invalid CIF — no structures found")
        except Exception as e:
            raise ValueError(f"Structure parse error: {e}")

        data  = structure_to_graph_v2(structs[0])
        batch = Batch.from_data_list([data]).to(self.device)

        with torch.no_grad():
            out  = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            phys = self.normalizer.denorm_dict(out)

        pred = {k: float(v[0]) for k, v in phys.items()}

        # Uncertainty via MC Dropout
        unc_out = self.model.predict_with_uncertainty(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch, T=30)
        unc_phys = self.normalizer.denorm_dict(
            {k.replace("_mean",""): v for k,v in unc_out.items() if "mean" in k})
        unc = {f"{k}_std": float(unc_out[f"{k}_std"][0] * self.normalizer.std[i])
               for i, k in enumerate(["band_gap","formation_energy","energy_above_hull"])}

        # Conformal intervals
        intervals = {}
        if self.conformal:
            for prop in ["band_gap","formation_energy","energy_above_hull"]:
                lo, hi = self.conformal.interval(pred[prop], unc[f"{prop}_std"], prop)
                intervals[prop] = {"lo": round(lo,4), "hi": round(hi,4)}

        return {
            "predictions":    {k: round(v,4) for k,v in pred.items()},
            "uncertainty":    {k: round(v,4) for k,v in unc.items()},
            "intervals_90pct": intervals,
            "violations":     check_violations(pred, unc),
            "structure_info": {
                "formula":    structs[0].formula,
                "n_sites":    structs[0].num_sites,
                "volume":     round(structs[0].volume, 3),
                "space_group": structs[0].get_space_group_info()[0],
            }
        }

predictor = PredictorV2()
