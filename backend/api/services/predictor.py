import torch
from pathlib import Path
from io import StringIO
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from torch_geometric.data import Batch
from scandium.models.pignet import PIGNet
from scandium.data.featurize import structure_to_graph
from scandium.physics.constraints import check_violations

CKPT = Path("backend/checkpoints/best_model.pt")

class PredictorService:
    """Singleton service. Loaded once at startup, reused for all requests."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._load_model()

    def _load_model(self):
        if not CKPT.exists():
            raise FileNotFoundError(f"No checkpoint at {CKPT}. Run train.py first.")
        ckpt = torch.load(CKPT, map_location=self.device)
        cfg  = ckpt.get("config", {})
        self.model = PIGNet(
            hidden_dim=cfg.get("hidden_dim", 256),
            n_conv=cfg.get("n_conv", 4),
            dropout=cfg.get("dropout", 0.1),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"PIGNet loaded from {CKPT} on {self.device} (epoch {ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss',0):.4f})")

    def predict_from_cif(self, cif_string: str, uncertainty: bool = True) -> dict:
        """Full pipeline: CIF string → dict with predictions, uncertainty, violations."""
        try:
            parser = CifParser(StringIO(cif_string))
            structures = parser.parse_structures(primitive=False)
            if not structures:
                raise ValueError("Invalid CIF file with no structures!")
            structure = structures[0]
        except Exception as e:
            raise ValueError(f"Structure parse error: {e}")

        data = structure_to_graph(structure)
        batch = Batch.from_data_list([data]).to(self.device)

        if uncertainty:
            unc = self.model.predict_with_uncertainty(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch, T=30
            )
            prediction = {
                "band_gap":          float(unc["band_gap_mean"][0]),
                "formation_energy":  float(unc["formation_energy_mean"][0]),
                "energy_above_hull": float(unc["energy_above_hull_mean"][0]),
            }
            uncertainty_out = {
                "band_gap_std":          float(unc["band_gap_std"][0]),
                "formation_energy_std":  float(unc["formation_energy_std"][0]),
                "energy_above_hull_std": float(unc["energy_above_hull_std"][0]),
            }
        else:
            with torch.no_grad():
                out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            prediction = {k: float(v[0]) for k, v in out.items()}
            uncertainty_out = {}

        violations = check_violations(prediction, uncertainty_out)

        return {
            "predictions": prediction,
            "uncertainty": uncertainty_out,
            "violations": violations,
            "structure_info": {
                "formula":    structure.formula,
                "n_sites":    structure.num_sites,
                "space_group": structure.get_space_group_info()[0],
                "volume":     structure.volume,
            }
        }

# Singleton — import this instance in FastAPI
predictor = PredictorService()