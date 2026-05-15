"""
Run conformal calibration on the validation set after ensemble training.

Usage:
    python scripts/calibrate_conformal.py --data_dir data/processed_v2
"""

from __future__ import annotations
import argparse
import sys
import torch
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scandium.models.pignet_v2 import PIGNetV2
from scandium.physics.normalizer import Normalizer
from scandium.training.calibrate import ConformalPredictor
from torch_geometric.loader import DataLoader


ENSEMBLE_SEEDS = [42, 123, 456, 789, 1337]
CKPT_DIR       = Path("backend/checkpoints")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed_v2")
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_data = torch.load(Path(args.data_dir) / "val.pt")
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=4)

    print(f"[Calibrate] Running on {len(val_data):,} validation samples")

    # ── Load all ensemble members ─────────────────────────────────────────────
    members = []
    for seed in ENSEMBLE_SEEDS:
        ckpt_path = CKPT_DIR / f"best_ensemble_seed_{seed}.pt"
        if not ckpt_path.exists():
            print(f"[Calibrate] WARNING: {ckpt_path} not found — skipping")
            continue

        ckpt  = torch.load(ckpt_path, map_location=device)
        model = PIGNetV2(**{k: ckpt["config"][k]
                            for k in ("hidden_dim", "n_conv", "dropout")}).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        norm = Normalizer.from_state_dict(ckpt["normalizer"])
        members.append((model, norm))

    if not members:
        raise SystemExit("[Calibrate] No ensemble members found. Run train_ensemble.py first.")

    print(f"[Calibrate] Loaded {len(members)} ensemble members")

    # ── Collect ensemble predictions on val set ───────────────────────────────
    all_preds  = defaultdict(list)
    all_targets = defaultdict(list)
    keys = ["band_gap", "formation_energy", "energy_above_hull"]

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            y     = batch.y.squeeze(1)   # (B, 3)

            # Ensemble mean prediction
            batch_preds = defaultdict(list)
            for (model, norm) in members:
                raw   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                phys  = norm.denorm_dict(raw)
                for k in keys:
                    if k in phys:
                        batch_preds[k].append(phys[k].cpu())

            for i, k in enumerate(keys):
                if k in batch_preds:
                    # Ensemble mean
                    mean_pred = torch.stack(batch_preds[k]).mean(dim=0)
                    all_preds[k].extend(mean_pred.tolist())
                    all_targets[k].extend(y[:, i].cpu().tolist())

    # ── Calibrate ─────────────────────────────────────────────────────────────
    cal = ConformalPredictor.calibrate(dict(all_preds), dict(all_targets))
    cal.save(str(CKPT_DIR / "conformal_quantiles.pt"))


if __name__ == "__main__":
    main()
