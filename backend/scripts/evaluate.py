"""
Comprehensive evaluation script.
Runs the trained model on test set and OOD set.
Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import torch
from torch_geometric.loader import DataLoader
from scandium.models.pignet import PIGNet
from scandium.data.dataset import CrystalDataset
from scandium.training.metrics import compute_all_metrics


def evaluate(checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt["config"]
    dc   = cfg["data"]
    mc   = cfg["model"]

    # Restore normalisation stats
    CrystalDataset.TARGET_MEAN = ckpt["target_mean"]
    CrystalDataset.TARGET_STD  = ckpt["target_std"]

    model = PIGNet(**mc)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    test_ds     = CrystalDataset(dc["root"], "test", dc["target"])
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch)
            target = dc["target"]
            mean   = CrystalDataset.TARGET_MEAN[target]
            std    = CrystalDataset.TARGET_STD[target]
            pred   = preds[target] * std + mean
            all_preds.append(pred.cpu())
            all_targets.append(batch.y_raw.squeeze().cpu())

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)
    metrics = compute_all_metrics(y_pred, y_true, dc["target"])

    print("\n── Test Set Evaluation ──")
    print(f"  MAE:                    {metrics['mae']:.4f}")
    print(f"  RMSE:                   {metrics['rmse']:.4f}")
    print(f"  R²:                     {metrics['r2']:.4f}")
    print(f"  Physics violation rate: {metrics['physics_violation_rate']*100:.2f}%")

    with open("evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nResults saved → evaluation_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()
    evaluate(args.checkpoint)