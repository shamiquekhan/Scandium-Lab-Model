"""
OOD evaluation script: tests model generalization to unseen compositions,
and generates degradation curves.
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import DataLoader, Batch

from ml.models.cgcnn_baseline import CGCNNBaseline
from ml.data.ood_splits import (
    compositional_split,
    element_holdout_split,
    property_range_ood_split,
    density_ood_split,
)
from ml.training.train_baseline import MaterialsDataset
from ml.metrics.ece import torch_regression_ece


def rmse(pred, true, mask=None):
    if mask is not None:
        pred = pred[mask]
        true = true[mask]
    if len(pred) == 0:
        return float("nan")
    return float(torch.sqrt(((pred - true) ** 2).mean()).detach().cpu().item())


def mae(pred, true, mask=None):
    if mask is not None:
        pred = pred[mask]
        true = true[mask]
    if len(pred) == 0:
        return float("nan")
    return float((pred - true).abs().mean().detach().cpu().item())


def evaluate_split(
    model,
    dataset,
    device="cuda",
    batch_size=32,
    property_idx=0,
):
    """
    Evaluate model on a dataset and return per-property metrics.

    property_idx: which property to report (0=band_gap, 1=formation_energy, etc.)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_preds = []
    all_true = []
    all_std = []

    with torch.no_grad():
        for batch_list in loader:
            batch = Batch.from_data_list(batch_list).to(device)
            preds, log_vars = model(batch)
            std = torch.exp(0.5 * log_vars)

            all_preds.append(preds.cpu())
            all_true.append(batch.y.cpu())
            all_std.append(std.cpu())

    preds_all = torch.cat(all_preds, dim=0)  # (N, P)
    true_all = torch.cat(all_true, dim=0)
    std_all = torch.cat(all_std, dim=0)

    prop_pred = preds_all[:, property_idx]
    prop_true = true_all[:, property_idx]
    prop_std = std_all[:, property_idx]

    mask = ~torch.isnan(prop_true)
    rmse_val = rmse(prop_pred, prop_true, mask=mask)
    mae_val = mae(prop_pred, prop_true, mask=mask)

    errors = (prop_pred[mask] - prop_true[mask]).abs()
    ece_val = torch_regression_ece(prop_std[mask], errors)

    return {
        "rmse": rmse_val,
        "mae": mae_val,
        "ece": ece_val,
        "n_samples": int(mask.sum()),
    }


def ood_benchmark(
    model_path,
    data_path,
    device="cuda",
    output_dir="evaluations",
):
    """
    Full OOD benchmark: run multiple split strategies and report degradation.
    """
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    Path(output_dir).mkdir(exist_ok=True)

    with open(data_path) as f:
        raw = json.load(f)

    records = [r for r in raw if r.get("band_gap") is not None and r.get("formation_energy_per_atom") is not None]

    # Load model
    model = CGCNNBaseline().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    results = {}

    # Split 1: Compositional OOD
    print("Running compositional OOD split...")
    train_comp, ood_comp, meta_comp = compositional_split(records, ood_frac=0.1)
    ds_train_comp = MaterialsDataset(train_comp)
    ds_ood_comp = MaterialsDataset(ood_comp)

    metrics_train_comp = evaluate_split(model, ds_train_comp, device=device)
    metrics_ood_comp = evaluate_split(model, ds_ood_comp, device=device)

    results["compositional_ood"] = {
        "metadata": meta_comp,
        "train_metrics": metrics_train_comp,
        "ood_metrics": metrics_ood_comp,
        "rmse_degradation": metrics_ood_comp["rmse"] - metrics_train_comp["rmse"],
    }

    # Split 2: Element holdout (lanthanides)
    print("Running element holdout split...")
    train_elem, ood_elem, meta_elem = element_holdout_split(records, holdout_elements=["La", "Ce", "Pr"])
    ds_train_elem = MaterialsDataset(train_elem)
    ds_ood_elem = MaterialsDataset(ood_elem)

    metrics_train_elem = evaluate_split(model, ds_train_elem, device=device)
    metrics_ood_elem = evaluate_split(model, ds_ood_elem, device=device)

    results["element_holdout"] = {
        "metadata": meta_elem,
        "train_metrics": metrics_train_elem,
        "ood_metrics": metrics_ood_elem,
        "rmse_degradation": metrics_ood_elem["rmse"] - metrics_train_elem["rmse"],
    }

    # Split 3: Property extrema (high band gap)
    print("Running high band-gap property extrema split...")
    train_prop, ood_prop, meta_prop = property_range_ood_split(records, property_name="band_gap", upper_percentile=90)
    ds_train_prop = MaterialsDataset(train_prop)
    ds_ood_prop = MaterialsDataset(ood_prop)

    metrics_train_prop = evaluate_split(model, ds_train_prop, device=device)
    metrics_ood_prop = evaluate_split(model, ds_ood_prop, device=device)

    results["property_extrema"] = {
        "metadata": meta_prop,
        "train_metrics": metrics_train_prop,
        "ood_metrics": metrics_ood_prop,
        "rmse_degradation": metrics_ood_prop["rmse"] - metrics_train_prop["rmse"],
    }

    # Split 4: Density anomalies
    print("Running density anomaly split...")
    train_dens, ood_dens, meta_dens = density_ood_split(records, ood_frac=0.1)
    ds_train_dens = MaterialsDataset(train_dens)
    ds_ood_dens = MaterialsDataset(ood_dens)

    metrics_train_dens = evaluate_split(model, ds_train_dens, device=device)
    metrics_ood_dens = evaluate_split(model, ds_ood_dens, device=device)

    results["density_anomaly"] = {
        "metadata": meta_dens,
        "train_metrics": metrics_train_dens,
        "ood_metrics": metrics_ood_dens,
        "rmse_degradation": metrics_ood_dens["rmse"] - metrics_train_dens["rmse"],
    }

    # Save results
    output_file = Path(output_dir) / "ood_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nOOD Benchmark Results:")
    print("=" * 60)
    for split_name, split_results in results.items():
        print(f"\n{split_name}:")
        print(f"  Train RMSE: {split_results['train_metrics']['rmse']:.4f}")
        print(f"  OOD RMSE:   {split_results['ood_metrics']['rmse']:.4f}")
        print(f"  Degradation: {split_results['rmse_degradation']:.4f} eV")
    print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python eval_ood.py <model_checkpoint> <data_path> [output_dir]")
        sys.exit(1)

    model_path = sys.argv[1]
    data_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "evaluations"

    ood_benchmark(model_path, data_path, output_dir=output_dir)
