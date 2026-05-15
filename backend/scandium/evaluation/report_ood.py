"""
Report generator for OOD evaluation results.
"""

import json
from pathlib import Path
from typing import Dict


def generate_ood_report(results_json_path, output_md_path=None):
    """
    Read OOD benchmark JSON and generate a markdown report.
    """
    with open(results_json_path) as f:
        results = json.load(f)

    if output_md_path is None:
        output_md_path = Path(results_json_path).parent / "ood_report.md"

    lines = [
        "# OOD (Out-of-Distribution) Benchmark Report\n",
        "## Overview\n",
        "This report summarizes model performance on held-out test sets across multiple OOD strategies.\n",
        "**Lower RMSE is better. Negative degradation indicates improved OOD performance (unlikely but noteworthy).**\n\n",
    ]

    # Summary table
    lines.append("## Summary Table\n\n")
    lines.append("| Split Strategy | Train RMSE | OOD RMSE | Degradation | Samples |\n")
    lines.append("|---|---|---|---|---|\n")

    for split_name, split_data in results.items():
        train_rmse = split_data["train_metrics"]["rmse"]
        ood_rmse = split_data["ood_metrics"]["rmse"]
        degradation = split_data["rmse_degradation"]
        n_samples = split_data["ood_metrics"]["n_samples"]
        lines.append(f"| {split_name} | {train_rmse:.4f} | {ood_rmse:.4f} | {degradation:+.4f} | {n_samples} |\n")

    lines.append("\n")

    # Detailed results per split
    lines.append("## Detailed Results\n\n")

    for split_name, split_data in results.items():
        lines.append(f"### {split_name}\n\n")

        meta = split_data["metadata"]
        lines.append("**Metadata:**\n")
        for k, v in meta.items():
            if k != "ood_composition_keys":  # skip verbose key lists
                lines.append(f"- {k}: {v}\n")
        lines.append("\n")

        train_m = split_data["train_metrics"]
        ood_m = split_data["ood_metrics"]

        lines.append("**Metrics:**\n")
        lines.append(f"- Train RMSE: {train_m['rmse']:.4f} eV\n")
        lines.append(f"- OOD RMSE: {ood_m['rmse']:.4f} eV\n")
        lines.append(f"- Train MAE: {train_m['mae']:.4f} eV\n")
        lines.append(f"- OOD MAE: {ood_m['mae']:.4f} eV\n")
        lines.append(f"- Train ECE: {train_m['ece']:.4f}\n")
        lines.append(f"- OOD ECE: {ood_m['ece']:.4f}\n")
        lines.append(f"- RMSE Degradation: {split_data['rmse_degradation']:+.4f} eV\n")
        lines.append(f"- Samples: {ood_m['n_samples']}\n")
        lines.append("\n")

    lines.append("## Interpretation\n\n")
    lines.append("- **RMSE Degradation**: positive = model performs worse on OOD data (expected). ")
    lines.append("Larger degradation suggests the model's learned features are sensitive to compositional/structural novelty.\n")
    lines.append("- **ECE (Expected Calibration Error)**: measure of uncertainty calibration. ")
    lines.append("Lower is better. If ECE increases on OOD, the model is less calibrated (overconfident) on novel data.\n")
    lines.append("- **Physics-constrained models should show graceful degradation**: ")
    lines.append("RMSE increase < 20% and maintained ECE indicate the model generalizes well to unseen compositions.\n")

    report = "".join(lines)
    with open(output_md_path, "w") as f:
        f.write(report)

    print(f"Report written to {output_md_path}")
    return report


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python report_ood.py <results_json>")
        sys.exit(1)

    generate_ood_report(sys.argv[1])
