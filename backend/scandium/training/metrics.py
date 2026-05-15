"""Evaluation metrics for material property prediction."""

import numpy as np
import torch
from torch import Tensor


def mae(y_pred: Tensor, y_true: Tensor) -> float:
    return float(torch.abs(y_pred - y_true).mean())


def rmse(y_pred: Tensor, y_true: Tensor) -> float:
    return float(torch.sqrt(((y_pred - y_true) ** 2).mean()))


def r2_score(y_pred: Tensor, y_true: Tensor) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / (ss_tot + 1e-8)


def physics_violation_rate(y_pred: Tensor, property_name: str) -> float:
    """
    Fraction of predictions that violate physics constraints.
    For well-trained PIGNet this should be 0.0.
    """
    if property_name in ("band_gap", "energy_above_hull", "bulk_modulus", "shear_modulus"):
        return float((y_pred < 0).float().mean())
    return 0.0


def compute_all_metrics(
    y_pred: Tensor, y_true: Tensor, property_name: str
) -> dict[str, float]:
    return {
        "mae":  mae(y_pred, y_true),
        "rmse": rmse(y_pred, y_true),
        "r2":   r2_score(y_pred, y_true),
        "physics_violation_rate": physics_violation_rate(y_pred, property_name),
    }