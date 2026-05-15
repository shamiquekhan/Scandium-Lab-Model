"""
Physics-constrained loss function and violation checker.
The loss adds penalty terms for physically impossible predictions.
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class PhysicsConfig:
    # Loss weights — tune these during training
    w_band_gap:          float = 0.5   # weight on band_gap MSE
    w_formation_energy:  float = 2.0   # weight on formation energy MSE
    w_hull_energy:       float = 0.5   # weight on hull energy MSE
    lambda_bg_penalty:   float = 5.0   # penalty for band_gap < 0 in raw logit
    lambda_hull_penalty: float = 5.0   # penalty for hull energy < 0
    lambda_fe_reg:       float = 0.01  # L2 regularization on formation energy magnitude
    # Violation thresholds
    uncertainty_warn:    float = 0.5   # flag if std > this * mean
    hull_instability:    float = 0.3   # eV/atom above hull → "unstable" warning


def physics_constrained_loss(
    predictions: dict,
    targets: torch.Tensor,
    config: PhysicsConfig = None,
    normalizer = None,
) -> dict[str, torch.Tensor]:
    """
    Compute total training loss with physics penalty terms.

    Args:
        predictions: dict from model.forward() — band_gap, formation_energy, energy_above_hull
        targets:     Tensor (B, 3) — [band_gap, formation_energy, hull_energy]
        config:      PhysicsConfig
        normalizer:  Normalizer instance to denormalize predictions for physics penalties

    Returns:
        dict with 'total', 'mse_bg', 'mse_fe', 'mse_hull', 'physics_penalty'
    """
    cfg = config or PhysicsConfig()

    bg_pred  = predictions["band_gap"]
    fe_pred  = predictions["formation_energy"]
    hul_pred = predictions["energy_above_hull"]

    bg_tgt   = targets[:, 0]
    fe_tgt   = targets[:, 1]
    hul_tgt  = targets[:, 2]

    # ── Regression losses (Huber = robust to outliers) ─────────────────────────
    mse_bg   = F.huber_loss(bg_pred,  bg_tgt,  delta=1.0)
    mse_fe   = F.huber_loss(fe_pred,  fe_tgt,  delta=1.0)
    mse_hull = F.huber_loss(hul_pred, hul_tgt, delta=1.0)

    # ── Physics penalty terms ─────────────────────────────────────────────────
    if normalizer is not None:
        phys_preds = normalizer.denorm_dict(predictions)
        bg_phys = phys_preds["band_gap"]
        fe_phys = phys_preds["formation_energy"]
        hul_phys = phys_preds["energy_above_hull"]
    else:
        bg_phys, fe_phys, hul_phys = bg_pred, fe_pred, hul_pred

    bg_neg_penalty  = F.relu(-bg_phys).mean()
    hul_neg_penalty = F.relu(-hul_phys).mean()
    fe_l2_reg       = (fe_phys ** 2).mean()

    physics_penalty = (
        cfg.lambda_bg_penalty   * bg_neg_penalty  +
        cfg.lambda_hull_penalty * hul_neg_penalty +
        cfg.lambda_fe_reg       * fe_l2_reg
    )

    total = (
        cfg.w_band_gap         * mse_bg   +
        cfg.w_formation_energy * mse_fe   +
        cfg.w_hull_energy      * mse_hull +
        physics_penalty
    )

    return {
        "total":           total,
        "mse_bg":          mse_bg,
        "mse_fe":          mse_fe,
        "mse_hull":        mse_hull,
        "physics_penalty": physics_penalty,
    }


def check_violations(prediction: dict, uncertainty: dict = None) -> list[str]:
    """
    Check a single prediction for physics violations.
    Returns a list of violation strings (empty = no violations).
    """
    cfg = PhysicsConfig()
    violations = []

    bg  = prediction.get("band_gap", 0.0)
    fe  = prediction.get("formation_energy", 0.0)
    hul = prediction.get("energy_above_hull", 0.0)

    if bg < 0:
        violations.append(f"BAND_GAP_NEGATIVE: {bg:.4f} eV (must be ≥ 0)")

    if hul < 0:
        violations.append(f"HULL_ENERGY_NEGATIVE: {hul:.4f} eV/atom")

    if hul > cfg.hull_instability:
        violations.append(f"STRUCTURE_UNSTABLE: {hul:.4f} eV/atom above hull (threshold {cfg.hull_instability})")

    if fe > 2.0:
        violations.append(f"FORMATION_ENERGY_HIGH: {fe:.4f} eV/atom (unusual for stable material)")

    if uncertainty:
        bg_std  = uncertainty.get("band_gap_std", 0.0)
        fe_std  = uncertainty.get("formation_energy_std", 0.0)
        if bg  > 0 and bg_std / bg > cfg.uncertainty_warn:
            violations.append(f"HIGH_UNCERTAINTY_BG: std/mean = {bg_std/bg:.2f} (model unsure)")
        if abs(fe) > 0.1 and fe_std / abs(fe) > cfg.uncertainty_warn:
            violations.append(f"HIGH_UNCERTAINTY_FE: std/|mean| = {fe_std/abs(fe):.2f}")

    return violations