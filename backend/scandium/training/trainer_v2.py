"""
Production-grade trainer for PIGNet V2.
Supports AMP, EMA, Warmup + Cosine LR, and Physics-constrained losses.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..models.pignet_v2 import PIGNetV2
from ..physics.normalizer import Normalizer
from ..physics.constraints import physics_constrained_loss, PhysicsConfig
from .ema import EMA

CKPT_DIR = Path("backend/checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
    """
    Linear warmup (1% -> 100%) for 'warmup_epochs',
    followed by Cosine Annealing decay down to 1% of base_lr.
    """
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=base_lr * 0.01)

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=base_lr * 0.01)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def train_v2(
    train_data: list,
    val_data:   list,
    hidden_dim: int   = 256,
    n_conv:     int   = 4,
    dropout:    float = 0.1,
    epochs:     int   = 300,
    batch_size: int   = 64,
    lr:         float = 3e-4,
    warmup_epochs: int = 10,
    accum_steps:   int = 1,
    ema_decay:     float = 0.999,
    seed:          int   = 42,
    run_name:      str   = "pignet_v2",
    device:        str | None = None,
) -> dict:
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device={device} | AMP={torch.cuda.is_available()} | Accum={accum_steps}")

    # ── 1. Scale targets ──────────────────────────────────────────────────────
    # Band gap and formation energy have different ranges; normalization
    # prevents one property from dominating the gradient.
    normalizer = Normalizer.fit(train_data)
    print(f"[Train] Normalizer: mean={normalizer.mean.tolist()} std={normalizer.std.tolist()}")

    # ── 2. Model & Optimization ───────────────────────────────────────────────
    model = PIGNetV2(hidden_dim=hidden_dim, n_conv=n_conv, dropout=dropout).to(device)
    ema   = EMA(model, decay=ema_decay)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = make_scheduler(optimizer, warmup_epochs, epochs, lr)
    scaler    = GradScaler(enabled=(device == "cuda"))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=0)

    physics_cfg = PhysicsConfig()
    best_mae    = float("inf")
    history     = []

    # ── 3. Training Loop ──────────────────────────────────────────────────────
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Ep {epoch}/{epochs}", leave=False)):
            batch = batch.to(device)
            y_true = batch.y.squeeze(1) if batch.y.dim() > 1 else batch.y

            # Forward pass with AMP
            with autocast(enabled=(device == "cuda")):
                preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                # Compute loss in normalized domain
                y_norm = normalizer.norm(y_true)
                losses = physics_constrained_loss(preds, y_norm, physics_cfg, normalizer)
                loss   = losses["total"] / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update()

            train_loss += losses["total"].item()

        scheduler.step()

        # ── 4. Validation (using EMA weights) ─────────────────────────────────
        ema.apply()
        model.eval()
        val_loss = 0.0
        val_bg_mae = 0.0
        val_fe_mae = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch  = batch.to(device)
                y_true = batch.y.squeeze(1) if batch.y.dim() > 1 else batch.y

                with autocast(enabled=(device == "cuda")):
                    preds  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    y_norm = normalizer.norm(y_true)
                    v_losses = physics_constrained_loss(preds, y_norm, physics_cfg, normalizer)

                val_loss += v_losses["total"].item()

                # Calculate MAE in physical units (eV)
                phys_preds = normalizer.denorm_dict(preds)
                val_bg_mae += (phys_preds["band_gap"] - y_true[:, 0]).abs().mean().item()
                val_fe_mae += (phys_preds["formation_energy"] - y_true[:, 1]).abs().mean().item()

        ema.restore()

        # Average metrics
        n_tr, n_vl = len(train_loader), len(val_loader)
        metrics = {
            "epoch":      epoch,
            "train_loss": train_loss / n_tr,
            "val_loss":   val_loss / n_vl,
            "val_bg_mae": val_bg_mae / n_vl,
            "val_fe_mae": val_fe_mae / n_vl,
            "lr":         scheduler.get_last_lr()[0],
        }
        history.append(metrics)

        print(f"Ep {epoch:03d} | Train: {metrics['train_loss']:.4f} | Val: {metrics['val_loss']:.4f} | "
              f"BG-MAE: {metrics['val_bg_mae']:.3f} eV | FE-MAE: {metrics['val_fe_mae']:.3f} eV/at")

        # ── 5. Checkpointing ──────────────────────────────────────────────────
        current_mae = metrics["val_bg_mae"]
        if current_mae < best_mae:
            best_mae = current_mae
            ckpt_path = CKPT_DIR / f"best_{run_name}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "ema_state_dict":   ema.shadow,
                "optimizer_state":  optimizer.state_dict(),
                "normalizer":       normalizer.state_dict(),
                "config": {
                    "hidden_dim": hidden_dim,
                    "n_conv":     n_conv,
                    "dropout":    dropout,
                },
                "metrics": metrics,
            }, ckpt_path)
            print(f"  [Checkpoint] Saved best model to {ckpt_path}")

    # Save final history
    with open(CKPT_DIR / f"history_{run_name}.json", "w") as f:
        json.dump(history, f, indent=2)

    total_time = (time.time() - start_time) / 60
    print(f"\n[Train] Finished in {total_time:.1f} minutes. Best BG-MAE: {best_mae:.4f} eV")

    return {
        "best_val_mae":    best_mae,
        "checkpoint_path": str(CKPT_DIR / f"best_{run_name}.pt"),
        "history":         history,
    }
