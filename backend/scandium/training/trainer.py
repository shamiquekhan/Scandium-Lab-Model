import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from pathlib import Path
from tqdm import tqdm
import json
import wandb
from ..models.pignet import PIGNet
from ..physics.constraints import physics_constrained_loss, PhysicsConfig

CKPT_DIR = Path("backend/checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

def train(
    train_data, val_data,
    hidden_dim=256, n_conv=4, dropout=0.1,
    epochs=100, batch_size=64, lr=3e-4, accum_steps=1,
    scheduler_type="cosine", resume=False, device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} | {len(train_data)} train / {len(val_data)} val | accum_steps={accum_steps}")

    model = PIGNet(hidden_dim=hidden_dim, n_conv=n_conv, dropout=dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    start_epoch = 1
    best_val = float("inf")
    if resume and (CKPT_DIR / "best_model.pt").exists():
        print(f"Resuming from {CKPT_DIR / 'best_model.pt'}...")
        ckpt = torch.load(CKPT_DIR / "best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("val_loss", float("inf"))
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    else:
        scheduler = None
        
    wandb.init(project="scandium-labs-pignet", config={
        "hidden_dim": hidden_dim, "n_conv": n_conv, "dropout": dropout,
        "batch_size": batch_size, "lr": lr, "accum_steps": accum_steps,
        "scheduler": scheduler_type, "epochs": epochs
    })
    cfg = PhysicsConfig()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False)

    history = []

    for epoch in range(start_epoch, epochs + 1):
        # ── Train ───────────────────────────────────────────────────────────
        model.train()
        t_loss, t_bg_mae, t_fe_mae = 0.0, 0.0, 0.0
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)):
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            targets = batch.y.squeeze(1)  # (B, 3)
            losses = physics_constrained_loss(preds, targets, cfg)
            
            # Scale loss for accumulation
            loss = losses["total"] / accum_steps
            loss.backward()
            
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            t_loss  += losses["total"].item()
            t_bg_mae += (preds["band_gap"] - targets[:, 0]).abs().mean().item()
            t_fe_mae += (preds["formation_energy"] - targets[:, 1]).abs().mean().item()
        if scheduler:
            scheduler.step()

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        v_loss, v_bg_mae, v_fe_mae = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                targets = batch.y.squeeze(1)
                losses = physics_constrained_loss(preds, targets, cfg)
                v_loss  += losses["total"].item()
                v_bg_mae += (preds["band_gap"] - targets[:, 0]).abs().mean().item()
                v_fe_mae += (preds["formation_energy"] - targets[:, 1]).abs().mean().item()

        n_tr, n_vl = len(train_loader), len(val_loader)
        epoch_stats = {
            "epoch":      epoch,
            "train_loss": t_loss / n_tr,
            "val_loss":   v_loss / n_vl,
            "val_bg_mae": v_bg_mae / n_vl,
            "val_fe_mae": v_fe_mae / n_vl,
            "lr":         scheduler.get_last_lr()[0] if scheduler else lr,
        }
        history.append(epoch_stats)
        wandb.log(epoch_stats)
        print(f"Ep{epoch:03d} | train {epoch_stats['train_loss']:.4f} | val {epoch_stats['val_loss']:.4f} | bg_mae {epoch_stats['val_bg_mae']:.3f} eV | fe_mae {epoch_stats['val_fe_mae']:.3f} eV/atom")

        if epoch_stats["val_loss"] < best_val:
            best_val = epoch_stats["val_loss"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val,
                "config": {"hidden_dim": hidden_dim, "n_conv": n_conv, "dropout": dropout},
            }, CKPT_DIR / "best_model.pt")
            print(f"  > Saved best model (val_loss={best_val:.4f})")

    with open(CKPT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Training complete.")
    wandb.finish()
    return model, history