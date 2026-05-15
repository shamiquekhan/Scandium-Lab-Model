import json, time, torch
from pathlib import Path
from datetime import datetime

REGISTRY = Path("backend/checkpoints/registry.json")

def save_versioned_checkpoint(
    model, ema, normalizer, optimizer,
    epoch: int, val_metrics: dict,
    training_config: dict, data_config: dict,
    tag: str = None,
):
    """Save checkpoint with full provenance metadata."""
    version = datetime.now().strftime("%Y%m%d_%H%M")
    if tag: version += f"_{tag}"
    fname = Path(f"backend/checkpoints/pignet_{version}.pt")

    metadata = {
        "version":         version,
        "saved_at":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epoch":           epoch,
        "val_metrics":     val_metrics,
        "training_config": training_config,
        "data_config":     data_config,
        "torch_version":   torch.__version__,
    }

    torch.save({
        "metadata":       metadata,
        "model_state":    model.state_dict(),
        "ema_shadow":     ema.shadow if ema else None,
        "normalizer":     normalizer.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config":         training_config,
    }, fname)

    # Update human-readable registry
    registry = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []
    registry.append({"file": str(fname), **metadata})
    REGISTRY.write_text(json.dumps(registry, indent=2))

    print(f"Saved versioned checkpoint → {fname}")
    return fname
