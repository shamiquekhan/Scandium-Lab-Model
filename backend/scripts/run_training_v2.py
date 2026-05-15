"""
Main training entry point for PIGNet V2 on real Materials Project data.

Usage (single seed):
    python scripts/run_training_v2.py

Usage (custom config):
    python scripts/run_training_v2.py --hidden 256 --n_conv 4 --epochs 300 --seed 42

Expected runtime on RTX 3090 (150k structures):
    ~20–24 hours for 300 epochs with early stopping.
Expected runtime on A100 (150k structures):
    ~9–11 hours.
"""

from __future__ import annotations
import argparse
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scandium.training.trainer_v2 import train_v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str,   default="data/processed_v2")
    parser.add_argument("--hidden",      type=int,   default=256)
    parser.add_argument("--n_conv",      type=int,   default=4)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--epochs",      type=int,   default=300)
    parser.add_argument("--batch",       type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--warmup",      type=int,   default=10)
    parser.add_argument("--accum",       type=int,   default=1)
    parser.add_argument("--ema_decay",   type=float, default=0.999)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--run_name",    type=str,   default="pignet_v2")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"[Train] Loading data from {data_dir}")
    train_data = torch.load(data_dir / "train.pt")
    val_data   = torch.load(data_dir / "val.pt")
    print(f"[Train] train={len(train_data):,}  val={len(val_data):,}")

    result = train_v2(
        train_data, val_data,
        hidden_dim    = args.hidden,
        n_conv        = args.n_conv,
        dropout       = args.dropout,
        epochs        = args.epochs,
        batch_size    = args.batch,
        lr            = args.lr,
        warmup_epochs = args.warmup,
        accum_steps   = args.accum,
        ema_decay     = args.ema_decay,
        seed          = args.seed,
        run_name      = args.run_name,
    )

    print(f"\nRun complete.")
    print(f"  Best checkpoint: {result['checkpoint_path']}")
    print(f"  Best val MAE:    {result['best_val_mae']:.4f}")


if __name__ == "__main__":
    main()
