"""
Train 5 ensemble members sequentially.
Each member: same architecture, same data, different random seed.
Total runtime: 5× single training time (~45–120h on RTX 3090 for 150k data).
Run on a GPU server overnight for all 5 seeds.

Usage:
    python scripts/train_ensemble.py --data_dir data/processed_v2
"""

from __future__ import annotations
import argparse
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scandium.training.trainer_v2 import train_v2

ENSEMBLE_SEEDS = [42, 123, 456, 789, 1337]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed_v2")
    parser.add_argument("--epochs",   type=int, default=300)
    parser.add_argument("--batch",    type=int, default=64)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    train_data = torch.load(data_dir / "train.pt")
    val_data   = torch.load(data_dir / "val.pt")

    results = []

    for seed in ENSEMBLE_SEEDS:
        print(f"\n{'#'*60}")
        print(f"# ENSEMBLE MEMBER — seed={seed}")
        print(f"{'#'*60}")

        result = train_v2(
            train_data, val_data,
            seed     = seed,
            epochs   = args.epochs,
            batch_size = args.batch,
            run_name = f"ensemble_seed_{seed}",
        )
        results.append(result)
        print(f"Seed {seed} done. Val MAE: {result['best_val_mae']:.4f}")

    print(f"\n{'='*60}")
    print("ENSEMBLE TRAINING COMPLETE")
    for seed, r in zip(ENSEMBLE_SEEDS, results):
        print(f"  seed={seed}  val_mae={r['best_val_mae']:.4f}  ckpt={r['checkpoint_path']}")
    avg = sum(r["best_val_mae"] for r in results) / len(results)
    print(f"  Average val MAE: {avg:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
