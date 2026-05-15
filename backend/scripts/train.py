"""
Training entry point.
Usage:
    python scripts/train.py --config configs/training_config.yaml
"""

import argparse
import os
import random
import numpy as np
import torch
import yaml
from dotenv import load_dotenv

load_dotenv()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    from scandium.training.trainer import Trainer
    trainer = Trainer(cfg)
    trainer.train()