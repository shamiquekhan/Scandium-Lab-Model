"""Run: python train.py --epochs 100 --batch 64 --n_conv 4 --hidden 256"""
import argparse
from scandium.data.dataset import CrystalDataset, split_dataset
from scandium.training.trainer import train

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--n_conv",  type=int,   default=3)
    p.add_argument("--hidden",  type=int,   default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--data_path", default="backend/data/raw")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--scheduler", type=str, default="cosine")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    dataset = CrystalDataset(args.data_path, cutoff=8.0, max_neighbors=16)
    train_data, val_data, test_data = split_dataset(dataset)
    train(
        train_data, val_data, args.hidden, args.n_conv, args.dropout,
        args.epochs, args.batch_size, lr=args.lr, accum_steps=args.accumulation_steps,
        scheduler_type=args.scheduler, resume=args.resume
    )
