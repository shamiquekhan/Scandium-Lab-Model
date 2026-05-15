import json, os
from pathlib import Path
from torch_geometric.data import Dataset
from pymatgen.core import Structure
from tqdm import tqdm
from .featurize import structure_to_graph

class CrystalDataset(Dataset):
    """PyG Dataset wrapping raw JSON structures."""

    def __init__(self, raw_dir: str, cutoff=6.0, max_neighbors=12):
        self.data_dir = Path(raw_dir)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self._files = sorted(self.data_dir.glob("*.json"))
        print(f"CrystalDataset: found {len(self._files)} structures")
        super().__init__()

    def len(self): return len(self._files)

    def get(self, idx):
        with open(self._files[idx]) as f:
            record = json.load(f)
        structure = Structure.from_dict(record["structure"])
        label = {
            "band_gap": record["band_gap"],
            "formation_energy_per_atom": record["formation_energy_per_atom"],
            "energy_above_hull": record["energy_above_hull"],
        }
        data = structure_to_graph(structure, self.cutoff, self.max_neighbors, label)
        data.material_id = record["material_id"]
        return data

def split_dataset(dataset, train_frac=0.8, val_frac=0.1):
    """Split into train / val / test with fixed seed."""
    import torch
    n = len(dataset)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return (
        [dataset[i] for i in perm[:n_train]],
        [dataset[i] for i in perm[n_train:n_train+n_val]],
        [dataset[i] for i in perm[n_train+n_val:]],
    )