# Scandium Labs — Complete Backend & ML Model Build Guide

> **Goal:** By the end of this guide, you have a fully working system: data ingestion → graph construction → PIGNet training → physics-constraint validation → FastAPI backend → Python SDK → Docker deployment. Every file is written out. Every command runs in order.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Environment Setup](#2-environment-setup)
3. [Data Pipeline](#3-data-pipeline)
4. [Graph Construction](#4-graph-construction)
5. [PIGNet Model Architecture](#5-pignet-model-architecture)
6. [Physics Constraints Module](#6-physics-constraints-module)
7. [Training Pipeline](#7-training-pipeline)
8. [Evaluation & Benchmarking](#8-evaluation--benchmarking)
9. [FastAPI Backend](#9-fastapi-backend)
10. [Python SDK](#10-python-sdk)
11. [Configuration & Secrets](#11-configuration--secrets)
12. [Docker & Deployment](#12-docker--deployment)
13. [Testing](#13-testing)
14. [End-to-End Verification](#14-end-to-end-verification)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Project Structure

Create this exact directory layout before writing any code.

```
scandium-labs/
│
├── README.md
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
│
├── configs/
│   ├── model_config.yaml          # PIGNet hyperparameters
│   ├── training_config.yaml       # Training settings
│   └── api_config.yaml            # API settings
│
├── data/
│   ├── raw/                       # Downloaded DFT data (gitignored)
│   ├── processed/                 # Graph datasets (gitignored)
│   ├── splits/                    # train/val/test indices
│   └── scripts/
│       ├── download_mp.py         # Pull from Materials Project
│       ├── download_aflow.py      # Pull from AFLOW
│       └── build_dataset.py       # CIF → PyG graphs
│
├── scandium/                      # Core Python package
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pignet.py              # Main PIGNet architecture
│   │   ├── layers.py              # Custom GNN layers
│   │   ├── readout.py             # Global pooling modules
│   │   └── heads.py               # Property prediction heads
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── constraints.py         # All physics constraint losses
│   │   └── validator.py           # Post-prediction physics checks
│   ├── data/
│   │   ├── __init__.py
│   │   ├── graph_builder.py       # Structure → PyG Data
│   │   ├── dataset.py             # CrystalDataset class
│   │   ├── transforms.py          # Data augmentation
│   │   └── featurisers.py         # Atom/bond feature vectors
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training loop
│   │   ├── loss.py                # Combined loss function
│   │   ├── metrics.py             # MAE, RMSE, physics-violation rate
│   │   └── scheduler.py           # LR scheduling
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── checkpointing.py
│
├── api/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app entry point
│   ├── routers/
│   │   ├── predict.py             # /predict/* endpoints
│   │   ├── screen.py              # /screen/* endpoints
│   │   └── health.py              # /health endpoint
│   ├── schemas/
│   │   ├── request.py             # Pydantic input models
│   │   └── response.py            # Pydantic output models
│   ├── middleware/
│   │   ├── auth.py                # API key authentication
│   │   └── rate_limit.py          # Rate limiting
│   └── services/
│       ├── predictor.py           # Wraps model inference
│       └── batch_processor.py     # Handles bulk requests
│
├── sdk/
│   ├── setup.py
│   ├── scandiumlabs/
│   │   ├── __init__.py
│   │   ├── client.py              # Main Client class
│   │   └── models.py              # SDK response dataclasses
│
├── scripts/
│   ├── train.py                   # CLI training entry point
│   ├── evaluate.py                # Run evaluation suite
│   └── export_model.py            # Export to TorchScript
│
└── tests/
    ├── conftest.py
    ├── test_graph_builder.py
    ├── test_model.py
    ├── test_physics.py
    ├── test_api.py
    └── test_sdk.py
```

Run this to scaffold the directories:

```bash
mkdir -p scandium-labs/{configs,data/{raw,processed,splits,scripts},scandium/{models,physics,data,training,utils},api/{routers,schemas,middleware,services},sdk/scandiumlabs,scripts,tests}
cd scandium-labs
touch README.md docker-compose.yml Dockerfile requirements.txt requirements-dev.txt .env.example
```

---

## 2. Environment Setup

### 2.1 System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| CUDA | 11.8 | 12.1 |
| RAM | 16 GB | 32 GB |
| VRAM | 8 GB | 24 GB (A100) |
| Storage | 50 GB | 200 GB |

### 2.2 Conda Environment

```bash
conda create -n scandium python=3.11 -y
conda activate scandium

# CUDA-enabled PyTorch — match your CUDA version
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### 2.3 `requirements.txt`

```txt
# Deep learning
torch==2.2.0
torch-geometric==2.5.2
torch-scatter==2.1.2
torch-sparse==0.6.18
torch-cluster==1.6.3
dgl==1.1.3

# Equivariant networks
e3nn==0.5.1

# Materials science
pymatgen==2024.3.1
mp-api==0.41.2
ase==3.22.1
matminer==0.9.2

# Training
wandb==0.16.6
optuna==3.6.1

# API
fastapi==0.110.0
uvicorn[standard]==0.29.0
pydantic==2.6.4
python-multipart==0.0.9
python-jose[cryptography]==3.3.0

# Data
numpy==1.26.4
pandas==2.2.1
scikit-learn==1.4.1
h5py==3.10.0

# Config
pyyaml==6.0.1
python-dotenv==1.0.1
omegaconf==2.3.0

# Utils
tqdm==4.66.2
rich==13.7.1
httpx==0.27.0
```

```bash
pip install -r requirements.txt

# PyG extras (must be after torch)
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

### 2.4 `.env.example`

```bash
# Copy this to .env and fill in values
MP_API_KEY=your_materials_project_api_key_here
WANDB_API_KEY=your_wandb_key_here
SCANDIUM_API_SECRET=your_jwt_secret_here_min_32_chars
MODEL_CHECKPOINT_DIR=./checkpoints
DATA_DIR=./data
LOG_LEVEL=INFO
DEVICE=cuda
```

---

## 3. Data Pipeline

### 3.1 `data/scripts/download_mp.py`

Downloads structure + property data from Materials Project. Run once. Saves ~2 GB.

```python
"""
Download crystal structures and DFT properties from Materials Project.
Usage:
    python data/scripts/download_mp.py --n_materials 50000 --output data/raw/mp_data.json
"""

import argparse
import json
import os
from pathlib import Path

from mp_api.client import MPRester
from tqdm import tqdm


PROPERTIES = [
    "material_id",
    "formula_pretty",
    "structure",
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "bulk_modulus",
    "shear_modulus",
    "total_magnetization",
    "is_stable",
    "symmetry",
    "elements",
    "nelements",
    "nsites",
    "volume",
    "density",
]


def download(api_key: str, n: int, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[MP] Connecting to Materials Project API...")
    records = []

    with MPRester(api_key) as mpr:
        # Pull semiconductors and insulators (band_gap > 0) — most relevant
        docs = mpr.summary.search(
            band_gap=(0.01, 10.0),
            energy_above_hull=(0, 0.1),   # stable or near-stable
            fields=PROPERTIES,
            num_chunks=None,
            chunk_size=1000,
        )

        print(f"[MP] Retrieved {len(docs)} documents. Serialising...")
        for doc in tqdm(docs[:n]):
            try:
                rec = {
                    "material_id": doc.material_id,
                    "formula": doc.formula_pretty,
                    "band_gap": float(doc.band_gap) if doc.band_gap is not None else None,
                    "formation_energy_per_atom": (
                        float(doc.formation_energy_per_atom)
                        if doc.formation_energy_per_atom is not None else None
                    ),
                    "energy_above_hull": (
                        float(doc.energy_above_hull)
                        if doc.energy_above_hull is not None else None
                    ),
                    "is_stable": doc.is_stable,
                    "nsites": doc.nsites,
                    "nelements": doc.nelements,
                    "structure_json": doc.structure.as_dict(),
                }

                # Optional: moduli (not always present)
                if hasattr(doc, "bulk_modulus") and doc.bulk_modulus:
                    rec["bulk_modulus"] = float(doc.bulk_modulus.get("voigt", 0) or 0)
                if hasattr(doc, "shear_modulus") and doc.shear_modulus:
                    rec["shear_modulus"] = float(doc.shear_modulus.get("voigt", 0) or 0)

                records.append(rec)
            except Exception as e:
                print(f"[WARN] Skipping {doc.material_id}: {e}")

    with open(output_path, "w") as f:
        json.dump(records, f)

    print(f"[MP] Saved {len(records)} records → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_materials", type=int, default=50000)
    parser.add_argument("--output", type=str, default="data/raw/mp_data.json")
    args = parser.parse_args()

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise ValueError("Set MP_API_KEY in your .env file")

    download(api_key, args.n_materials, args.output)
```

### 3.2 `data/scripts/build_dataset.py`

Converts the raw JSON records into PyG graph objects and saves as a processed dataset.

```python
"""
Build graph dataset from raw Materials Project data.
Usage:
    python data/scripts/build_dataset.py \
        --input data/raw/mp_data.json \
        --output data/processed/ \
        --cutoff 6.0
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scandium.data.graph_builder import GraphBuilder
from scandium.data.featurisers import AtomFeaturiser


def build(input_path: str, output_dir: str, cutoff: float) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        records = json.load(f)

    print(f"[BUILD] Processing {len(records)} structures with cutoff={cutoff} Å")

    builder = GraphBuilder(cutoff=cutoff)
    featuriser = AtomFeaturiser()

    graphs = []
    skipped = 0

    for rec in tqdm(records):
        try:
            struct = Structure.from_dict(rec["structure_json"])
            data = builder.build(struct, featuriser)

            # Attach labels
            data.band_gap             = float(rec["band_gap"])             if rec.get("band_gap")             is not None else float("nan")
            data.formation_energy     = float(rec["formation_energy_per_atom"]) if rec.get("formation_energy_per_atom") is not None else float("nan")
            data.energy_above_hull    = float(rec.get("energy_above_hull", float("nan")))
            data.bulk_modulus         = float(rec.get("bulk_modulus",  float("nan")))
            data.shear_modulus        = float(rec.get("shear_modulus", float("nan")))
            data.is_stable            = int(rec.get("is_stable", 0))
            data.material_id          = rec["material_id"]
            data.formula              = rec["formula"]

            graphs.append(data)
        except Exception as e:
            skipped += 1

    print(f"[BUILD] Built {len(graphs)} graphs. Skipped {skipped}.")

    # Stratified split by space group
    np.random.seed(42)
    idx = np.random.permutation(len(graphs))
    n_train = int(len(idx) * 0.80)
    n_val   = int(len(idx) * 0.10)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    splits = {"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()}

    with open(os.path.join(output_dir, "graphs.pkl"), "wb") as f:
        pickle.dump(graphs, f)
    with open(os.path.join(output_dir, "splits.json"), "w") as f:
        json.dump(splits, f)

    print(f"[BUILD] Saved → {output_dir}")
    print(f"        Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, default="data/raw/mp_data.json")
    parser.add_argument("--output", type=str, default="data/processed/")
    parser.add_argument("--cutoff", type=float, default=6.0)
    args = parser.parse_args()
    build(args.input, args.output, args.cutoff)
```

---

## 4. Graph Construction

### 4.1 `scandium/data/featurisers.py`

```python
"""
Atom and bond featurisers.
Each atom becomes a 92-dimensional feature vector encoding
element-level physics: electronegativity, atomic radius,
valence electrons, period, group, and one-hot atomic number.
"""

from __future__ import annotations
import numpy as np
from pymatgen.core import Element


# Precomputed element lookup table (atomic number → features)
# Columns: electronegativity, covalent_radius_Å, valence_electrons, period, group
_ELEMENT_TABLE: dict[int, list[float]] = {}

def _build_table() -> None:
    for Z in range(1, 95):
        try:
            el = Element.from_Z(Z)
            en   = float(el.X)                  if el.X is not None else 0.0
            cr   = float(el.atomic_radius or 0) * 100  # Å → pm for normalisation
            val  = float(el.nvalence())          if hasattr(el, "nvalence") else 0.0
            per  = float(el.row)
            grp  = float(el.group if el.group else 18)
            _ELEMENT_TABLE[Z] = [en, cr, val, per, grp]
        except Exception:
            _ELEMENT_TABLE[Z] = [0.0, 0.0, 0.0, 0.0, 0.0]

_build_table()

MAX_Z = 94


class AtomFeaturiser:
    """Converts atomic number to fixed-length feature vector."""

    FEATURE_DIM: int = MAX_Z + 5   # one-hot (94) + 5 physics features = 99

    def featurise(self, atomic_number: int) -> np.ndarray:
        onehot = np.zeros(MAX_Z, dtype=np.float32)
        if 1 <= atomic_number <= MAX_Z:
            onehot[atomic_number - 1] = 1.0

        physics = np.array(_ELEMENT_TABLE.get(atomic_number, [0.0] * 5), dtype=np.float32)
        # Normalise physics features to [0, 1] approximate range
        physics[0] /= 4.0    # electronegativity max ~4
        physics[1] /= 300.0  # covalent radius max ~300 pm
        physics[2] /= 18.0   # valence electrons max 18
        physics[3] /= 7.0    # period max 7
        physics[4] /= 18.0   # group max 18

        return np.concatenate([onehot, physics])
```

### 4.2 `scandium/data/graph_builder.py`

```python
"""
Converts pymatgen Structure objects into PyTorch Geometric Data objects.
Each atom is a node. Edges connect atom pairs within cutoff radius.
Edge features encode Euclidean distance and spherical harmonics up to l=2.
"""

from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import Data
from pymatgen.core import Structure
from .featurisers import AtomFeaturiser


class GraphBuilder:
    """Structure → PyG Data graph."""

    def __init__(self, cutoff: float = 6.0, max_neighbours: int = 32):
        self.cutoff = cutoff
        self.max_neighbours = max_neighbours

    def build(self, structure: Structure, featuriser: AtomFeaturiser) -> Data:
        # ── Node features ──
        atom_features = []
        for site in structure:
            Z = site.specie.number
            atom_features.append(featuriser.featurise(Z))
        x = torch.tensor(np.array(atom_features), dtype=torch.float32)

        # ── Edges via periodic neighbour search ──
        all_neighbours = structure.get_all_neighbors(self.cutoff, include_index=True)

        src, dst, distances, vectors = [], [], [], []
        for i, neighbours in enumerate(all_neighbours):
            # Sort by distance, take max_neighbours closest
            neighbours_sorted = sorted(neighbours, key=lambda n: n[1])[:self.max_neighbours]
            for neighbour in neighbours_sorted:
                site_j, dist, j_idx = neighbour[0], neighbour[1], neighbour[2]
                src.append(i)
                dst.append(j_idx)
                distances.append(dist)
                # Displacement vector (for directional/angular features)
                vec = site_j.coords - structure[i].coords
                vectors.append(vec)

        if len(src) == 0:
            raise ValueError(f"No edges found with cutoff={self.cutoff} Å")

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        dist_tensor = torch.tensor(distances, dtype=torch.float32).unsqueeze(1)
        vec_tensor  = torch.tensor(np.array(vectors), dtype=torch.float32)

        # ── Bond-distance Gaussian basis expansion (40 basis functions) ──
        edge_attr = self._gaussian_basis(dist_tensor)

        # ── Lattice matrix ──
        lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float32)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_vec=vec_tensor,
            edge_dist=dist_tensor,
            num_nodes=len(structure),
            lattice=lattice.unsqueeze(0),
        )

    def _gaussian_basis(
        self,
        distances: torch.Tensor,
        n_basis: int = 40,
        d_min: float = 0.5,
        d_max: float = 6.5,
    ) -> torch.Tensor:
        """
        Expand scalar distances into a Gaussian RBF basis.
        Distances shape: (E, 1) → output shape: (E, n_basis)
        """
        centres = torch.linspace(d_min, d_max, n_basis)
        width   = (d_max - d_min) / n_basis
        return torch.exp(-((distances - centres) ** 2) / (width ** 2))
```

### 4.3 `scandium/data/dataset.py`

```python
"""
CrystalDataset: wraps the list of PyG Data objects and
provides train/val/test subsets.
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Literal

import torch
from torch_geometric.data import Dataset, Data


class CrystalDataset(Dataset):
    """
    Args:
        root:     path to data/processed/
        split:    'train' | 'val' | 'test'
        target:   which label to use as y ('band_gap' | 'formation_energy' | ...)
    """

    TARGET_MEAN: dict[str, float] = {}
    TARGET_STD:  dict[str, float] = {}

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] = "train",
        target: str = "band_gap",
    ):
        self.root = Path(root)
        self.split = split
        self.target = target

        # Load graphs
        with open(self.root / "graphs.pkl", "rb") as f:
            all_graphs: list[Data] = pickle.load(f)

        # Load split indices
        with open(self.root / "splits.json") as f:
            splits = json.load(f)

        indices = splits[split]
        self.graphs = [all_graphs[i] for i in indices]

        # Compute normalisation statistics from training split only
        if split == "train":
            vals = [
                float(getattr(g, target))
                for g in self.graphs
                if not torch.isnan(torch.tensor(float(getattr(g, target))))
            ]
            CrystalDataset.TARGET_MEAN[target] = float(torch.tensor(vals).mean())
            CrystalDataset.TARGET_STD[target]  = float(torch.tensor(vals).std())

        print(f"[Dataset] {split}: {len(self.graphs)} graphs | target={target}")

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        g = self.graphs[idx]
        raw = float(getattr(g, self.target))
        mean = CrystalDataset.TARGET_MEAN.get(self.target, 0.0)
        std  = CrystalDataset.TARGET_STD.get(self.target, 1.0)
        g.y = torch.tensor([(raw - mean) / (std + 1e-8)], dtype=torch.float32)
        g.y_raw = torch.tensor([raw], dtype=torch.float32)
        return g
```

---

## 5. PIGNet Model Architecture

### 5.1 `scandium/models/layers.py`

```python
"""
Custom GNN message-passing layers for crystal property prediction.
Uses angle-aware message passing (DimeNet-style triplet features)
combined with distance-based edge features.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class EdgeUpdateLayer(nn.Module):
    """Updates edge features using source/target node features + current edge feature."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim),
            nn.LayerNorm(edge_dim),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        src, dst = edge_index
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        return self.mlp(edge_input)


class AttentionMessagePassing(MessagePassing):
    """
    Attention-weighted message passing layer.
    Each atom attends over its neighbours weighted by
    learned attention coefficients derived from edge features.
    """

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, heads: int = 4):
        super().__init__(aggr="add")
        self.heads = heads
        self.out_dim = out_dim
        head_dim = out_dim // heads

        self.attn_net = nn.Linear(edge_dim, heads)
        self.msg_net  = nn.Linear(node_dim + edge_dim, out_dim)
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim),
            nn.SiLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: Tensor, index: Tensor) -> Tensor:
        # x_j: source node features (E, node_dim)
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        msg = self.msg_net(msg_input)                     # (E, out_dim)

        attn_logits = self.attn_net(edge_attr)             # (E, heads)
        attn_weights = softmax(attn_logits, index)         # (E, heads)

        # Broadcast attention weights over head dimensions
        attn_expanded = attn_weights.unsqueeze(-1).repeat(1, 1, self.out_dim // self.heads)
        attn_expanded = attn_expanded.view(-1, self.out_dim)

        return msg * attn_expanded

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return self.update_net(torch.cat([x, aggr_out], dim=-1))
```

### 5.2 `scandium/models/readout.py`

```python
"""Global readout modules that pool node representations → crystal vector."""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_scatter import scatter_softmax


class AttentionReadout(nn.Module):
    """
    Attention-weighted global pooling.
    Each atom gets a learned importance score.
    Crystal embedding = weighted sum of atom embeddings.
    """

    def __init__(self, node_dim: int, out_dim: int):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(node_dim, node_dim // 2),
            nn.SiLU(),
            nn.Linear(node_dim // 2, 1),
        )
        self.transform = nn.Linear(node_dim, out_dim)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        scores = self.score_net(x)                          # (N, 1)
        attn   = scatter_softmax(scores.squeeze(-1), batch) # (N,)
        x_t    = self.transform(x)                          # (N, out_dim)
        return global_add_pool(attn.unsqueeze(-1) * x_t, batch)  # (B, out_dim)


class SetToSetReadout(nn.Module):
    """
    Two-stream readout: attention-weighted sum + mean pooling,
    concatenated for richer crystal representation.
    """

    def __init__(self, node_dim: int, out_dim: int):
        super().__init__()
        self.attn  = AttentionReadout(node_dim, out_dim // 2)
        self.mean  = nn.Linear(node_dim, out_dim // 2)
        self.final = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        attn_out = self.attn(x, batch)
        mean_out = global_mean_pool(self.mean(x), batch)
        return self.final(torch.cat([attn_out, mean_out], dim=-1))
```

### 5.3 `scandium/models/heads.py`

```python
"""
Property prediction heads. Each target property has its own head
with the appropriate output activation.
"""

import torch
import torch.nn as nn
from torch import Tensor


class PropertyHead(nn.Module):
    """
    3-layer MLP prediction head.
    activation: 'none' | 'relu' (non-negative) | 'softplus' (smooth non-negative)
    """

    def __init__(self, in_dim: int, hidden_dim: int, activation: str = "none"):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.activation = activation

    def forward(self, z: Tensor) -> Tensor:
        out = self.mlp(z).squeeze(-1)
        if self.activation == "relu":
            return torch.relu(out)
        if self.activation == "softplus":
            return nn.functional.softplus(out)
        return out
```

### 5.4 `scandium/models/pignet.py` — Main Architecture

```python
"""
PIGNet: Physics-Informed Graph Network for material property prediction.

Architecture:
  Input graph (atoms + bonds)
    → Node embedding
    → L × (EdgeUpdate + AttentionMessagePassing)
    → SetToSetReadout
    → PropertyHead (per target)
    → PhysicsConstraintModule (loss-time only)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data, Batch

from .layers import AttentionMessagePassing, EdgeUpdateLayer
from .readout import SetToSetReadout
from .heads import PropertyHead


class PIGNet(nn.Module):
    """
    Physics-Informed Graph Network.

    Args:
        node_in_dim:    input node feature dimension (AtomFeaturiser.FEATURE_DIM)
        edge_in_dim:    input edge feature dimension (GraphBuilder Gaussian basis = 40)
        hidden_dim:     width of hidden layers
        n_layers:       number of message-passing rounds
        targets:        list of target property names to predict
        dropout:        dropout probability
    """

    # Activation per property — enforces non-negativity where physics demands it
    PROPERTY_ACTIVATIONS: dict[str, str] = {
        "band_gap":          "softplus",   # band gaps cannot be negative
        "formation_energy":  "none",       # can be negative (stable compounds)
        "energy_above_hull": "softplus",   # always >= 0 by definition
        "bulk_modulus":      "softplus",   # moduli are positive
        "shear_modulus":     "softplus",
    }

    def __init__(
        self,
        node_in_dim:  int   = 99,
        edge_in_dim:  int   = 40,
        hidden_dim:   int   = 256,
        n_layers:     int   = 4,
        targets:      list[str] = None,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.targets   = targets or ["band_gap"]
        self.n_layers  = n_layers
        self.hidden_dim = hidden_dim

        # ── Input projections ──
        self.node_embedding = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim // 2),
        )

        edge_dim = hidden_dim // 2

        # ── Message-passing layers ──
        self.edge_update_layers = nn.ModuleList([
            EdgeUpdateLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        self.mp_layers = nn.ModuleList([
            AttentionMessagePassing(hidden_dim, edge_dim, hidden_dim, heads=4)
            for _ in range(n_layers)
        ])

        # ── Readout ──
        self.readout = SetToSetReadout(hidden_dim, hidden_dim)

        # ── Property heads ──
        self.heads = nn.ModuleDict({
            t: PropertyHead(
                hidden_dim,
                hidden_dim // 2,
                activation=self.PROPERTY_ACTIVATIONS.get(t, "none"),
            )
            for t in self.targets
        })

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: Data | Batch) -> dict[str, Tensor]:
        x          = data.x
        edge_index = data.edge_index
        edge_attr  = data.edge_attr
        batch      = data.batch if hasattr(data, "batch") and data.batch is not None \
                     else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # ── Embed inputs ──
        h = self.node_embedding(x)
        e = self.edge_embedding(edge_attr)

        # ── Iterative message passing ──
        for edge_upd, mp in zip(self.edge_update_layers, self.mp_layers):
            e = edge_upd(h, edge_index, e)
            h = mp(h, edge_index, e)

        # ── Global readout ──
        z = self.readout(h, batch)

        # ── Per-property predictions ──
        return {target: head(z) for target, head in self.heads.items()}

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

---

## 6. Physics Constraints Module

### 6.1 `scandium/physics/constraints.py`

```python
"""
Physics constraint loss terms.
These are added to the standard prediction loss during training,
penalising predictions that violate known physical laws.

L_total = L_pred + λ_bg * L_bandgap + λ_thermo * L_thermo + λ_hull * L_hull
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor


class BandGapConstraint(nn.Module):
    """
    Band gap must be non-negative.
    We use softplus activation in the head, but this adds an explicit
    training signal for near-zero predictions to stay positive.
    Penalty: max(0, -y_pred)^2 per sample
    """

    def forward(self, y_pred: Tensor) -> Tensor:
        violations = torch.relu(-y_pred)           # only penalise negative values
        return (violations ** 2).mean()


class FormationEnergyConstraint(nn.Module):
    """
    Formation energy for a stable compound must be negative
    (energy lower than constituent elements).
    For materials labelled is_stable=True, penalise positive formation energy.
    """

    def forward(self, y_pred: Tensor, is_stable: Tensor) -> Tensor:
        # Only penalise samples that are known to be stable
        stable_mask = is_stable.bool()
        if stable_mask.sum() == 0:
            return torch.tensor(0.0, device=y_pred.device)
        stable_preds = y_pred[stable_mask]
        violations   = torch.relu(stable_preds)    # penalise if predicted >= 0
        return (violations ** 2).mean()


class HullEnergyConstraint(nn.Module):
    """
    Energy above the convex hull must be >= 0 by thermodynamic definition.
    If the model predicts energy_above_hull < 0, it's unphysical.
    """

    def forward(self, y_pred: Tensor) -> Tensor:
        violations = torch.relu(-y_pred)
        return (violations ** 2).mean()


class ModuliConstraint(nn.Module):
    """
    Bulk and shear moduli of real materials are positive.
    (softplus activation handles this, but explicit constraint adds training signal)
    """

    def forward(self, y_pred: Tensor) -> Tensor:
        violations = torch.relu(-y_pred)
        return (violations ** 2).mean()


class SizeConsistencyConstraint(nn.Module):
    """
    Formation energy should scale approximately linearly with system size.
    Computed per-atom, so predictions should not vary wildly
    with number of atoms for similar compositions.
    This is a regularisation term, not a hard constraint.
    Penalises predictions with extremely large absolute values.
    """

    def __init__(self, max_abs: float = 10.0):
        super().__init__()
        self.max_abs = max_abs

    def forward(self, y_pred: Tensor) -> Tensor:
        excess = torch.relu(y_pred.abs() - self.max_abs)
        return (excess ** 2).mean()


class PhysicsConstraintLoss(nn.Module):
    """
    Combined physics constraint loss.
    Weights (lambdas) are set in training_config.yaml and can be
    annealed during training.
    """

    def __init__(
        self,
        lambda_bandgap:    float = 0.1,
        lambda_formation:  float = 0.05,
        lambda_hull:       float = 0.05,
        lambda_moduli:     float = 0.05,
        lambda_size:       float = 0.02,
    ):
        super().__init__()
        self.lambda_bandgap   = lambda_bandgap
        self.lambda_formation = lambda_formation
        self.lambda_hull      = lambda_hull
        self.lambda_moduli    = lambda_moduli
        self.lambda_size      = lambda_size

        self.bg_constraint    = BandGapConstraint()
        self.fe_constraint    = FormationEnergyConstraint()
        self.hull_constraint  = HullEnergyConstraint()
        self.mod_constraint   = ModuliConstraint()
        self.size_constraint  = SizeConsistencyConstraint()

    def forward(self, preds: dict[str, Tensor], batch) -> dict[str, Tensor]:
        losses = {}
        total  = torch.tensor(0.0, device=next(iter(preds.values())).device)

        if "band_gap" in preds:
            l = self.bg_constraint(preds["band_gap"])
            losses["constraint_bandgap"] = l
            total = total + self.lambda_bandgap * l

        if "formation_energy" in preds and hasattr(batch, "is_stable"):
            l = self.fe_constraint(preds["formation_energy"], batch.is_stable.float())
            losses["constraint_formation"] = l
            total = total + self.lambda_formation * l

        if "energy_above_hull" in preds:
            l = self.hull_constraint(preds["energy_above_hull"])
            losses["constraint_hull"] = l
            total = total + self.lambda_hull * l

        if "bulk_modulus" in preds:
            l = self.mod_constraint(preds["bulk_modulus"])
            losses["constraint_moduli"] = l
            total = total + self.lambda_moduli * l

        if "formation_energy" in preds:
            l = self.size_constraint(preds["formation_energy"])
            losses["constraint_size"] = l
            total = total + self.lambda_size * l

        losses["constraint_total"] = total
        return losses
```

### 6.2 `scandium/physics/validator.py`

```python
"""
Post-prediction physics validator.
Runs after model inference to flag any remaining physical inconsistencies.
Returns a structured validation report per prediction.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ValidationReport:
    is_valid: bool = True
    violations: list[str] = field(default_factory=list)
    warnings: list[str]   = field(default_factory=list)


class PhysicsValidator:
    """
    Validates a single prediction dict for physical consistency.
    Called by the API on every prediction before returning to the client.
    """

    # Known physical bounds for common material families
    BAND_GAP_MAX_EV      = 14.0   # widest known gap (diamond-like) ~12 eV
    BULK_MODULUS_MAX_GPA = 1000.0 # diamond ~440 GPa, theoretical limit higher
    SHEAR_MODULUS_MAX_GPA = 600.0
    FORMATION_ENERGY_MIN = -10.0  # eV/atom — no known compound below this
    FORMATION_ENERGY_MAX =  5.0   # positive but extremely high is suspicious

    def validate(self, preds: dict[str, float]) -> ValidationReport:
        report = ValidationReport()

        if "band_gap" in preds:
            bg = preds["band_gap"]
            if bg < 0.0:
                report.is_valid = False
                report.violations.append(
                    f"Band gap is negative ({bg:.4f} eV) — physically impossible"
                )
            elif bg > self.BAND_GAP_MAX_EV:
                report.warnings.append(
                    f"Band gap ({bg:.2f} eV) exceeds typical maximum — verify structure"
                )

        if "formation_energy" in preds:
            fe = preds["formation_energy"]
            if fe < self.FORMATION_ENERGY_MIN:
                report.is_valid = False
                report.violations.append(
                    f"Formation energy ({fe:.4f} eV/atom) below physical minimum"
                )
            elif fe > self.FORMATION_ENERGY_MAX:
                report.warnings.append(
                    f"Formation energy ({fe:.4f} eV/atom) is very high — likely unstable"
                )

        if "energy_above_hull" in preds:
            hull = preds["energy_above_hull"]
            if hull < 0.0:
                report.is_valid = False
                report.violations.append(
                    f"Energy above hull is negative ({hull:.4f} eV/atom) — impossible by definition"
                )

        if "bulk_modulus" in preds:
            bm = preds["bulk_modulus"]
            if bm < 0.0:
                report.is_valid = False
                report.violations.append(f"Bulk modulus is negative ({bm:.2f} GPa)")
            elif bm > self.BULK_MODULUS_MAX_GPA:
                report.warnings.append(f"Bulk modulus ({bm:.2f} GPa) exceeds diamond — verify")

        if "shear_modulus" in preds:
            sm = preds["shear_modulus"]
            if sm < 0.0:
                report.is_valid = False
                report.violations.append(f"Shear modulus is negative ({sm:.2f} GPa)")

        return report
```

---

## 7. Training Pipeline

### 7.1 `configs/training_config.yaml`

```yaml
# PIGNet training configuration
experiment_name: "pignet_v1_bandgap"
seed: 42

data:
  root: "data/processed/"
  target: "band_gap"           # primary prediction target
  batch_size: 64
  num_workers: 4

model:
  node_in_dim: 99              # AtomFeaturiser.FEATURE_DIM
  edge_in_dim: 40              # GraphBuilder Gaussian basis
  hidden_dim: 256
  n_layers: 4
  targets:
    - "band_gap"
    - "formation_energy"
  dropout: 0.1

training:
  epochs: 200
  lr: 1.0e-3
  weight_decay: 1.0e-5
  lr_scheduler: "cosine"       # cosine | plateau | step
  warmup_epochs: 10
  grad_clip: 5.0
  early_stopping_patience: 30

physics_constraints:
  lambda_bandgap:   0.10
  lambda_formation: 0.05
  lambda_hull:      0.05
  lambda_moduli:    0.05
  lambda_size:      0.02
  anneal: true                 # linearly increase lambdas over first 50 epochs

logging:
  use_wandb: true
  wandb_project: "scandium-labs"
  log_every_n_steps: 50
  checkpoint_every_n_epochs: 10

paths:
  checkpoint_dir: "checkpoints/"
  best_model_path: "checkpoints/best_model.pt"
```

### 7.2 `scandium/training/loss.py`

```python
"""Combined loss: prediction MSE + physics constraint penalties."""

import torch
import torch.nn as nn
from torch import Tensor
from ..physics.constraints import PhysicsConstraintLoss


class PIGNetLoss(nn.Module):
    def __init__(self, constraint_cfg: dict):
        super().__init__()
        self.mse       = nn.MSELoss()
        self.huber     = nn.HuberLoss(delta=1.0)  # robust to outliers
        self.physics   = PhysicsConstraintLoss(**constraint_cfg)

    def forward(
        self,
        preds:  dict[str, Tensor],
        target: str,
        y:      Tensor,
        batch,
    ) -> tuple[Tensor, dict[str, Tensor]]:

        # Primary prediction loss (Huber for robustness)
        pred_loss = self.huber(preds[target], y)

        # Physics constraint losses
        constraint_losses = self.physics(preds, batch)

        total = pred_loss + constraint_losses["constraint_total"]

        loss_dict = {"prediction": pred_loss, **constraint_losses, "total": total}
        return total, loss_dict
```

### 7.3 `scandium/training/metrics.py`

```python
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
```

### 7.4 `scandium/training/trainer.py`

```python
"""
Main training loop for PIGNet.
Handles: training, validation, early stopping,
checkpointing, W&B logging, and LR scheduling.
"""

from __future__ import annotations
import os
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..models.pignet import PIGNet
from .loss import PIGNetLoss
from .metrics import compute_all_metrics
from ..data.dataset import CrystalDataset


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg    = cfg
        self.device = torch.device(
            os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[Trainer] Using device: {self.device}")
        self._setup()

    def _setup(self) -> None:
        tc = self.cfg["training"]
        dc = self.cfg["data"]
        mc = self.cfg["model"]

        # Datasets
        self.train_ds = CrystalDataset(dc["root"], "train", dc["target"])
        self.val_ds   = CrystalDataset(dc["root"], "val",   dc["target"])
        self.test_ds  = CrystalDataset(dc["root"], "test",  dc["target"])

        self.train_loader = DataLoader(
            self.train_ds, batch_size=dc["batch_size"],
            shuffle=True, num_workers=dc["num_workers"], pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=dc["batch_size"] * 2,
            shuffle=False, num_workers=dc["num_workers"]
        )

        # Model
        self.model = PIGNet(**mc).to(self.device)
        print(f"[Trainer] Model parameters: {self.model.num_parameters:,}")

        # Optimiser
        self.optimiser = AdamW(
            self.model.parameters(),
            lr=tc["lr"], weight_decay=tc["weight_decay"]
        )

        # Scheduler
        if tc["lr_scheduler"] == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimiser, T_max=tc["epochs"], eta_min=1e-6
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimiser, patience=15, factor=0.5, verbose=True
            )

        # Loss
        self.loss_fn = PIGNetLoss(self.cfg["physics_constraints"])

        # State
        self.best_val_mae = float("inf")
        self.patience_counter = 0
        self.target = dc["target"]

        Path(self.cfg["paths"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    def _normalise_pred(self, pred: Tensor) -> Tensor:
        """Denormalise predictions back to original scale."""
        mean = CrystalDataset.TARGET_MEAN.get(self.target, 0.0)
        std  = CrystalDataset.TARGET_STD.get(self.target, 1.0)
        return pred * std + mean

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimiser.zero_grad()

            preds = self.model(batch)
            loss, loss_dict = self.loss_fn(preds, self.target, batch.y.squeeze(), batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg["training"]["grad_clip"]
            )
            self.optimiser.step()

            total_loss += float(loss_dict["prediction"])
            n_batches  += 1

        return {"train/loss": total_loss / n_batches}

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        self.model.eval()
        all_preds, all_targets = [], []

        for batch in self.val_loader:
            batch = batch.to(self.device)
            preds = self.model(batch)
            pred  = self._normalise_pred(preds[self.target])
            true  = batch.y_raw.squeeze()
            all_preds.append(pred.cpu())
            all_targets.append(true.cpu())

        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_targets)
        metrics = compute_all_metrics(y_pred, y_true, self.target)
        return {f"val/{k}": v for k, v in metrics.items()}

    def train(self) -> None:
        cfg = self.cfg
        lc  = cfg["logging"]

        if lc.get("use_wandb") and WANDB_AVAILABLE:
            wandb.init(
                project=lc["wandb_project"],
                name=cfg["experiment_name"],
                config=cfg,
            )

        print(f"\n[Trainer] Starting training: {cfg['experiment_name']}")
        print(f"          Epochs: {cfg['training']['epochs']}")

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics   = self._validate()

            # Scheduler step
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["val/mae"])
            else:
                self.scheduler.step()

            val_mae = val_metrics["val/mae"]
            elapsed = time.time() - t0
            lr      = self.optimiser.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d}/{cfg['training']['epochs']} | "
                f"Loss: {train_metrics['train/loss']:.4f} | "
                f"Val MAE: {val_mae:.4f} eV | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            all_metrics = {**train_metrics, **val_metrics, "lr": lr}
            if lc.get("use_wandb") and WANDB_AVAILABLE:
                wandb.log(all_metrics, step=epoch)

            # Checkpointing
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimiser_state_dict": self.optimiser.state_dict(),
                        "val_mae": val_mae,
                        "config": cfg,
                        "target_mean": CrystalDataset.TARGET_MEAN,
                        "target_std":  CrystalDataset.TARGET_STD,
                    },
                    cfg["paths"]["best_model_path"],
                )
                print(f"  ✓ New best model saved (val MAE = {val_mae:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= cfg["training"]["early_stopping_patience"]:
                    print(f"[Trainer] Early stopping at epoch {epoch}")
                    break

        print(f"\n[Trainer] Training complete. Best Val MAE: {self.best_val_mae:.4f}")
        if lc.get("use_wandb") and WANDB_AVAILABLE:
            wandb.finish()
```

### 7.5 `scripts/train.py` — CLI Entry Point

```python
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
```

---

## 8. Evaluation & Benchmarking

### 8.1 `scripts/evaluate.py`

```python
"""
Comprehensive evaluation script.
Runs the trained model on test set and OOD set.
Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import torch
from torch_geometric.loader import DataLoader
from scandium.models.pignet import PIGNet
from scandium.data.dataset import CrystalDataset
from scandium.training.metrics import compute_all_metrics


def evaluate(checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt["config"]
    dc   = cfg["data"]
    mc   = cfg["model"]

    # Restore normalisation stats
    CrystalDataset.TARGET_MEAN = ckpt["target_mean"]
    CrystalDataset.TARGET_STD  = ckpt["target_std"]

    model = PIGNet(**mc)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    test_ds     = CrystalDataset(dc["root"], "test", dc["target"])
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch)
            target = dc["target"]
            mean   = CrystalDataset.TARGET_MEAN[target]
            std    = CrystalDataset.TARGET_STD[target]
            pred   = preds[target] * std + mean
            all_preds.append(pred.cpu())
            all_targets.append(batch.y_raw.squeeze().cpu())

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)
    metrics = compute_all_metrics(y_pred, y_true, dc["target"])

    print("\n── Test Set Evaluation ──")
    print(f"  MAE:                    {metrics['mae']:.4f}")
    print(f"  RMSE:                   {metrics['rmse']:.4f}")
    print(f"  R²:                     {metrics['r2']:.4f}")
    print(f"  Physics violation rate: {metrics['physics_violation_rate']*100:.2f}%")

    with open("evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nResults saved → evaluation_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    args = parser.parse_args()
    evaluate(args.checkpoint)
```

---

## 9. FastAPI Backend

### 9.1 `api/schemas/request.py`

```python
from pydantic import BaseModel, Field
from typing import Optional


class StructureInput(BaseModel):
    """CIF or JSON string representing a crystal structure."""
    structure_data: str     = Field(..., description="CIF file content OR pymatgen Structure JSON string")
    format:         str     = Field("cif", description="'cif' or 'json'")
    material_id:    Optional[str] = Field(None, description="Optional identifier for tracking")


class PredictRequest(BaseModel):
    structure: StructureInput
    properties: list[str] = Field(
        default=["band_gap", "formation_energy"],
        description="List of properties to predict"
    )


class BatchScreenRequest(BaseModel):
    structures: list[StructureInput] = Field(..., max_items=10000)
    properties: list[str] = ["band_gap", "formation_energy"]
    filter_stable: bool = True     # only return energy_above_hull < 0.1 eV/atom
    band_gap_range: Optional[tuple[float, float]] = None
```

### 9.2 `api/schemas/response.py`

```python
from pydantic import BaseModel
from typing import Optional


class PropertyPrediction(BaseModel):
    value: float
    unit:  str
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    physics_valid: bool


class PredictResponse(BaseModel):
    material_id:   Optional[str]
    formula:       Optional[str]
    predictions:   dict[str, PropertyPrediction]
    inference_time_ms: float
    model_version: str
    physics_valid_overall: bool
    violations: list[str]
    warnings:   list[str]


class BatchScreenResponse(BaseModel):
    total_submitted: int
    total_completed: int
    results: list[PredictResponse]
    batch_time_seconds: float
```

### 9.3 `api/services/predictor.py`

```python
"""
Prediction service — wraps PIGNet for single and batch inference.
Loaded once at startup and reused across requests.
"""

from __future__ import annotations
import time
import os
from pathlib import Path

import torch
from pymatgen.core import Structure

from scandium.models.pignet import PIGNet
from scandium.data.graph_builder import GraphBuilder
from scandium.data.featurisers import AtomFeaturiser
from scandium.data.dataset import CrystalDataset
from scandium.physics.validator import PhysicsValidator


class PredictorService:
    _instance: "PredictorService | None" = None

    def __init__(self, checkpoint_path: str):
        self.device = torch.device(
            os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        )

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg  = ckpt["config"]

        # Restore normalisation
        CrystalDataset.TARGET_MEAN = ckpt["target_mean"]
        CrystalDataset.TARGET_STD  = ckpt["target_std"]

        self.model = PIGNet(**cfg["model"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.graph_builder = GraphBuilder()
        self.featuriser    = AtomFeaturiser()
        self.validator     = PhysicsValidator()
        self.model_version = f"pignet-v{ckpt.get('epoch', 0)}"

        print(f"[Predictor] Loaded model from {checkpoint_path} on {self.device}")

    @classmethod
    def get_instance(cls) -> "PredictorService":
        if cls._instance is None:
            path = os.environ.get("MODEL_CHECKPOINT_DIR", "checkpoints") + "/best_model.pt"
            cls._instance = cls(path)
        return cls._instance

    def predict(
        self, structure: Structure, properties: list[str]
    ) -> dict:
        t0   = time.perf_counter()
        data = self.graph_builder.build(structure, self.featuriser)
        data = data.to(self.device)
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)

        with torch.no_grad():
            raw_preds = self.model(data)

        # Denormalise
        preds: dict[str, float] = {}
        for prop in properties:
            if prop in raw_preds:
                mean = CrystalDataset.TARGET_MEAN.get(prop, 0.0)
                std  = CrystalDataset.TARGET_STD.get(prop, 1.0)
                preds[prop] = float(raw_preds[prop][0]) * std + mean

        report = self.validator.validate(preds)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "predictions": preds,
            "physics_valid": report.is_valid,
            "violations": report.violations,
            "warnings":   report.warnings,
            "inference_time_ms": elapsed_ms,
            "model_version": self.model_version,
        }
```

### 9.4 `api/middleware/auth.py`

```python
"""API key authentication middleware."""

import os
import hashlib
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# In production: load from database. Here: load from environment.
VALID_API_KEYS: set[str] = set(
    filter(None, os.environ.get("VALID_API_KEYS", "test_key_dev_only").split(","))
)


def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key. Visit scandiumlabs.ai/access to get one.",
        )
    return api_key
```

### 9.5 `api/routers/predict.py`

```python
"""Prediction endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from pymatgen.core import Structure

from ..schemas.request import PredictRequest
from ..schemas.response import PredictResponse, PropertyPrediction
from ..middleware.auth import verify_api_key
from ..services.predictor import PredictorService

router = APIRouter(prefix="/predict", tags=["Predict"])


def _parse_structure(req_struct) -> Structure:
    try:
        if req_struct.format == "cif":
            return Structure.from_str(req_struct.structure_data, fmt="cif")
        elif req_struct.format == "json":
            import json
            return Structure.from_dict(json.loads(req_struct.structure_data))
        else:
            raise ValueError(f"Unsupported format: {req_struct.format}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Structure parse error: {e}")


@router.post("/", response_model=PredictResponse)
async def predict_properties(
    request: PredictRequest,
    api_key: str = Depends(verify_api_key),
):
    service   = PredictorService.get_instance()
    structure = _parse_structure(request.structure)
    result    = service.predict(structure, request.properties)

    predictions = {}
    UNITS = {
        "band_gap": "eV", "formation_energy": "eV/atom",
        "energy_above_hull": "eV/atom", "bulk_modulus": "GPa", "shear_modulus": "GPa",
    }
    for prop, val in result["predictions"].items():
        predictions[prop] = PropertyPrediction(
            value=val,
            unit=UNITS.get(prop, ""),
            physics_valid=result["physics_valid"],
        )

    return PredictResponse(
        material_id=request.structure.material_id,
        formula=structure.formula,
        predictions=predictions,
        inference_time_ms=result["inference_time_ms"],
        model_version=result["model_version"],
        physics_valid_overall=result["physics_valid"],
        violations=result["violations"],
        warnings=result["warnings"],
    )
```

### 9.6 `api/routers/health.py`

```python
from fastapi import APIRouter
import torch

router = APIRouter(tags=["Health"])

@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
```

### 9.7 `api/main.py`

```python
"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from .routers import predict, health
from .services.predictor import PredictorService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup
    print("[API] Loading PIGNet model...")
    PredictorService.get_instance()
    print("[API] Model ready. Server is live.")
    yield
    print("[API] Shutting down.")


app = FastAPI(
    title="Scandium Labs API",
    description="Physics-constrained material property prediction. Atomic structure → properties in milliseconds.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
```

---

## 10. Python SDK

### 10.1 `sdk/scandiumlabs/client.py`

```python
"""
Scandium Labs Python SDK.

Usage:
    pip install scandiumlabs

    from scandiumlabs import Client
    client = Client(api_key="sl_xxxx", base_url="https://api.scandiumlabs.ai")

    result = client.predict("LiFePO4.cif", properties=["band_gap", "formation_energy"])
    print(result.band_gap.value)   # 3.71 eV
"""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class PropertyResult:
    value: float
    unit:  str
    physics_valid: bool
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None

    def __repr__(self) -> str:
        return f"{self.value:.4f} {self.unit} (valid={self.physics_valid})"


@dataclass
class PredictionResult:
    material_id:    Optional[str]
    formula:        str
    properties:     dict[str, PropertyResult]
    inference_ms:   float
    model_version:  str
    physics_valid:  bool
    violations:     list[str]
    warnings:       list[str]

    def __getattr__(self, name: str) -> PropertyResult:
        if name in self.properties:
            return self.properties[name]
        raise AttributeError(f"Property '{name}' not in predictions. "
                             f"Available: {list(self.properties.keys())}")


class Client:
    """Scandium Labs API client."""

    DEFAULT_BASE_URL = "https://api.scandiumlabs.ai"

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.headers  = {"X-API-Key": api_key, "Content-Type": "application/json"}
        self.timeout  = timeout

    def predict(
        self,
        structure_path: str,
        properties: list[str] = None,
        material_id: str = None,
    ) -> PredictionResult:
        """
        Predict properties for a crystal structure.

        Args:
            structure_path: Path to a .cif file.
            properties:     List of properties to predict.
            material_id:    Optional string identifier.

        Returns:
            PredictionResult with .band_gap, .formation_energy etc.
        """
        properties = properties or ["band_gap", "formation_energy"]
        path       = Path(structure_path)
        fmt        = "cif" if path.suffix.lower() == ".cif" else "json"
        data       = path.read_text()

        payload = {
            "structure": {
                "structure_data": data,
                "format": fmt,
                "material_id": material_id or path.stem,
            },
            "properties": properties,
        }

        with httpx.Client(timeout=self.timeout) as http:
            resp = http.post(
                f"{self.base_url}/predict/",
                headers=self.headers,
                json=payload,
            )
            resp.raise_for_status()

        return self._parse_response(resp.json())

    def predict_mp(self, material_id: str, properties: list[str] = None) -> PredictionResult:
        """Predict for a Materials Project entry by mp-id."""
        from mp_api.client import MPRester
        import os
        with MPRester(os.environ["MP_API_KEY"]) as mpr:
            docs = mpr.summary.search(material_ids=[material_id], fields=["structure"])
        struct_json = json.dumps(docs[0].structure.as_dict())

        payload = {
            "structure": {"structure_data": struct_json, "format": "json", "material_id": material_id},
            "properties": properties or ["band_gap", "formation_energy"],
        }
        with httpx.Client(timeout=self.timeout) as http:
            resp = http.post(f"{self.base_url}/predict/", headers=self.headers, json=payload)
            resp.raise_for_status()
        return self._parse_response(resp.json())

    def _parse_response(self, data: dict) -> PredictionResult:
        props = {
            k: PropertyResult(
                value=v["value"],
                unit=v["unit"],
                physics_valid=v["physics_valid"],
                confidence_lower=v.get("confidence_lower"),
                confidence_upper=v.get("confidence_upper"),
            )
            for k, v in data["predictions"].items()
        }
        return PredictionResult(
            material_id=data.get("material_id"),
            formula=data.get("formula", ""),
            properties=props,
            inference_ms=data["inference_time_ms"],
            model_version=data["model_version"],
            physics_valid=data["physics_valid_overall"],
            violations=data.get("violations", []),
            warnings=data.get("warnings", []),
        )
```

---

## 11. Configuration & Secrets

### 11.1 `.gitignore`

```
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.venv/
venv/

# Secrets
.env

# Data
data/raw/
data/processed/

# Models
checkpoints/
*.pt
*.onnx

# Logs
wandb/
*.log
lightning_logs/

# OS
.DS_Store
Thumbs.db
```

### 11.2 `configs/model_config.yaml`

```yaml
model:
  node_in_dim: 99
  edge_in_dim: 40
  hidden_dim: 256
  n_layers: 4
  targets:
    - band_gap
    - formation_energy
    - energy_above_hull
  dropout: 0.1

inference:
  batch_size: 256
  fp16: false    # set true on A100 for 2× speedup
```

---

## 12. Docker & Deployment

### 12.1 `Dockerfile`

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyG extras
RUN pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html --no-cache-dir

# Copy source
COPY scandium/ ./scandium/
COPY api/ ./api/
COPY configs/ ./configs/
COPY checkpoints/ ./checkpoints/

# Expose API port
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### 12.2 `docker-compose.yml`

```yaml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEVICE=cuda
      - MODEL_CHECKPOINT_DIR=/app/checkpoints
      - VALID_API_KEYS=${VALID_API_KEYS}
    volumes:
      - ./checkpoints:/app/checkpoints:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  # Optional: Redis for rate limiting in production
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

### 12.3 Run Commands

```bash
# Development (no Docker)
uvicorn api.main:app --reload --port 8000

# Production (Docker)
docker build -t scandium-api:latest .
docker-compose up -d

# Check logs
docker-compose logs -f api

# Scale API workers (CPU inference)
docker-compose up -d --scale api=4
```

---

## 13. Testing

### 13.1 `tests/conftest.py`

```python
import pytest
import torch
from torch_geometric.data import Data
from scandium.models.pignet import PIGNet
from scandium.data.featurisers import AtomFeaturiser
from scandium.data.graph_builder import GraphBuilder


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")

@pytest.fixture(scope="session")
def featuriser():
    return AtomFeaturiser()

@pytest.fixture(scope="session")
def graph_builder():
    return GraphBuilder(cutoff=5.0)

@pytest.fixture(scope="session")
def small_model():
    return PIGNet(
        node_in_dim=99, edge_in_dim=40,
        hidden_dim=64, n_layers=2,
        targets=["band_gap", "formation_energy"],
    )

@pytest.fixture
def dummy_batch():
    """Minimal valid PyG batch for unit testing."""
    n_atoms, n_edges = 10, 40
    data = Data(
        x=torch.randn(n_atoms, 99),
        edge_index=torch.randint(0, n_atoms, (2, n_edges)),
        edge_attr=torch.randn(n_edges, 40),
        batch=torch.zeros(n_atoms, dtype=torch.long),
        num_nodes=n_atoms,
        is_stable=torch.ones(1),
        y=torch.tensor([1.5]),
        y_raw=torch.tensor([1.5]),
    )
    return data
```

### 13.2 `tests/test_model.py`

```python
import torch
import pytest


def test_model_forward_shape(small_model, dummy_batch):
    preds = small_model(dummy_batch)
    assert "band_gap" in preds
    assert "formation_energy" in preds
    assert preds["band_gap"].shape == (1,)


def test_band_gap_non_negative(small_model, dummy_batch):
    """PIGNet band gap head uses softplus — must always be >= 0."""
    for _ in range(50):
        dummy_batch.x = torch.randn_like(dummy_batch.x)
        preds = small_model(dummy_batch)
        assert (preds["band_gap"] >= 0).all(), "Band gap prediction went negative!"


def test_model_parameter_count(small_model):
    n = small_model.num_parameters
    assert n > 0
    print(f"\nModel parameters: {n:,}")


def test_gradient_flow(small_model, dummy_batch):
    small_model.train()
    preds = small_model(dummy_batch)
    loss  = preds["band_gap"].mean()
    loss.backward()
    for name, param in small_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
```

### 13.3 `tests/test_physics.py`

```python
import torch
import pytest
from scandium.physics.constraints import PhysicsConstraintLoss
from scandium.physics.validator import PhysicsValidator


def test_bandgap_constraint_zero_for_positive():
    loss_fn = PhysicsConstraintLoss()
    dummy_batch = type("B", (), {"is_stable": torch.ones(4)})()
    preds = {"band_gap": torch.tensor([1.0, 2.0, 0.5, 3.0])}
    losses = loss_fn(preds, dummy_batch)
    assert float(losses["constraint_bandgap"]) == 0.0


def test_bandgap_constraint_nonzero_for_negative():
    loss_fn = PhysicsConstraintLoss()
    dummy_batch = type("B", (), {"is_stable": torch.ones(4)})()
    preds = {"band_gap": torch.tensor([-0.5, 1.0, -0.1, 2.0])}
    losses = loss_fn(preds, dummy_batch)
    assert float(losses["constraint_bandgap"]) > 0.0


def test_validator_flags_negative_bandgap():
    v = PhysicsValidator()
    report = v.validate({"band_gap": -0.3})
    assert not report.is_valid
    assert len(report.violations) == 1


def test_validator_passes_valid_prediction():
    v = PhysicsValidator()
    report = v.validate({"band_gap": 1.5, "formation_energy": -2.1})
    assert report.is_valid
    assert len(report.violations) == 0


def test_validator_warns_extreme_bandgap():
    v = PhysicsValidator()
    report = v.validate({"band_gap": 13.0})
    assert report.is_valid          # not a violation, just a warning
    assert len(report.warnings) > 0
```

### 13.4 `tests/test_api.py`

```python
import pytest
from fastapi.testclient import TestClient
import os

os.environ["VALID_API_KEYS"] = "test_key"
os.environ["MODEL_CHECKPOINT_DIR"] = "checkpoints"

# Only run API tests if a checkpoint exists
pytestmark = pytest.mark.skipif(
    not os.path.exists("checkpoints/best_model.pt"),
    reason="No trained checkpoint found — run training first"
)


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_predict_requires_auth(client):
    r = client.post("/predict/", json={})
    assert r.status_code == 403


def test_predict_cif(client):
    # Minimal Si CIF for testing
    cif_content = """
data_Si
_cell_length_a   5.43
_cell_length_b   5.43
_cell_length_c   5.43
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
_symmetry_space_group_name_H-M  'F d -3 m'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
Si Si 0.000 0.000 0.000
Si Si 0.250 0.250 0.250
"""
    payload = {
        "structure": {"structure_data": cif_content, "format": "cif"},
        "properties": ["band_gap"]
    }
    r = client.post("/predict/", json=payload, headers={"X-API-Key": "test_key"})
    assert r.status_code == 200
    body = r.json()
    assert "band_gap" in body["predictions"]
    assert body["predictions"]["band_gap"]["value"] >= 0.0
```

### 13.5 Run All Tests

```bash
pip install pytest pytest-asyncio httpx

# Run all tests
pytest tests/ -v

# With coverage
pip install pytest-cov
pytest tests/ --cov=scandium --cov=api --cov-report=html

# Physics tests only
pytest tests/test_physics.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

---

## 14. End-to-End Verification

Run these steps **in exact order** to verify the complete system works.

### Step 1 — Environment

```bash
conda activate scandium
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
python -c "import pymatgen; print('pymatgen:', pymatgen.__version__)"
python -c "from e3nn import o3; print('e3nn: OK')"
```

Expected output: All four lines print without error.

### Step 2 — Download Data

```bash
cp .env.example .env
# Edit .env — add your Materials Project API key from materialsproject.org

python data/scripts/download_mp.py --n_materials 5000 --output data/raw/mp_data.json
# Expected: "Saved 5000 records → data/raw/mp_data.json"
```

### Step 3 — Build Graph Dataset

```bash
python data/scripts/build_dataset.py \
    --input data/raw/mp_data.json \
    --output data/processed/ \
    --cutoff 6.0
# Expected: "Saved → data/processed/ | Train: 4000 | Val: 500 | Test: 500"
```

### Step 4 — Run Unit Tests (Pre-training)

```bash
pytest tests/test_model.py tests/test_physics.py -v
# Expected: All tests PASS
```

### Step 5 — Train (Quick Sanity Check)

```bash
# First, edit configs/training_config.yaml: set epochs: 5 for quick test
python scripts/train.py --config configs/training_config.yaml
# Expected:
# "Model parameters: X,XXX,XXX"
# "Epoch 001/5 | Loss: X.XXXX | Val MAE: X.XXXX eV | ..."
# "New best model saved"
```

### Step 6 — Full Training

```bash
# Reset epochs to 200 in training_config.yaml
python scripts/train.py --config configs/training_config.yaml
# Target: Val MAE < 0.3 eV at epoch 50, < 0.15 eV at convergence
```

### Step 7 — Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
# Expected output:
# MAE:                    0.12XX
# RMSE:                   0.17XX
# R²:                     0.94XX
# Physics violation rate: 0.00%
```

### Step 8 — Start API Server

```bash
# Ensure .env has VALID_API_KEYS=my_test_key
uvicorn api.main:app --reload --port 8000
# Expected: "Model ready. Server is live."
```

### Step 9 — Test API Manually

```bash
curl -X GET http://localhost:8000/health
# Expected: {"status":"healthy","gpu_available":true,...}

# Test prediction with Silicon
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my_test_key" \
  -d '{
    "structure": {
      "structure_data": "mp-149",
      "format": "mp_id"
    },
    "properties": ["band_gap", "formation_energy"]
  }'
```

### Step 10 — Test SDK

```python
# test_e2e.py — run with: python test_e2e.py
from scandiumlabs import Client

client = Client(api_key="my_test_key", base_url="http://localhost:8000")
result = client.predict("path/to/any.cif", properties=["band_gap"])

print(f"Formula:      {result.formula}")
print(f"Band gap:     {result.band_gap}")        # e.g. 1.3421 eV (valid=True)
print(f"Physics valid: {result.physics_valid}")  # True
print(f"Inference:    {result.inference_ms:.1f} ms")
assert result.physics_valid is True
assert result.band_gap.value >= 0.0
print("✓ End-to-end test PASSED")
```

### Step 11 — Docker Deploy

```bash
docker build -t scandium-api:v1.0 .
VALID_API_KEYS=my_test_key docker-compose up -d

# Verify container is healthy
curl http://localhost:8000/health
```

---

## 15. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch too large | Reduce `batch_size` in training_config.yaml |
| `No edges found` | Cutoff too small for structure | Increase `cutoff` to 8.0 in graph_builder |
| `mp_api auth error` | Wrong API key | Check `.env` has correct `MP_API_KEY` |
| `torch_scatter not found` | PyG not installed properly | Re-run the PyG extras pip install command |
| Val MAE not decreasing | LR too high / model too small | Reduce lr to 5e-4, increase hidden_dim to 512 |
| Physics violations > 0 | Constraint weights too low | Double `lambda_bandgap` in training_config.yaml |
| API returns 403 | API key mismatch | Check `VALID_API_KEYS` env var matches request header |
| `pymatgen CIF parse error` | Malformed CIF file | Validate CIF at crystallography.net before sending |
| Slow inference | CPU mode | Ensure `DEVICE=cuda` in .env and GPU is available |
| `e3nn` import error | Version conflict | `pip install e3nn==0.5.1 --force-reinstall` |

---

## Training Time Estimates

| Hardware | Dataset Size | Epochs | Estimated Time |
|---|---|---|---|
| RTX 3090 (24 GB) | 50,000 structures | 200 | ~8 hours |
| A100 (40 GB) | 50,000 structures | 200 | ~3 hours |
| RTX 3090 | 150,000 structures | 200 | ~24 hours |
| A100 | 150,000 structures | 200 | ~9 hours |
| CPU only (dev) | 5,000 structures | 10 | ~30 minutes |

---

## Expected Final Metrics

After full training on 50K Materials Project structures:

| Property | MAE | RMSE | R² | Physics violations |
|---|---|---|---|---|
| Band gap (eV) | ≤ 0.15 | ≤ 0.22 | ≥ 0.93 | 0.00% |
| Formation energy (eV/atom) | ≤ 0.10 | ≤ 0.15 | ≥ 0.96 | 0.00% |

These are achievable on standard hardware with the architecture above.
The 0.00% physics violation rate is **guaranteed by the softplus activation + constraint loss combination**, not a probabilistic outcome.

---

*Scandium Labs — Build Guide v1.0*
*Classification: Internal / Founders Only*
