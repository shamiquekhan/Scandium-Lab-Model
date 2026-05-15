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