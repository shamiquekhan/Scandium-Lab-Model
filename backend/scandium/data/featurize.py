"""
Crystal structure → PyTorch Geometric Data object.
All atom and bond features defined here.
"""
import torch
import numpy as np
from torch_geometric.data import Data
from pymatgen.core import Structure, Element

# ─── Atom feature lookup tables ───────────────────────────────────────────────
# Pauling electronegativities (subset; add more as needed)
ELECTRONEGATIVITY = {
    'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55,
    'N': 3.04,  'O': 3.44, 'F': 3.98, 'Na': 0.93, 'Mg': 1.31,
    'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
    'K': 0.82,  'Ca': 1.00, 'Ti': 1.54, 'Fe': 1.83, 'Ni': 1.91,
    'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01, 'As': 2.18,
    'Zr': 1.33, 'La': 1.10, 'Ba': 0.89, 'Pb': 2.33,
}

COMMON_OXIDATION = {
    'Li': 1, 'Na': 1, 'K': 1, 'Ca': 2, 'Ba': 2, 'Mg': 2,
    'Al': 3, 'Si': 4, 'Ti': 4, 'Fe': 3, 'Zr': 4, 'La': 3,
    'Ga': 3, 'N': -3, 'O': -2, 'S': -2, 'Cl': -1, 'F': -1, 'P': -3,
}

def atom_features(element_symbol: str) -> list[float]:
    """Return a 10-dimensional feature vector for an element."""
    try:
        el = Element(element_symbol)
    except:
        return [0.0] * 10

    Z = el.Z
    en = ELECTRONEGATIVITY.get(element_symbol, 2.0)
    cr = el.atomic_radius_calculated or el.atomic_radius or 1.5
    cr_pm = float(cr) * 100 if float(cr) < 10 else float(cr)
    try:
        ve = el.valence[-1] if isinstance(el.valence, (tuple, list)) else el.valence
    except:
        ve = 4
    ie = float(el.ionization_energies[0]) if el.ionization_energies else 10.0
    ox = COMMON_OXIDATION.get(element_symbol, 0)

    return [
        Z / 94.0,
        el.row / 7.0,
        (el.group or 1) / 18.0,
        en / 4.0,
        min(cr_pm, 250) / 250.0,
        min(ve, 8) / 8.0,
        min(ie, 25) / 25.0,
        float(el.is_metal),
        float(el.is_transition_metal),
        ox / 8.0,
    ]

def rbf_expansion(distances: np.ndarray, d_min=0.5, d_max=6.0, n_bins=40) -> np.ndarray:
    """
    Gaussian Radial Basis Function expansion.
    Converts scalar distances to 40-dim smooth feature vectors.
    """
    centers = np.linspace(d_min, d_max, n_bins)
    width = ((d_max - d_min) / n_bins) ** 2
    distances = distances[:, np.newaxis]
    rbf = np.exp(-((distances - centers) ** 2) / width)
    return rbf  # shape: (n_bonds, 40)

def structure_to_graph(
    structure: Structure,
    cutoff: float = 6.0,
    max_neighbors: int = 12,
    label: dict = None,
) -> Data:
    """
    Convert a pymatgen Structure to a torch_geometric.data.Data graph.

    Args:
        structure:      Pymatgen Structure object
        cutoff:         Bond cutoff radius in Angstroms
        max_neighbors:  Maximum number of neighbors per atom
        label:          Dict with target values: band_gap, formation_energy_per_atom,
                        energy_above_hull

    Returns:
        PyG Data object with x (node features), edge_index, edge_attr, y
    """
    # ── Node features ──────────────────────────────────────────────────────────
    node_features = []
    for site in structure:
        symbol = site.specie.symbol
        node_features.append(atom_features(symbol))
    x = torch.tensor(node_features, dtype=torch.float)  # (N, 10)

    # ── Edge features ──────────────────────────────────────────────────────────
    edge_src, edge_dst, distances = [], [], []
    for i, site in enumerate(structure):
        neighbors = structure.get_neighbors(site, r=cutoff)
        neighbors = sorted(neighbors, key=lambda n: n.nn_distance)[:max_neighbors]
        for neighbor in neighbors:
            edge_src.append(i)
            edge_dst.append(neighbor.index)
            distances.append(neighbor.nn_distance)

    if not distances:
        raise ValueError(f"No edges found — check cutoff ({cutoff} Å) vs structure size")

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    rbf = rbf_expansion(np.array(distances))
    edge_attr = torch.tensor(rbf, dtype=torch.float)  # (E, 40)

    # ── Labels ─────────────────────────────────────────────────────────────────
    y = None
    if label is not None:
        y = torch.tensor([
            label.get("band_gap", 0.0),
            label.get("formation_energy_per_atom", 0.0),
            label.get("energy_above_hull", 0.0),
        ], dtype=torch.float).unsqueeze(0)  # (1, 3)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
