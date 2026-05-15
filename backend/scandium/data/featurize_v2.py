import torch, numpy as np
from torch_geometric.data import Data
from pymatgen.core import Structure
from .featurize import atom_features, rbf_expansion   # reuse from v1

ANGLE_BINS = 16   # Fourier expansion of bond angles → 16 dims

def angle_fourier(cos_theta: np.ndarray, n_bins: int = ANGLE_BINS) -> np.ndarray:
    """
    Expand cos(θ) into a Fourier basis.
    cos_theta shape: (K,) — one value per (center, j, k) triplet.
    Returns: (K, n_bins)
    """
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))   # → [0, π]
    ns    = np.arange(1, n_bins + 1)                     # [1, 2, ..., n_bins]
    return np.cos(ns[np.newaxis, :] * theta[:, np.newaxis])  # (K, n_bins)

def compute_edge_angle_features(
    vecs: np.ndarray,      # (E, 3) — displacement vectors per edge
    edge_src: list[int],   # source node per edge
    n_nodes: int,
    n_bins: int = ANGLE_BINS,
) -> np.ndarray:
    """
    For each edge (i→j), average the Fourier-expanded angles
    to all other edges (i→k, k≠j) incident at node i.
    Returns: (E, n_bins)  — one angle feature vector per edge.
    """
    n_edges    = len(edge_src)
    angle_feat = np.zeros((n_edges, n_bins))

    # Group edge indices by source node
    from collections import defaultdict
    node2edges = defaultdict(list)
    for eid, src in enumerate(edge_src):
        node2edges[src].append(eid)

    # Normalise displacement vectors
    norms      = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    unit_vecs  = vecs / norms

    for eid in range(n_edges):
        i       = edge_src[eid]
        nbr_ids = [e for e in node2edges[i] if e != eid]
        if not nbr_ids:
            continue
        # cos(θ) between this edge and all other incident edges
        cos_vals       = unit_vecs[nbr_ids] @ unit_vecs[eid]          # (K,)
        angle_feat[eid] = angle_fourier(cos_vals, n_bins).mean(axis=0)  # mean over K

    return angle_feat   # (E, 16)


def structure_to_graph_v2(
    structure: Structure,
    cutoff: float = 6.0,
    max_neighbors: int = 12,
    label: dict = None,
) -> Data:
    """
    Updated featurization.
    Edge features: 40 RBF + 16 angle Fourier = 56 dims total.
    """
    # Node features (unchanged from v1)
    x = torch.tensor(
        [atom_features(s.specie.symbol) for s in structure],
        dtype=torch.float
    )

    edge_src, edge_dst, dists, vecs = [], [], [], []
    for i, site in enumerate(structure):
        nbrs = sorted(structure.get_neighbors(site, r=cutoff), key=lambda n: n.nn_distance)[:max_neighbors]
        for nb in nbrs:
            edge_src.append(i)
            edge_dst.append(nb.index)
            dists.append(nb.nn_distance)
            # displacement vector from site i to neighbour j (fractional → Cartesian)
            vecs.append(nb.coords - site.coords)

    vecs_np  = np.array(vecs)
    dists_np = np.array(dists)

    rbf      = rbf_expansion(dists_np)              # (E, 40)
    ang      = compute_edge_angle_features(          # (E, 16)
                   vecs_np, edge_src, structure.num_sites)
    edge_attr = torch.tensor(
        np.concatenate([rbf, ang], axis=1), dtype=torch.float)  # (E, 56)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    y = None
    if label is not None:
        y = torch.tensor([[label["band_gap"],
                           label["formation_energy_per_atom"],
                           label["energy_above_hull"]]],
                         dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
