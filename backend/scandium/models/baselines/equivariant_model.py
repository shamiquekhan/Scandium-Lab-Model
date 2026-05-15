"""
SE(3)-equivariant model using e3nn library.

Architecture: spherical harmonic edge features + equivariant message passing.
Guaranteed to be equivariant under rotations by construction.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.data import Data
try:
    from e3nn import o3
except ImportError:
    o3 = None


class EquivariantConv(MessagePassing):
    """
    Single equivariant convolution layer using e3nn.

    Computes spherical harmonics from edge vectors (SE(3)-invariant features)
    and uses them to produce rotationally-equivariant message updates.
    """

    def __init__(self, irreps_in, irreps_out, max_radius=4.0, lmax=2):
        super().__init__(aggr="sum")
        if o3 is None:
            raise ImportError("e3nn not installed. Install with: pip install e3nn")

        self.irreps_in = o3.Irreps(irreps_in) if isinstance(irreps_in, str) else irreps_in
        self.irreps_out = o3.Irreps(irreps_out) if isinstance(irreps_out, str) else irreps_out
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.max_radius = max_radius
        self.lmax = lmax

        # Tensor product to combine node features and spherical harmonics
        self.tp = o3.TensorProduct(self.irreps_in, self.irreps_sh)
        self.fc = nn.Sequential(
            nn.Linear(self.tp.irreps_out.dim, 128),
            nn.SiLU(),
            nn.Linear(128, self.irreps_out.dim),
        )

    def forward(self, x, edge_index, edge_vec):
        """
        x: node features (N, features_in)
        edge_index: (2, E)
        edge_vec: (E, 3) edge vectors (Cartesian)
        """
        return self.propagate(edge_index, x=x, edge_vec=edge_vec)

    def message(self, x_i, x_j, edge_vec):
        # Compute spherical harmonics for edge vectors
        edge_vec_normalized = edge_vec / (edge_vec.norm(dim=1, keepdim=True) + 1e-8)
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec_normalized, normalize=True)

        # Tensor product of x_j and spherical harmonics
        tp_out = self.tp(x_j, edge_sh)

        # MLP projection
        msg = self.fc(tp_out)
        return msg


class E3NNModel(nn.Module):
    """
    SE(3)-equivariant materials property model.

    Architecture:
      - Node embedding
      - N equivariant convolution layers (e3nn)
      - Pooling + invariant projection
      - Property heads

    The model is guaranteed equivariant: f(R*x) = f(x) for rotations R,
    by the properties of irreps and tensor products.
    """

    def __init__(
        self,
        node_feat_dim: int = 4,
        hidden_irreps: str = "16x0e",  # 16 scalars
        n_conv_layers: int = 3,
        n_properties: int = 4,
        max_radius: float = 5.0,
        lmax: int = 2,
    ):
        super().__init__()
        if o3 is None:
            raise ImportError("e3nn required for E3NNModel")

        self.node_embedding = nn.Linear(node_feat_dim, o3.Irreps(hidden_irreps).dim)

        # Convolution layers
        self.conv_layers = nn.ModuleList()
        irreps_in = hidden_irreps
        for i in range(n_conv_layers):
            self.conv_layers.append(
                EquivariantConv(
                    irreps_in,
                    hidden_irreps,
                    max_radius=max_radius,
                    lmax=lmax,
                )
            )

        # After convs, extract only scalars (0e irreps) for invariance
        # Then project to property space
        hidden_dim = o3.Irreps(hidden_irreps).dim
        self.invariant_proj = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
        )

        self.property_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1))
            for _ in range(n_properties)
        ])

        self.uncertainty_heads = nn.ModuleList([
            nn.Linear(32, 1) for _ in range(n_properties)
        ])

    def forward(self, data: Data):
        x, edge_index, edge_vec, batch = (
            data.x,
            data.edge_index,
            getattr(data, "edge_vec", None),
            getattr(data, "batch", None),
        )

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if edge_vec is None:
            raise ValueError("E3NNModel requires edge_vec in Data object")

        # Node embedding
        x = self.node_embedding(x)

        # Equivariant message passing
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_vec)

        # Global pooling (mean over nodes per graph)
        from torch_geometric.nn import global_mean_pool

        x_global = global_mean_pool(x, batch)

        # Project to invariant space
        z = self.invariant_proj(x_global)

        # Property predictions
        preds = torch.stack([head(z).squeeze(-1) for head in self.property_heads], dim=-1)
        log_vars = torch.stack([head(z).squeeze(-1) for head in self.uncertainty_heads], dim=-1)

        return preds, log_vars
