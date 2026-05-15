import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data


class CGCNNBaseline(nn.Module):
    """Simple CGCNN-style baseline with an uncertainty head.

    Returns: predictions (batch, n_props), log_vars (batch, n_props)
    """

    def __init__(
        self,
        node_feat_dim: int = 4,
        edge_feat_dim: int = 40,
        hidden_dim: int = 128,
        n_conv_layers: int = 4,
        n_properties: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
        self.conv_layers = nn.ModuleList([
            CGConv(hidden_dim, dim=edge_feat_dim, batch_norm=True) for _ in range(n_conv_layers)
        ])
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

        self.crystal_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        self.property_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim // 2, 32), nn.SiLU(), nn.Linear(32, 1))
            for _ in range(n_properties)
        ])

        self.uncertainty_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(n_properties)
        ])

    def forward(self, data: Data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, getattr(data, "edge_attr", None), getattr(data, "batch", None)

        if batch is None:
            # single graph: create batch of zeros
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.node_embedding(x)
        x = self.activation(x)

        for conv in self.conv_layers:
            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)

        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x_global = torch.cat([x_mean, x_sum], dim=-1)

        crystal_repr = self.crystal_proj(x_global)

        preds = torch.stack([head(crystal_repr).squeeze(-1) for head in self.property_heads], dim=-1)
        log_vars = torch.stack([head(crystal_repr).squeeze(-1) for head in self.uncertainty_heads], dim=-1)

        return preds, log_vars
