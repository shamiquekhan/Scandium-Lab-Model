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