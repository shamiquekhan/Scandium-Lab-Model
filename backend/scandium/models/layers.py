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