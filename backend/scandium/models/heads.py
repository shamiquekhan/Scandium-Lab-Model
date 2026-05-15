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