"""
PIGNet: Physics-Informed Graph Network for material property prediction.
Architecture: PIGNetConv layers → global pool → physics-constrained heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops

ATOM_DIM = 10   # featurize.py atom feature dim
BOND_DIM = 40   # RBF expansion bins

# ─── Utility blocks ───────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=2, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


# ─── PIGNetConv ───────────────────────────────────────────────────────────────
class PIGNetConv(MessagePassing):
    """
    One message-passing layer.
    Message:   m_ij = MLP(h_i || h_j || e_ij)
    Aggregate: sum
    Update:    h_i' = LayerNorm(h_i + W * agg)
    """
    def __init__(self, node_dim: int, edge_dim: int, dropout=0.1):
        super().__init__(aggr="sum")
        msg_in = node_dim + node_dim + edge_dim
        self.message_net = MLP(msg_in, node_dim * 2, node_dim, n_layers=3, dropout=dropout)
        self.update_net  = MLP(node_dim * 2, node_dim * 2, node_dim, n_layers=2)
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x_new = self.update_net(torch.cat([x, out], dim=-1))
        return self.norm(x + self.dropout(x_new))  # residual connection

    def message(self, x_i, x_j, edge_attr):
        inp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_net(inp)


# ─── Physics-constrained property heads ───────────────────────────────────────
class BandGapHead(nn.Module):
    """Band gap ≥ 0 enforced via Softplus (smooth alternative to ReLU)."""
    def __init__(self, in_dim, dropout=0.1):
        super().__init__()
        self.net = MLP(in_dim, 128, 1, dropout=dropout)

    def forward(self, x):
        raw = self.net(x).squeeze(-1)
        return F.softplus(raw)  # Always ≥ 0. Softplus is differentiable at 0.


class FormationEnergyHead(nn.Module):
    """Formation energy is unconstrained (can be positive or negative)."""
    def __init__(self, in_dim, dropout=0.1):
        super().__init__()
        self.net = MLP(in_dim, 128, 1, dropout=dropout)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HullEnergyHead(nn.Module):
    """Energy above hull ≥ 0 (0 = on hull = thermodynamically stable)."""
    def __init__(self, in_dim, dropout=0.1):
        super().__init__()
        self.net = MLP(in_dim, 128, 1, dropout=dropout)

    def forward(self, x):
        return F.softplus(self.net(x).squeeze(-1))  # Always ≥ 0


# ─── Full PIGNet model ────────────────────────────────────────────────────────
class PIGNet(nn.Module):
    """
    Full Physics-Informed Graph Network.

    Args:
        hidden_dim:   Node embedding dimension (default 256)
        n_conv:       Number of message-passing layers (default 4)
        dropout:      Dropout rate for MC Dropout uncertainty (default 0.1)
    """
    def __init__(self, hidden_dim=256, n_conv=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # Input projection
        self.node_embed = nn.Sequential(
            nn.Linear(ATOM_DIM, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(BOND_DIM, hidden_dim),
            nn.SiLU(),
        )

        # Message-passing layers
        self.convs = nn.ModuleList([
            PIGNetConv(hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(n_conv)
        ])

        # Post-pool MLP for structure-level embedding
        self.structure_mlp = MLP(hidden_dim, hidden_dim * 2, hidden_dim, n_layers=3, dropout=dropout)

        # Property heads
        self.band_gap_head     = BandGapHead(hidden_dim, dropout)
        self.formation_head    = FormationEnergyHead(hidden_dim, dropout)
        self.hull_energy_head  = HullEnergyHead(hidden_dim, dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x:          Node features (N, 10)
            edge_index: Edge indices (2, E)
            edge_attr:  Edge (bond) features (E, 40)
            batch:      Batch assignment for each node (N,)
        Returns:
            dict with band_gap, formation_energy, energy_above_hull — each (B,)
        """
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)

        for conv in self.convs:
            h = conv(h, edge_index, e)

        # Global pool: atoms → single structure vector per graph
        h_graph = global_mean_pool(h, batch)  # (B, hidden_dim)
        h_graph = self.structure_mlp(h_graph)

        return {
            "band_gap":           self.band_gap_head(h_graph),
            "formation_energy":   self.formation_head(h_graph),
            "energy_above_hull":  self.hull_energy_head(h_graph),
        }

    @torch.no_grad()
    def predict_with_uncertainty(self, x, edge_index, edge_attr, batch, T=30):
        """
        Monte Carlo Dropout uncertainty estimation.
        Runs T stochastic forward passes with dropout ON, returns mean + std.

        Returns: dict with *_mean, *_std for each property
        """
        self.train()  # Enable dropout
        preds = {k: [] for k in ["band_gap", "formation_energy", "energy_above_hull"]}

        for _ in range(T):
            out = self.forward(x, edge_index, edge_attr, batch)
            for k, v in out.items():
                preds[k].append(v.cpu())

        self.eval()
        results = {}
        for k, vs in preds.items():
            stacked = torch.stack(vs)  # (T, B)
            results[f"{k}_mean"] = stacked.mean(dim=0)
            results[f"{k}_std"]  = stacked.std(dim=0)
        return results