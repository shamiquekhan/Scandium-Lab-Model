"""
PIGNet V2 — Production architecture.

Improvements over V1:
  1. Attention-gated messages (a_ij = sigmoid(Linear(h_i, h_j, e_ij)))
  2. Edge update network (e_ij' = e_ij + MLP(h_i, h_j, e_ij))
  3. Dual pooling: mean + max → projected → crystal vector
  4. 56-dim edge features (40 RBF + 16 angular)
  5. Softplus-constrained heads for physically bounded properties

These changes are transparent to the FastAPI layer —
the model still returns a dict {"band_gap", "formation_energy", "energy_above_hull"}.
The only change at serving time: edge_attr is now (E, 56) not (E, 40).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool


# ── Shared MLP building block ─────────────────────────────────────────────────

class MLP(nn.Module):
    """
    N-layer MLP with SiLU activations, optional dropout, and no final activation.
    Used for message networks, gate networks, and prediction heads.
    """

    def __init__(
        self,
        in_dim:  int,
        hid_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dropout:  float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hid_dim), nn.SiLU()]
        for _ in range(n_layers - 2):
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers += [nn.Linear(hid_dim, hid_dim), nn.SiLU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hid_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ── Upgraded message-passing layer ────────────────────────────────────────────

class PIGNetConvV2(MessagePassing):
    """
    Upgraded message-passing layer with:

    1. Edge update (runs BEFORE message passing):
       e_ij' = e_ij + Dropout(MLP(h_i || h_j || e_ij))

    2. Attention-gated messages:
       m_ij  = MLP(h_i || h_j || e_ij')
       a_ij  = Sigmoid(Linear → Linear → Sigmoid)(h_i || h_j || e_ij')
       send  = a_ij * m_ij     ← learned per-neighbor importance

    3. Residual node update:
       h_i'  = LayerNorm(h_i + Dropout(MLP(h_i || Σ a_ij·m_ij)))

    Returns updated (h, e) so edge state is propagated across layers.
    """

    def __init__(self, node_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__(aggr="add")   # sum-aggregate gated messages
        self.n_dim = node_dim
        self.e_dim = edge_dim

        inp_dim = node_dim * 2 + edge_dim

        # Edge update: learns to enrich edge context before message passing
        self.edge_net = MLP(inp_dim, node_dim, edge_dim, n_layers=2, dropout=dropout)

        # Message network: encodes what atom j sends to atom i
        self.msg_net  = MLP(inp_dim, node_dim * 2, node_dim, n_layers=3, dropout=dropout)

        # Attention gate: scalar in [0, 1] per (i, j) pair
        # Two-layer design: wide → narrow → sigmoid
        self.att_gate = nn.Sequential(
            nn.Linear(inp_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 1),
            nn.Sigmoid(),
        )

        # Node update after aggregation
        self.upd_net  = MLP(node_dim * 2, node_dim * 2, node_dim, n_layers=2)
        self.norm     = nn.LayerNorm(node_dim)
        self.drop     = nn.Dropout(dropout)

    def forward(
        self,
        x:          Tensor,      # (N, node_dim)
        edge_index: Tensor,      # (2, E)
        edge_attr:  Tensor,      # (E, edge_dim)
    ) -> tuple[Tensor, Tensor]:  # updated (h, e)

        src, dst = edge_index

        # ── 1. Update edge features ───────────────────────────────────────────
        # Gives edges richer context before they generate messages
        ctx   = torch.cat([x[src], x[dst], edge_attr], dim=-1)  # (E, inp_dim)
        new_e = edge_attr + self.drop(self.edge_net(ctx))        # residual

        # ── 2. Propagate with updated edges ──────────────────────────────────
        agg = self.propagate(edge_index, x=x, edge_attr=new_e)  # (N, node_dim)

        # ── 3. Residual node update ───────────────────────────────────────────
        h   = self.norm(x + self.drop(
            self.upd_net(torch.cat([x, agg], dim=-1))
        ))

        return h, new_e

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        """
        x_i: features of destination atom i  (E, node_dim)
        x_j: features of source atom j        (E, node_dim)
        edge_attr: updated edge features       (E, edge_dim)
        """
        inp = torch.cat([x_i, x_j, edge_attr], dim=-1)  # (E, inp_dim)
        msg = self.msg_net(inp)                          # (E, node_dim)
        att = self.att_gate(inp)                         # (E, 1) in [0, 1]
        return att * msg                                 # soft-gated message


# ── Full PIGNet V2 model ──────────────────────────────────────────────────────

ATOM_FEAT_DIM: int = 10   # from featurize.atom_features
EDGE_FEAT_DIM: int = 56   # 40 RBF + 16 angular (v2)


class PIGNetV2(nn.Module):
    """
    Full PIGNet V2.

    Input graph (from featurize_v2.structure_to_graph_v2):
      x:          (N, 10)  atom features
      edge_index: (2, E)
      edge_attr:  (E, 56)  RBF + angular features
      batch:      (N,)     batch assignment

    Output dict:
      band_gap:          (B,) eV          — Softplus constrained ≥ 0
      formation_energy:  (B,) eV/atom     — unconstrained (can be negative)
      energy_above_hull: (B,) eV/atom     — Softplus constrained ≥ 0
    """

    def __init__(
        self,
        hidden_dim: int   = 256,
        n_conv:     int   = 4,
        dropout:    float = 0.1,
    ):
        super().__init__()

        # ── Input embeddings ─────────────────────────────────────────────────
        self.node_emb = nn.Sequential(
            nn.Linear(ATOM_FEAT_DIM, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_emb = nn.Sequential(
            nn.Linear(EDGE_FEAT_DIM, hidden_dim),
            nn.SiLU(),
        )

        # ── Message-passing stack ─────────────────────────────────────────────
        self.convs = nn.ModuleList([
            PIGNetConvV2(hidden_dim, hidden_dim, dropout)
            for _ in range(n_conv)
        ])

        # ── Dual pooling: mean + max → project back to hidden_dim ────────────
        self.pool_proj = MLP(
            hidden_dim * 2, hidden_dim * 2, hidden_dim,
            n_layers=3, dropout=dropout,
        )

        # ── Physics-constrained prediction heads ──────────────────────────────
        # band_gap ≥ 0 by physics  → Softplus activation
        self.bg_head   = nn.Sequential(
            MLP(hidden_dim, 128, 1, dropout=dropout),
            nn.Softplus(),
        )
        # formation_energy can be negative → no constraint
        self.fe_head   = MLP(hidden_dim, 128, 1, dropout=dropout)
        # energy_above_hull ≥ 0 by definition → Softplus
        self.hull_head = nn.Sequential(
            MLP(hidden_dim, 128, 1, dropout=dropout),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x:          Tensor,
        edge_index: Tensor,
        edge_attr:  Tensor,
        batch:      Tensor,
    ) -> dict[str, Tensor]:

        # ── Embed inputs ──────────────────────────────────────────────────────
        h = self.node_emb(x)
        e = self.edge_emb(edge_attr)

        # ── Message passing ───────────────────────────────────────────────────
        for conv in self.convs:
            h, e = conv(h, edge_index, e)

        # ── Dual pooling ──────────────────────────────────────────────────────
        h_mean  = global_mean_pool(h, batch)                          # (B, H)
        h_max   = global_max_pool(h,  batch)                          # (B, H)
        h_graph = self.pool_proj(torch.cat([h_mean, h_max], dim=-1)) # (B, H)

        # ── Property heads ────────────────────────────────────────────────────
        return {
            "band_gap":          self.bg_head(h_graph).squeeze(-1),
            "formation_energy":  self.fe_head(h_graph).squeeze(-1),
            "energy_above_hull": self.hull_head(h_graph).squeeze(-1),
        }

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def mc_dropout_predict(
        self,
        x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor,
        T: int = 30,
    ) -> dict[str, dict[str, Tensor]]:
        """
        Monte Carlo Dropout uncertainty estimate.
        Runs T forward passes with dropout active.
        Used by the ensemble service as a cheaper uncertainty estimate
        when running fewer than 5 ensemble members.
        """
        self.train()   # activate dropout
        samples = {k: [] for k in ["band_gap", "formation_energy", "energy_above_hull"]}

        for _ in range(T):
            preds = self.forward(x, edge_index, edge_attr, batch)
            for k, v in preds.items():
                samples[k].append(v.cpu())

        self.eval()

        results = {}
        for k, vs in samples.items():
            stacked     = torch.stack(vs)               # (T, B)
            results[k]  = {
                "mean": stacked.mean(dim=0),
                "std":  stacked.std(dim=0),
                "q05":  stacked.quantile(0.05, dim=0),
                "q95":  stacked.quantile(0.95, dim=0),
            }
        return results
