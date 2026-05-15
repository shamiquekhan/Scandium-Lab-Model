"""Combined loss: prediction MSE + physics constraint penalties."""

import torch
import torch.nn as nn
from torch import Tensor
from ..physics.constraints import PhysicsConstraintLoss


class PIGNetLoss(nn.Module):
    def __init__(self, constraint_cfg: dict):
        super().__init__()
        self.mse       = nn.MSELoss()
        self.huber     = nn.HuberLoss(delta=1.0)  # robust to outliers
        self.physics   = PhysicsConstraintLoss(**constraint_cfg)

    def forward(
        self,
        preds:  dict[str, Tensor],
        target: str,
        y:      Tensor,
        batch,
    ) -> tuple[Tensor, dict[str, Tensor]]:

        # Primary prediction loss (Huber for robustness)
        pred_loss = self.huber(preds[target], y)

        # Physics constraint losses
        constraint_losses = self.physics(preds, batch)

        total = pred_loss + constraint_losses["constraint_total"]

        loss_dict = {"prediction": pred_loss, **constraint_losses, "total": total}
        return total, loss_dict