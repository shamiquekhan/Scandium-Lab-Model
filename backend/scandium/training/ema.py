import torch

class EMA:
    """
    Exponential Moving Average of model weights.
    During training: update shadow weights every step.
    At validation / inference: apply shadow weights for prediction.
    After inference: restore training weights.
    decay=0.999 → shadow weights update slowly, ignoring noisy batches.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model  = model
        self.decay  = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self):
        """Call after every optimizer.step()."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay)

    def apply(self):
        """Swap in shadow weights (for eval / inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Swap training weights back after eval."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
