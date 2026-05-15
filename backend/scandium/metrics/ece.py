import numpy as np
import torch


def regression_ece(pred_std: np.ndarray, errors: np.ndarray, n_bins: int = 10) -> float:
    """
    Simple regression Expected Calibration Error (ECE) approximation.

    - pred_std: predicted standard deviations (N,)
    - errors: absolute errors |y_pred - y_true| (N,)

    The metric computes, per bin of predicted uncertainty, the difference between
    the mean predicted uncertainty and the empirical RMSE in that bin, then
    returns a weighted average of absolute differences.
    """
    assert pred_std.shape == errors.shape
    N = len(pred_std)
    if N == 0:
        return 0.0

    bins = np.linspace(pred_std.min(), pred_std.max(), n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (pred_std >= bins[i]) & (pred_std < bins[i + 1])
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        mean_pred = pred_std[mask].mean()
        rmse = np.sqrt((errors[mask] ** 2).mean())
        ece += n_bin * abs(mean_pred - rmse)
    return float(ece / N)


def torch_regression_ece(pred_std: torch.Tensor, errors: torch.Tensor, n_bins: int = 10) -> float:
    pred_std = pred_std.detach().cpu().numpy()
    errors = errors.detach().cpu().numpy()
    return regression_ece(pred_std, errors, n_bins=n_bins)
