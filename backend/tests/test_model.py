import torch
import pytest


def test_model_forward_shape(small_model, dummy_batch):
    preds = small_model(dummy_batch)
    assert "band_gap" in preds
    assert "formation_energy" in preds
    assert preds["band_gap"].shape == (1,)


def test_band_gap_non_negative(small_model, dummy_batch):
    """PIGNet band gap head uses softplus — must always be >= 0."""
    for _ in range(50):
        dummy_batch.x = torch.randn_like(dummy_batch.x)
        preds = small_model(dummy_batch)
        assert (preds["band_gap"] >= 0).all(), "Band gap prediction went negative!"


def test_model_parameter_count(small_model):
    n = small_model.num_parameters
    assert n > 0
    print(f"\nModel parameters: {n:,}")


def test_gradient_flow(small_model, dummy_batch):
    small_model.train()
    preds = small_model(dummy_batch)
    loss = sum(v.mean() for v in preds.values())
    loss.backward()
    for name, param in small_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"