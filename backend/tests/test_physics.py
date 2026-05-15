import torch
import pytest
from scandium.physics.constraints import physics_constrained_loss, check_violations, PhysicsConfig


def test_physics_loss_bandgap_penalty():
    cfg = PhysicsConfig()
    preds = {
        "band_gap": torch.tensor([-0.5, 1.0]),
        "formation_energy": torch.tensor([-2.0, -1.0]),
        "energy_above_hull": torch.tensor([0.0, 0.0])
    }
    targets = torch.tensor([
        [0.0, -2.0, 0.0],
        [1.0, -1.0, 0.0]
    ])
    
    losses = physics_constrained_loss(preds, targets, cfg)
    # The first prediction has a negative band_gap (-0.5), so penalty should be > 0
    assert float(losses["physics_penalty"]) > 0.0


def test_check_violations_negative_bandgap():
    violations = check_violations({"band_gap": -0.3, "formation_energy": -2.0, "energy_above_hull": 0.0})
    assert len(violations) > 0
    assert any("BAND_GAP_NEGATIVE" in v for v in violations)


def test_check_violations_valid():
    violations = check_violations({"band_gap": 1.5, "formation_energy": -2.1, "energy_above_hull": 0.0})
    assert len(violations) == 0


def test_check_violations_unstable():
    violations = check_violations({"band_gap": 1.5, "formation_energy": -1.0, "energy_above_hull": 0.5})
    assert len(violations) > 0
    assert any("STRUCTURE_UNSTABLE" in v for v in violations)