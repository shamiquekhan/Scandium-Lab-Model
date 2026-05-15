"""
Compile best_model_v2.pt to TorchScript for fast production inference.
Run once after training: python -m scripts.compile_model
"""
import torch
from pathlib import Path
from scandium.models.pignet_v2 import PIGNetV2

CKPT_IN  = Path("backend/checkpoints/best_model_v2.pt")
CKPT_OUT = Path("backend/checkpoints/best_model_v2_scripted.pt")

ckpt  = torch.load(CKPT_IN, map_location="cpu")
cfg   = ckpt["config"]
model = PIGNetV2(cfg["hidden"], cfg["n_conv"], drop=0.0)  # drop=0 for inference
state = ckpt.get("ema_shadow") or ckpt["model_state"]
model.load_state_dict(state, strict=False)
model.eval()

# Create representative dummy input for tracing
N_ATOMS = 8; N_EDGES = 48
dummy_x         = torch.randn(N_ATOMS, 10)
dummy_edge_idx  = torch.randint(0, N_ATOMS, (2, N_EDGES))
dummy_edge_attr = torch.randn(N_EDGES, 56)
dummy_batch     = torch.zeros(N_ATOMS, dtype=torch.long)

with torch.no_grad():
    traced = torch.jit.trace(
        model,
        (dummy_x, dummy_edge_idx, dummy_edge_attr, dummy_batch),
        strict=False,
    )

traced.save(CKPT_OUT)
print(f"TorchScript model saved → {CKPT_OUT}")

# Quick sanity check: compare outputs
with torch.no_grad():
    out_orig   = model(dummy_x, dummy_edge_idx, dummy_edge_attr, dummy_batch)
    out_traced = traced(dummy_x, dummy_edge_idx, dummy_edge_attr, dummy_batch)
for k in out_orig:
    diff = (out_orig[k] - out_traced[k]).abs().max().item()
    print(f"  {k}: max_diff = {diff:.2e}")   # should be < 1e-5
