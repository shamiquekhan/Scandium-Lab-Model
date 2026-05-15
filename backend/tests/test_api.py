import pytest
from fastapi.testclient import TestClient
import os

os.environ["VALID_API_KEYS"] = "test_key"
os.environ["MODEL_CHECKPOINT_DIR"] = "checkpoints"

# Only run API tests if a checkpoint exists
pytestmark = pytest.mark.skipif(
    not os.path.exists("checkpoints/best_model.pt"),
    reason="No trained checkpoint found — run training first"
)


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_predict_requires_auth(client):
    r = client.post("/predict/", json={})
    assert r.status_code == 403


def test_predict_cif(client):
    # Minimal Si CIF for testing
    cif_content = """
data_Si
_cell_length_a   5.43
_cell_length_b   5.43
_cell_length_c   5.43
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
_symmetry_space_group_name_H-M  'F d -3 m'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
Si Si 0.000 0.000 0.000
Si Si 0.250 0.250 0.250
"""
    payload = {
        "structure": {"structure_data": cif_content, "format": "cif"},
        "properties": ["band_gap"]
    }
    r = client.post("/predict/", json=payload, headers={"X-API-Key": "test_key"})
    assert r.status_code == 200
    body = r.json()
    assert "band_gap" in body["predictions"]
    assert body["predictions"]["band_gap"]["value"] >= 0.0