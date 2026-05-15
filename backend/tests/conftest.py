import pytest
import torch
from torch_geometric.data import Data
from scandium.models.pignet import PIGNet
from scandium.data.featurisers import AtomFeaturiser
from scandium.data.graph_builder import GraphBuilder


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")

@pytest.fixture(scope="session")
def featuriser():
    return AtomFeaturiser()

@pytest.fixture(scope="session")
def graph_builder():
    return GraphBuilder(cutoff=5.0)

@pytest.fixture(scope="session")
def small_model():
    return PIGNet(
        node_in_dim=99, edge_in_dim=40,
        hidden_dim=64, n_layers=2,
        targets=["band_gap", "formation_energy"],
    )

@pytest.fixture
def dummy_batch():
    """Minimal valid PyG batch for unit testing."""
    n_atoms, n_edges = 10, 40
    data = Data(
        x=torch.randn(n_atoms, 99),
        edge_index=torch.randint(0, n_atoms, (2, n_edges)),
        edge_attr=torch.randn(n_edges, 40),
        batch=torch.zeros(n_atoms, dtype=torch.long),
        num_nodes=n_atoms,
        is_stable=torch.ones(1),
        y=torch.tensor([1.5]),
        y_raw=torch.tensor([1.5]),
    )
    return data