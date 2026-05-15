# Scandium Labs: PIGNet V2 Architecture

This document outlines the detailed architecture of the Physics-Informed Graph Neural Network (PIGNet V2) used for property prediction in Scandium Labs.

## 1. Featurization

Crystal structures are converted into PyTorch Geometric (PyG) directed graphs:
- **Nodes ($N$)**: Atoms in the unit cell. Initial node features are embeddings of the atomic numbers (Z).
- **Edges ($E$)**: Bonds between atoms within a specific cutoff radius (default: 6.0 Å).
- **Edge Features**: 56-dimensional tensor per edge.
  - **Radial Basis Functions (RBF)**: 40 dimensions representing the pairwise interatomic distances.
  - **Angular Features**: 16 dimensions capturing the 3-body coordination geometry using Fourier expansions of bond angles.

## 2. Core Architecture (`PIGNetV2`)

The model contains ~7.3 Million trainable parameters and is built to be robust, scale-invariant, and physics-constrained.

### 2.1 Embedding Layers
- **Node Embedding**: Maps raw atomic numbers (1-100) into a continuous $D_{node} = 128$ dimensional space.

### 2.2 Message Passing (PIGNetConvV2)
The network uses 5 sequential Graph Neural Network layers. Each layer performs:
1. **Message Construction**: Concatenates source node, target node, and edge features.
2. **Attention Gating**: A Multi-Layer Perceptron (MLP) calculates an attention score (Sigmoid) for each message, dynamically weighting the importance of different neighbor interactions.
3. **Message Aggregation**: Messages are sum-aggregated across neighbors.
4. **Node Update**: An MLP updates the node states using the aggregated messages and previous node states, stabilized with LayerNorm and SiLU activations.
5. **Edge Update**: A dedicated Edge MLP updates the edge features (distance + angular features) at every layer to allow the network to learn complex geometric transformations deep in the network.

### 2.3 Global Pooling
To convert atom-level representations to a crystal-level representation, PIGNet V2 uses **Dual Pooling**:
- `global_mean_pool`: Captures intensive properties (average per-atom behavior).
- `global_max_pool`: Captures extreme local environments (defects, specific heavy atoms).
These are concatenated into a 256-dimensional global state vector.

## 3. Multi-Task Property Heads

The final 256-dimensional graph embedding is fed into three independent prediction heads. Each head enforces specific physics constraints:

1. **Band Gap ($E_g$)**
   - **Constraint**: Must be strictly non-negative ($\ge 0$ eV).
   - **Implementation**: 3-layer MLP ending in a `Softplus` activation.
2. **Formation Energy ($E_f$)**
   - **Constraint**: Can be positive or negative.
   - **Implementation**: 3-layer MLP ending in a linear layer (no activation).
3. **Energy Above Hull ($E_{hull}$)**
   - **Constraint**: Must be strictly non-negative ($\ge 0$ eV/atom).
   - **Implementation**: 3-layer MLP ending in a `Softplus` activation.

## 4. Uncertainty Quantification (UQ)

PIGNet V2 supports UQ via two primary mechanisms:
1. **Deep Ensembles**: Training 5 identical models with different random seeds. The variance of their predictions provides epistemic uncertainty.
2. **MC Dropout**: The model includes a `mc_dropout_predict()` method. At inference time, dropout layers remain active to generate a distribution of predictions from a single forward pass, providing a fast alternative to full ensembling.
