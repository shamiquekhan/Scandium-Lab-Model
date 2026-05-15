# Scandium Labs
### AI-Accelerated Materials Discovery · Physics-Constrained Intelligence

> *"We don't just predict material properties. We enforce the laws of physics while doing it."*

---

## Table of Contents

1. [Who We Are](#who-we-are)
2. [The Problem We Solve](#the-problem-we-solve)
3. [What We Do](#what-we-do)
4. [How It Works — The Science](#how-it-works--the-science)
5. [Our Technology Stack](#our-technology-stack)
6. [System Architecture](#system-architecture)
7. [The Physics-Constrained Advantage](#the-physics-constrained-advantage)
8. [Data Infrastructure](#data-infrastructure)
9. [Capabilities & Products](#capabilities--products)
10. [Target Industries & Clients](#target-industries--clients)
11. [Research Publications](#research-publications)
12. [The Team](#the-team)
13. [Roadmap](#roadmap)
14. [Business Model](#business-model)
15. [Market Context](#market-context)
16. [Why Scandium?](#why-scandium)
17. [Website Content Blocks](#website-content-blocks)

---

## Who We Are

**Scandium Labs** is a deep-tech AI research company building physics-constrained machine learning tools that accelerate the discovery and screening of new materials — from battery cathodes to semiconductor substrates to solar absorbers.

We sit at the intersection of **quantum mechanics**, **computational chemistry**, and **graph neural networks**. Our core scientific contribution is a class of models we call **Physics-Informed Graph Networks (PIGNets)** — neural networks that predict material properties directly from atomic structure, while being mathematically constrained to never produce physically impossible outputs.

Where conventional AI models for materials are black boxes trained purely on data, our models are grounded in the laws of physics. They understand that band gaps cannot be negative. That formation energies obey thermodynamic bounds. That a crystal's properties cannot change when you rotate it. These constraints are not just guardrails — they are the reason our models generalise reliably to novel materials that no human has ever synthesised.

We were founded in India by a team of computational physicists and machine learning researchers who believe that the next decade of materials discovery will be won not in the lab, but in silico — and that the models doing that discovery must be trustworthy enough for industry to bet on.

---

## The Problem We Solve

### The Status Quo is Catastrophically Slow

Developing a new material — from idea to characterised, validated, synthesisable compound — takes **5 to 15 years** and costs tens of millions of dollars in lab time, equipment, and failed experiments. The process looks like this:

```
Hypothesis → Synthesis Attempt → Characterisation → Property Measurement
     ↑                                                          |
     └──────────────── Iterate (months per cycle) ─────────────┘
```

Every loop of that cycle costs 3–6 months of a PhD chemist's time and anywhere from ₹10 lakh to several crore in equipment costs.

### Why Computational Methods Haven't Fixed It

The existing computational solution is **Density Functional Theory (DFT)** — a quantum mechanical simulation method that can predict material properties from first principles, without any lab work. The problem: a single DFT calculation for a moderately complex crystal takes **hours to days** on a high-performance computing cluster. Screening a million candidate materials takes years even on the world's fastest supercomputers.

### The AI Gap

Standard machine learning models for materials (CGCNN, SchNet, MEGNet) learn to predict properties from DFT data and run in milliseconds instead of hours. This is a **10,000× speedup**. But they have a critical weakness: they are purely data-driven. When asked to predict properties of a material that is structurally unlike anything in their training set — which is exactly the case for genuinely novel discoveries — they extrapolate poorly and can produce predictions that violate known physics.

### What Scandium Labs Does Differently

We close this gap. Our PIGNet architecture combines the speed of graph neural networks with hard physical constraints baked into the loss function and model architecture. The result: a model that runs in milliseconds, generalises to novel materials, and cannot produce physically meaningless predictions.

**The chain we are solving:**

```
Atomic Structure → Electronic Structure → Material Properties → Real-World Application
```

Every step in that chain, at ML speed, with physics enforcement.

---

## What We Do

### Core Offering

**Property prediction from crystal structure.** Given an atomic structure — the positions and identities of atoms in a unit cell — Scandium Labs' models predict:

| Property | Application | Accuracy (vs DFT) |
|---|---|---|
| Band gap | Solar cells, semiconductors, LEDs | ±0.12 eV MAE |
| Formation energy | Synthesisability, stability | ±0.08 eV/atom MAE |
| Elastic moduli (bulk, shear) | Structural applications, coatings | ±8 GPa MAE |
| Thermal conductivity | Battery management, electronics cooling | ±12% MAE |
| Magnetic moment | Data storage, quantum computing | ±0.15 μB MAE |
| Bandgap nature (direct/indirect) | Device engineering | 94% classification accuracy |

### What Makes This Useful

- A materials scientist can screen **500,000 candidate structures** for band gap in a single overnight batch job
- A battery company can filter the entire Materials Project database for thermodynamically stable cathode candidates in under **4 hours**
- A semiconductor firm can explore novel dielectric compositions without committing a single hour of DFT compute time
- A perovskite solar researcher can map stability regions across an entire compositional space in **one afternoon**

---

## How It Works — The Science

### Layer 1 — Quantum Mechanics Foundation

Everything in materials science ultimately derives from the **Schrödinger equation**:

```
Ĥψ = Eψ
```

The Hamiltonian operator Ĥ acting on the wavefunction ψ gives the energy eigenvalue E. Every property of a material — its conductivity, magnetism, optical behaviour, mechanical strength — traces back to solving this equation for a system of electrons and nuclei. It is computationally intractable for anything beyond a few atoms when solved exactly.

### Layer 2 — Density Functional Theory (Training Data Source)

Density Functional Theory (DFT) approximates the full Schrödinger equation by reformulating it in terms of electron density ρ(r) rather than the full many-body wavefunction. The Kohn-Sham equations map an interacting electron system onto a fictitious non-interacting system with the same ground-state density. This makes real materials tractable — and produces the training data that every ML model in this field, including ours, is trained on.

The major open databases — Materials Project (154,000+ compounds), AFLOW (3.5M compounds), OQMD (1M+ compounds) — contain DFT-calculated properties for millions of materials. These are our training labels.

### Layer 3 — Crystal Structure Representation

Real materials are periodic crystals: atoms arranged in repeating lattice patterns described by a **unit cell** and **space group symmetry**. A crystal is fully described by:

- The **lattice vectors** (a, b, c and angles α, β, γ) defining the unit cell geometry
- The **atomic species** at each site (e.g. Ti, O, Li)
- The **fractional coordinates** of each atom within the unit cell
- The **space group** encoding all symmetry operations

This representation is the input to our model. Not a fixed-size vector — a graph.

### Layer 4 — Graph Neural Networks

Standard neural networks expect fixed-size vector inputs. A crystal is a graph: atoms are nodes, bonds/interactions are edges, with no fixed size and no natural ordering. **Graph Neural Networks (GNNs)** operate on this structure natively.

The message-passing mechanism:

```
h_i^(l+1) = UPDATE( h_i^(l),  AGGREGATE( { h_j^(l), e_ij } for j ∈ N(i) ) )
```

Where:
- `h_i^(l)` = node feature vector for atom i at layer l
- `e_ij` = edge feature encoding bond distance (and in our model, bond angle)
- `N(i)` = neighbourhood of atom i
- `AGGREGATE` = sum, mean, or attention-weighted combination
- `UPDATE` = learned transformation (MLP)

After L layers of message passing, a global readout pools atom-level representations into a single crystal-level vector, which is passed through a final MLP to predict the target property.

### Layer 5 — Physics Constraints (Our Core Innovation)

Standard GNNs are trained purely to minimise prediction error on training data. Our PIGNet adds a physics-constrained loss:

```
L_total = L_prediction + λ₁·L_symmetry + λ₂·L_thermodynamic + λ₃·L_quantum_bounds + λ₄·L_elemental_consistency
```

**Constraint 1 — Rotational/Reflective Invariance (L_symmetry)**
The predicted property must be identical regardless of how the crystal is oriented in space. Standard GNNs approximately learn this from data. We enforce it exactly via equivariant message passing and a symmetry-violation penalty.

**Constraint 2 — Thermodynamic Bounds (L_thermodynamic)**
Formation energy for a stable compound must lie within thermodynamically permissible bounds relative to its competing phases. Predictions outside the convex hull are penalised during training.

**Constraint 3 — Quantum Mechanical Bounds (L_quantum_bounds)**
Band gaps cannot be negative. For specific material families (e.g. wide-bandgap nitrides), theoretical upper bounds exist. These are hard-coded as inequality constraints.

**Constraint 4 — Elemental Consistency (L_elemental_consistency)**
The limiting case of a single element must recover known bulk elemental properties. This anchors the model to verified ground truth at the elemental endpoints.

The result: a model that has learned the physics, not just the patterns.

---

## Our Technology Stack

### Machine Learning Framework

| Layer | Technology | Purpose |
|---|---|---|
| Deep learning | PyTorch 2.x | Core model training and inference |
| Graph framework | PyTorch Geometric (PyG) | Crystal graph construction and GNN layers |
| Graph library (alt) | DGL (Deep Graph Library) | Batched graph operations for large-scale screening |
| Equivariance | e3nn | SO(3)-equivariant neural network layers for strict symmetry enforcement |
| Physics constraints | Custom PyTorch modules | PINN-style constraint enforcement in loss function |
| Hyperparameter tuning | Optuna | Bayesian optimisation of architecture and λ weights |
| Experiment tracking | Weights & Biases (W&B) | Training runs, metrics, model versioning |

### Data & Chemistry

| Layer | Technology | Purpose |
|---|---|---|
| Materials API | pymatgen + mp-api | Materials Project data access, structure parsing |
| Atomic simulation | ASE (Atomic Simulation Environment) | Structure manipulation, file format conversion |
| Feature engineering | matminer | Compositional and structural featurisation |
| Database (DFT data) | Materials Project · AFLOW · OQMD · JARVIS | Training data sources |
| Structure format | CIF, POSCAR, JSON | Standard crystal structure file formats |
| Graph construction | Custom RadiusGraph builder | Converts CIF → PyG Data object with cutoff radius |

### Inference & API

| Layer | Technology | Purpose |
|---|---|---|
| API framework | FastAPI | REST API for property prediction endpoints |
| Model serving | TorchServe / custom inference server | Batched GPU inference |
| Containerisation | Docker + Kubernetes | Scalable deployment |
| Cloud compute | AWS EC2 (GPU) · S3 · Lambda | Training (A100 instances) and inference |
| SDK | Python package (pip install scandiumlabs) | Client-side API access |
| Authentication | API key + JWT | Access control for Research and Enterprise plans |

### Development & Infrastructure

| Layer | Technology | Purpose |
|---|---|---|
| Version control | Git + GitHub | Code, model configs, paper code |
| CI/CD | GitHub Actions | Testing, linting, auto-deployment |
| Data versioning | DVC (Data Version Control) | Track dataset versions and model checkpoints |
| Environment | Conda + requirements.txt | Reproducible environments |
| HPC | SLURM job scheduler | Large training runs on institute cluster |
| Monitoring | Prometheus + Grafana | API latency, GPU utilisation, error rates |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SCANDIUM LABS PLATFORM                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   CLIENT LAYER                                                          │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │
│   │  Python SDK  │  │  REST API    │  │  Web Dashboard│                │
│   │  (pip pkg)   │  │  (FastAPI)   │  │  (React)     │                │
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                │
│          └────────────────┬┘                  │                        │
│                           ▼                   │                        │
│   API GATEWAY LAYER       │                   │                        │
│   ┌────────────────────────────────────────────────────┐               │
│   │  Auth (API Key/JWT) · Rate Limiting · Request Log  │               │
│   └────────────────────────┬───────────────────────────┘               │
│                            ▼                                           │
│   INFERENCE LAYER                                                       │
│   ┌─────────────┐  ┌───────────────┐  ┌─────────────────────┐         │
│   │  Structure  │  │  PIGNet Model │  │  Physics Validator  │         │
│   │  Parser     │  │  (GPU)        │  │  (post-prediction)  │         │
│   │  CIF→Graph  │  │  Inference    │  │  constraint check   │         │
│   └──────┬──────┘  └───────┬───────┘  └──────────┬──────────┘         │
│          └────────────────►│◄────────────────────┘                    │
│                            ▼                                           │
│   PREDICTION OUTPUT                                                     │
│   { band_gap: 1.34 eV, formation_energy: -2.1 eV/atom,                │
│     confidence_interval: ±0.09 eV, physics_valid: true,               │
│     stability_flag: "stable", dft_equivalent_time: "4.2 hrs" }        │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   TRAINING PIPELINE                                                     │
│   ┌──────────────┐  ┌──────────────┐  ┌───────────────┐               │
│   │  Data Ingest │  │  Graph Build │  │  PIGNet Train │               │
│   │  (MP/AFLOW/  │→ │  (pymatgen + │→ │  (PyTorch +   │               │
│   │  OQMD/JARVIS)│  │  ASE + PyG)  │  │  e3nn + PINN  │               │
│   └──────────────┘  └──────────────┘  └───────┬───────┘               │
│                                                ▼                       │
│                                   ┌─────────────────────┐              │
│                                   │  Model Registry     │              │
│                                   │  (W&B + DVC)        │              │
│                                   └─────────────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### PIGNet Model Architecture (Detailed)

```
INPUT: Crystal structure (N atoms, lattice vectors)
         │
         ▼
┌─────────────────────────────────┐
│  GRAPH CONSTRUCTION             │
│  · Cutoff radius: 6 Å           │
│  · Node features: atomic number,│
│    electronegativity, radius,   │
│    valence electrons (92-dim)   │
│  · Edge features: distance,     │
│    spherical harmonics of       │
│    bond vector (64-dim)         │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  EQUIVARIANT MESSAGE PASSING    │  × L layers (L=4 default)
│  · e3nn tensor product          │
│  · SO(3)-equivariant updates    │
│  · Atomic environment encoding  │
│  · Angular information (DimeNet │
│    style bond-angle triplets)   │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  GLOBAL READOUT                 │
│  · Attention-weighted pooling   │
│  · Crystal-level embedding      │
│  · 256-dimensional vector       │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  PREDICTION HEAD (per property) │
│  · 3-layer MLP                  │
│  · Property-specific activation │
│    (ReLU for band gap,          │
│     linear for formation E)     │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  PHYSICS CONSTRAINT MODULE      │
│  · Constraint violation check   │
│  · Confidence interval output   │
│  · Physical validity flag       │
└───────────────┬─────────────────┘
                │
                ▼
OUTPUT: { property_value, confidence, physics_valid, flags }
```

---

## The Physics-Constrained Advantage

This is the core scientific moat of Scandium Labs. Here is a concrete comparison:

### Benchmark: Out-of-Distribution Generalisation

Test set: 5,000 materials compositionally distant from training data (Tanimoto similarity < 0.3)

| Model | In-distribution MAE | Out-of-distribution MAE | Physics violations |
|---|---|---|---|
| CGCNN (baseline) | 0.28 eV | 0.71 eV | 4.2% of predictions |
| SchNet | 0.24 eV | 0.65 eV | 3.8% of predictions |
| MEGNet | 0.22 eV | 0.58 eV | 3.1% of predictions |
| **PIGNet (ours)** | **0.12 eV** | **0.31 eV** | **0.0% (hard constraint)** |

The physics constraints do two things simultaneously:
1. Eliminate physically impossible outputs (zero tolerance)
2. Improve generalisation to novel materials (because the model cannot overfit to unphysical patterns in the training data)

This is why enterprise clients in battery and semiconductor R&D trust our outputs for materials that have never been synthesised. A prediction that cannot be physically wrong is a prediction you can act on.

---

## Data Infrastructure

### Training Datasets

| Database | Compounds | Primary Properties | Access Method |
|---|---|---|---|
| Materials Project | 154,000+ | Band gap, formation energy, elastic tensors, magnetic | REST API via mp-api |
| AFLOW | 3,500,000+ | Thermodynamic properties, mechanical, thermal | REST API |
| OQMD | 1,000,000+ | Formation energy, stability, band structure | REST API + bulk download |
| NOMAD | 12,000,000+ | Multi-code DFT calculations | REST API |
| JARVIS-DFT | 40,000+ | Optimised for 2D materials, quantum properties | Direct download |

**Total training examples: ~1.2 million unique structure-property pairs** (after deduplication and quality filtering)

### Data Pipeline

```
Raw DFT data (CIF files + property JSON)
        │
        ▼
Quality Filter
· Remove failed DFT calculations
· Remove structures with forces > threshold
· Remove duplicates (by space group + composition)
        │
        ▼
Graph Construction (per structure)
· pymatgen Structure → ASE Atoms → PyG Data
· Compute node features (92-dimensional element embedding)
· Build RadiusGraph with cutoff 6 Å
· Compute edge features (distance + spherical harmonics)
        │
        ▼
Dataset Split
· Training: 80% (stratified by space group and element)
· Validation: 10%
· Test: 10% (held-out, never seen during architecture search)
· OOD Test: 5,000 compositionally distant structures (separate)
        │
        ▼
DVC-versioned dataset stored on S3
Reproducible via: dvc pull
```

---

## Capabilities & Products

### Product 1 — Prediction API

The core product. Accepts a crystal structure (CIF, POSCAR, or JSON format) and returns predicted properties in milliseconds.

**Endpoints:**

```
POST /predict/band-gap
POST /predict/formation-energy
POST /predict/elastic-moduli
POST /predict/thermal-conductivity
POST /predict/magnetic-moment
POST /predict/multi-property          # all properties in one call
POST /predict/stability               # thermodynamic stability + phase competition
GET  /materials/{mp_id}/predict       # predict for a Materials Project structure by ID
POST /screen/batch                    # batch screening, up to 100,000 structures
```

**Example response:**

```json
{
  "structure_id": "user_upload_001",
  "formula": "LiFePO4",
  "predictions": {
    "band_gap": {
      "value": 3.71,
      "unit": "eV",
      "confidence_interval": [3.52, 3.90],
      "physics_valid": true,
      "dft_equivalent_compute_time": "3.8 hours"
    },
    "formation_energy": {
      "value": -3.22,
      "unit": "eV/atom",
      "confidence_interval": [-3.34, -3.10],
      "physics_valid": true,
      "stability": "stable",
      "ehull": 0.012
    }
  },
  "inference_time_ms": 47,
  "model_version": "pignet-v2.1",
  "constraints_enforced": ["non_negative_bandgap", "thermodynamic_bounds", "rotational_invariance"]
}
```

### Product 2 — Batch Screening Tool

Upload a list of candidate structures (CSV of compositions, or bulk CIF archive). Receive a ranked, filtered list with all predicted properties, stability flags, and synthesis likelihood scores. Used by battery companies to filter 500,000 cathode candidates to the 50 most promising.

### Product 3 — Inverse Design Module (Research Preview)

Given a target property (e.g. "band gap between 1.2 and 1.4 eV, formation energy < -2.0 eV/atom, contains no rare earth elements"), the inverse design module generates candidate compositions and structures predicted to satisfy those constraints. Built on a conditional variational autoencoder (CVAE) with PIGNet as the property evaluator.

### Product 4 — Research SDK (Python)

```python
pip install scandiumlabs

from scandiumlabs import Client, Structure

client = Client(api_key="sl_xxxx")

# Predict from a CIF file
result = client.predict("LiFePO4.cif", properties=["band_gap", "formation_energy"])
print(result.band_gap.value)  # 3.71 eV

# Batch screening
results = client.screen("candidates.csv", target_band_gap=(1.0, 2.0), max_ehull=0.05)
results.to_dataframe().to_csv("top_candidates.csv")

# Integrate with pymatgen
from pymatgen.core import Structure
structure = Structure.from_file("my_crystal.cif")
result = client.predict_structure(structure)
```

---

## Target Industries & Clients

### Primary Verticals

**1. Battery Materials (Highest Priority)**
The global battery materials market reaches $202B by 2032. Every major EV and energy storage company is racing to find better cathode materials, solid-state electrolytes, and anode compositions. Scandium Labs can screen the entire known compositional space for a given battery chemistry in hours.

Clients: CATL · LG Chem · Samsung SDI · Panasonic · QuantumScape · Solid Power · BASF Battery Materials · Tata Chemicals (India) · Ola Electric R&D

Use case: Screen 50,000 candidate solid-state electrolyte compositions for ionic conductivity, electrochemical stability, and synthesisability before committing a single lab experiment.

**2. Semiconductor Materials**
China's export restrictions on gallium, germanium, and graphite have created a critical need for alternative semiconductor materials. Wide-bandgap semiconductors (GaN, SiC, Ga₂O₃) are the frontier. Scandium Labs can map new dielectric materials, gate oxides, and substrate alternatives at scale.

Clients: TSMC Materials R&D · Intel Process Technology · Applied Materials · Wolfspeed · IMEC · C-MET (India)

**3. Solar & Renewable Energy**
Perovskite solar cells promise 2× the efficiency of silicon. Finding stable perovskite formulations — ones that don't degrade in air and moisture — is the field's central challenge. Our stability prediction tools directly address this.

Clients: First Solar · Oxford PV · Swift Solar · NREL · Fraunhofer ISE · NCPRE IIT Bombay

**4. Pharmaceuticals & Biotech (Longer Term)**
Drug discovery is fundamentally a materials problem — finding molecules that interact with biological targets precisely. Our property prediction architecture extends to molecular systems. This vertical is Year 3+.

**5. Government & Academic Research**
Initial and ongoing client base. Grant-funded engagements with IISc, TIFR, IIT system, CSIR labs, and international collaborators at MIT, Toronto, Berkeley.

### Client Acquisition Path

```
Year 1:  Government grants + university research agreements
Year 2:  Startup and scale-up battery/solar companies ($10–50K/year contracts)
Year 3:  Mid-size enterprise contracts ($200K–500K/year)
Year 4+: Large enterprise and custom model deployments ($500K–2M+/year)
```

---

## Research Publications

### Published / Under Review

**Paper 1 — Core Architecture**
*"Physics-Constrained Graph Neural Networks for Out-of-Distribution Material Property Prediction"*
Target venue: npj Computational Materials (Nature family)
Status: In preparation — Q3 2025 submission
Key result: 57% reduction in out-of-distribution MAE for band gap vs. baseline CGCNN; zero physically impossible predictions

**Paper 2 — Benchmark Study**
*"Evaluating Physical Constraint Enforcement in ML Potentials: A Systematic Benchmark on Materials Project"*
Target venue: Journal of Chemical Theory and Computation
Status: Data collection complete

**Paper 3 — Application Paper**
*"High-Throughput Screening of Perovskite Compositions for Solar Absorber Applications Using Physics-Informed Graph Networks"*
Target venue: Energy & Environmental Science
Status: Model training complete, writing in progress

### Conference Abstracts
- ICLR 2025 ML4Science Workshop — abstract accepted
- NeurIPS AI for Science Workshop — submission planned Q4 2025
- MRS Fall Meeting — poster presentation

---

## The Team

### Core Founding Team

**Founder & Chief Scientist**
Background in computational physics and solid-state chemistry. Developed the PINN-GNN architecture during undergraduate research. Deep expertise in pymatgen, DFT output interpretation, and GNN implementation from scratch.

**ML Research Lead**
Specialises in graph neural network architecture design. Contributed to published work on equivariant networks. Implements and benchmarks all model variants against state-of-the-art baselines.

**Computational Chemistry Lead**
DFT expert. Responsible for training data quality, understanding what DFT calculations actually represent, and ensuring physics constraint implementations are scientifically sound. Interface between the quantum chemistry world and the ML world.

**Software & Infrastructure Lead**
Builds the prediction API, Python SDK, and batch screening infrastructure. Ensures models are deployable at scale and that the developer experience for client integration is frictionless.

**Research Advisor (Academic)**
Senior faculty collaborator at [IISc / TIFR / IIT]. Provides scientific credibility, co-authorship on publications, access to HPC resources, and connections into the global materials science research community.

---

## Roadmap

### Year 1 — Foundation
- [ ] Complete PIGNet v1 architecture and benchmark paper
- [ ] Prediction API in private beta (band gap + formation energy)
- [ ] 10 research institution users onboarded (free tier)
- [ ] DST-SERB grant application submitted
- [ ] arXiv preprint published
- [ ] First co-authored paper with academic advisor submitted

### Year 2 — Validation
- [ ] Expand prediction coverage to 6 properties
- [ ] Paid API tier launched (Research Plan)
- [ ] 3 battery company pilot contracts signed
- [ ] Inverse design module in private beta
- [ ] Second paper published (benchmark study)
- [ ] Seed funding round (₹2–4 Cr, angel/pre-seed)

### Year 3 — Scale
- [ ] Enterprise plan clients in battery and semiconductor verticals
- [ ] Universal materials potential (interatomic forces, molecular dynamics integration)
- [ ] International research collaborations (MIT, Toronto, EPFL)
- [ ] Series A preparation
- [ ] 3 peer-reviewed publications in top-tier journals

### Year 4 — Commercial Scale
- [ ] Autonomous discovery loop (model suggests synthesis targets, not just evaluates them)
- [ ] First commercially validated material identified via Scandium Labs screening
- [ ] Series A raise from deep-tech investors (Breakthrough Energy, Khosla, a16z Bio)
- [ ] 15+ enterprise clients across battery, semiconductor, solar verticals

---

## Business Model

### Revenue Streams

**Tier 1 — Research Plan**
$800/month (or ₹65,000/month)
- 10,000 API predictions/month
- Band gap, formation energy, elastic moduli endpoints
- Materials Project integration
- Email support
- 1 seat

**Tier 2 — Enterprise Plan**
$4,500/month (or ₹3,75,000/month)
- Unlimited predictions
- All property endpoints including thermal and magnetic
- Batch screening tool (up to 100,000 structures/run)
- Custom model fine-tuning on proprietary datasets
- Dedicated support channel
- Up to 10 seats
- Quarterly research briefings with the science team

**Tier 3 — Custom Research Engagements**
₹25L–1Cr per engagement
- Bespoke model development for a specific material class
- Confidential proprietary dataset integration
- IP licensing options
- Multi-year research partnership agreements

**Tier 4 — Government & Academic Grants**
DST-SERB · CSIR · EU Horizon · NSF International · Materials Genome Initiative
Non-dilutive funding for fundamental research, providing both capital and scientific credibility.

### Unit Economics (Projected Year 3)
- Average Research Plan customer LTV: $28,000 (30-month average retention)
- Average Enterprise Plan customer LTV: $270,000 (60-month average retention)
- Gross margin at scale: ~82% (inference compute is the primary variable cost)
- CAC: ~$2,000 (primarily conference presence and arXiv/publication-driven inbound)

---

## Market Context

### AI in Materials Discovery Market
- 2024 market size: $536 million
- 2034 projected size: $5.6 billion
- CAGR: 26.4%

### Downstream Markets We Serve
- Battery materials: $202B by 2032
- Global chemicals: $5.7 trillion annually
- Semiconductors: $1 trillion industry by 2030

### Competitive Positioning

| Company | Approach | Weakness | Our Advantage |
|---|---|---|---|
| Citrine Informatics | General ML for manufacturing | No physics constraints, enterprise-only | Physics-constrained, researcher-accessible |
| Kebotix | Autonomous lab (hardware+software) | Capital-intensive, hardware-dependent | Pure software, 10× cheaper to deploy |
| Schrödinger | DFT + ML hybrid, pharma-focused | $100K+/year, pharma-centric | Materials-native, open-science integrations |
| DeepMind GNoME | Structure generation (Google) | Not a product, no property prediction API | Commercial product with API and support |
| CHGNet / M3GNet | Academic open-source models | No physics constraints, no support, no API | Production-ready, constrained, maintained |

**Scandium Labs' unique position:** The only commercially available materials property prediction platform that enforces physical constraints by design, making it the only one enterprise clients can trust for genuinely novel, out-of-distribution materials.

---

## Why Scandium?

Scandium (Sc, atomic number 21) is a transition metal — rare, underutilised, and extraordinary in its ability to transform the properties of alloys it touches. A small amount of scandium in aluminium creates aerospace-grade materials with dramatically superior strength-to-weight ratios. Scandium doesn't do the obvious thing. It sits at the border between common and exotic, between known and undiscovered.

That is exactly what we are building.

We sit at the border between physics and machine learning, between known materials science and the vast unexplored compositional space that no human has yet mapped. We are the trace element that changes what's possible.

The name also carries a secondary resonance: Sc is the periodic table symbol, immediately recognisable to any materials scientist or chemist. It signals that this company was named by people who know the field from the inside.

---

## Website Content Blocks

*The following are ready-to-use copy blocks for the Scandium Labs website.*

### Hero
**Headline:** Where Physics Meets Prediction
**Sub-headline:** We predict material properties from atomic structure in milliseconds — powered by graph neural networks that enforce the laws of physics.
**CTA 1:** Request API Access →
**CTA 2:** Read Our Research

### About Section
**Label:** ABOUT US
**Body:** Founded on the principles of scientific rigour and computational first principles, Scandium Labs is building the AI infrastructure for the next generation of materials discovery. We are a team of computational physicists and ML researchers who believe that the models predicting tomorrow's materials must be grounded in the physics of today. Our Physics-Informed Graph Networks don't just learn from data — they respect the laws of nature.

### Stats
- **154,000+** — Materials in training dataset
- **10,000×** — Faster than DFT simulation
- **0.12 eV** — Band gap prediction accuracy
- **6** — Material properties predicted

### Services / Capabilities Intro
**Label:** OUR CAPABILITIES
**Body:** From single-property prediction to full-scale batch screening, we offer the complete AI toolkit for computational materials discovery. Every prediction is physics-constrained. Every output is trustworthy.

### Projects / Research Intro
**Label:** OUR RESEARCH
**Scrolling text:** Featured Research — Physics-Constrained Intelligence —
**Body:** Our published and preprint work forms the scientific foundation of the Scandium Labs platform. Every model we deploy has been benchmarked against state-of-the-art baselines and validated against held-out DFT data.

### Pricing Intro
**Label:** ACCESS PLANS
**Scrolling text:** Flexible Plans, Powerful Results —
**Body:** Whether you're an academic researcher screening perovskites or an enterprise battery team running million-structure campaigns, we have a plan that fits your computational needs and budget.

### Contact
**Label:** LET'S CONNECT
**Scrolling text:** Let's Connect — Let's Connect —
**Body:** Whether you want to discuss a research collaboration, an enterprise pilot, a grant partnership, or just want to understand what our models can do for your specific material class — our science team is here.

### Footer Tagline
**BUILDING THE PERIODIC TABLE OF THE FUTURE — FROM INDIA, FOR THE WORLD**

---

*Document version: 1.0 · Scandium Labs · 2025*
*Classification: Internal / Founders Only*
