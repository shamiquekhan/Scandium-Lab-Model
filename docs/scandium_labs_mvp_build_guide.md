# Scandium Labs — Complete MVP Build Guide

> **Goal:** A working AI-powered materials property prediction platform that a researcher can log into, submit a crystal structure, and get back physics-constrained property predictions with explainability. Something you can demo to a grant committee or early research partner.

---

## Part 1 — Tech Stack Decisions

### Why These Choices

Every tool below is chosen for one of three reasons: it is the industry standard in its domain, it has the best open-source ecosystem for materials ML, or it minimizes yak-shaving so you spend time on science not DevOps.

---

### Backend — Python (FastAPI)

```
Python 3.11+
FastAPI          — async REST API, auto-generates OpenAPI docs
Uvicorn          — ASGI server
Celery + Redis   — async job queue (predictions take seconds to minutes)
SQLAlchemy 2.0   — ORM
PostgreSQL 15    — primary database
Alembic          — database migrations
Pydantic v2      — data validation and serialization
```

**Why FastAPI over Flask/Django:** Auto-generated docs (critical for research API), native async, Pydantic integration, and 3x faster than Flask. Django is overkill for an API-first product.

**Why Celery:** Property prediction jobs are not instant. A user submits a structure, you run inference (maybe 2–30 seconds), and return results. Never block an HTTP request for that. Celery lets you queue jobs and poll for results.

---

### ML Stack — PyTorch Geometric

```
PyTorch 2.2+
PyTorch Geometric (PyG)   — graph neural networks for materials
pymatgen                   — crystal structure parsing and analysis
matminer                   — feature engineering for materials
CHGNet / CGCNN             — pretrained model baselines
scikit-learn               — classical ML utilities
numpy / scipy              — numerical computing
ase                        — Atomic Simulation Environment
```

**Why PyG over DGL:** Better documentation, tighter PyTorch integration, more materials-specific model implementations available (SchNet, DimeNet, NequIP all have PyG versions).

**Why CHGNet as baseline:** Released 2023, currently state-of-the-art for universal interatomic potentials, has a clean Python API, and you can fine-tune it. Start here before training from scratch.

---

### Frontend — React + TypeScript

```
React 18
TypeScript 5
Vite                  — build tool (much faster than CRA/webpack)
TailwindCSS 3         — utility-first styling
React Query           — server state, polling for async job results
Zustand               — client state management
React Router v6       — routing
Recharts              — property prediction charts
3Dmol.js / Mol*       — 3D crystal structure visualization
shadcn/ui             — accessible component primitives
Axios                 — HTTP client
```

**Why 3Dmol.js:** Open-source, WebGL-based, specifically built for molecular and crystal structure visualization. Used by major structural biology databases. Alternative: Mol* (used by RCSB PDB) if you need more features.

---

### Infrastructure

```
Docker + Docker Compose     — local development and deployment
Nginx                       — reverse proxy
GitHub Actions              — CI/CD
AWS EC2 (g4dn.xlarge)       — GPU instance for inference ($0.526/hr, 1x T4 GPU)
AWS S3                       — file storage (CIF files, results)
AWS RDS PostgreSQL           — managed database
Redis (AWS ElastiCache)     — job queue and caching
```

**For MVP:** Run everything on a single EC2 instance with Docker Compose. Migrate to proper services (ECS, RDS) only when you have real users.

**GPU instance note:** g4dn.xlarge has 1x NVIDIA T4 (16GB VRAM) — sufficient for inference on all models in this guide. Spot pricing drops it to ~$0.16/hr. Use spot for batch jobs, on-demand for the API server.

---

## Part 2 — System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                         │
│              React SPA (Vite + TypeScript)                  │
│   Structure Upload → Job Submission → Results Dashboard     │
└───────────────────┬─────────────────────────────────────────┘
                    │ HTTPS
┌───────────────────▼─────────────────────────────────────────┐
│                      API LAYER (FastAPI)                    │
│                                                             │
│  /auth      /structures    /jobs     /predictions  /users   │
│                                                             │
│  Auth Middleware → Route Handlers → Pydantic Validation     │
└──────┬──────────────────────────────────────┬───────────────┘
       │                                      │
┌──────▼──────┐                    ┌──────────▼──────────────┐
│  PostgreSQL  │                    │      Redis Queue        │
│  (Primary   │                    │  Job submission →        │
│   Database) │                    │  Celery workers pick up  │
└─────────────┘                    └──────────┬──────────────┘
                                              │
┌─────────────────────────────────────────────▼──────────────┐
│                     ML WORKER LAYER (Celery)               │
│                                                             │
│  1. Parse CIF/POSCAR/JSON structure (pymatgen)             │
│  2. Build crystal graph (PyG)                              │
│  3. Run PINN-augmented GNN inference                       │
│  4. Apply physical constraint validation                   │
│  5. Generate explainability data (attention weights)       │
│  6. Store results → PostgreSQL + S3                        │
└────────────────────────────────────────────────────────────┘
```

---

### Database Schema

```sql
-- Users
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       VARCHAR(255) UNIQUE NOT NULL,
    name        VARCHAR(255),
    institution VARCHAR(255),
    api_key     VARCHAR(64) UNIQUE,
    plan        VARCHAR(20) DEFAULT 'free',  -- free | research | enterprise
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Crystal Structures (user uploads)
CREATE TABLE structures (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id),
    name            VARCHAR(255),
    formula         VARCHAR(100),         -- e.g. "Li2O", "GaN"
    space_group     VARCHAR(50),
    n_atoms         INTEGER,
    source_format   VARCHAR(20),          -- cif | poscar | json
    s3_key          VARCHAR(500),         -- raw file location
    pymatgen_json   JSONB,               -- parsed structure as JSON
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Prediction Jobs
CREATE TABLE jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID REFERENCES users(id),
    structure_id    UUID REFERENCES structures(id),
    status          VARCHAR(20) DEFAULT 'queued',  -- queued | running | complete | failed
    model_version   VARCHAR(50),
    properties      TEXT[],                        -- which properties to predict
    celery_task_id  VARCHAR(255),
    error_message   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ
);

-- Prediction Results
CREATE TABLE predictions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id              UUID REFERENCES jobs(id),
    property_name       VARCHAR(100),    -- band_gap | formation_energy | bulk_modulus | ...
    value               FLOAT,
    unit                VARCHAR(20),
    uncertainty         FLOAT,           -- model confidence / uncertainty estimate
    physical_valid      BOOLEAN,         -- passed PINN constraint checks
    constraint_details  JSONB,           -- which constraints passed/failed and by how much
    attention_weights   JSONB,           -- per-atom contribution for explainability
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Model Registry
CREATE TABLE models (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(100),
    version         VARCHAR(50),
    architecture    VARCHAR(100),        -- cgcnn | chgnet | pinn_gnn
    properties      TEXT[],             -- what it can predict
    rmse_band_gap   FLOAT,
    rmse_form_e     FLOAT,
    checkpoint_s3   VARCHAR(500),
    is_active       BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

---

### Directory Structure

```
scandium-labs/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI app factory
│   │   ├── config.py                # Settings (pydantic-settings)
│   │   ├── database.py              # SQLAlchemy engine + session
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── deps.py              # Shared dependencies (get_db, get_current_user)
│   │   │   └── routes/
│   │   │       ├── auth.py          # POST /auth/register, /auth/login, /auth/token
│   │   │       ├── structures.py    # POST /structures, GET /structures/{id}
│   │   │       ├── jobs.py          # POST /jobs, GET /jobs/{id}
│   │   │       ├── predictions.py   # GET /predictions/{job_id}
│   │   │       └── users.py         # GET /users/me, PATCH /users/me
│   │   │
│   │   ├── models/                  # SQLAlchemy ORM models
│   │   │   ├── user.py
│   │   │   ├── structure.py
│   │   │   ├── job.py
│   │   │   └── prediction.py
│   │   │
│   │   ├── schemas/                 # Pydantic request/response schemas
│   │   │   ├── user.py
│   │   │   ├── structure.py
│   │   │   ├── job.py
│   │   │   └── prediction.py
│   │   │
│   │   ├── services/
│   │   │   ├── storage.py           # S3 upload/download
│   │   │   ├── structure_parser.py  # CIF/POSCAR → pymatgen Structure
│   │   │   └── auth.py              # JWT creation/verification
│   │   │
│   │   └── workers/
│   │       ├── celery_app.py        # Celery instance
│   │       └── tasks/
│   │           ├── predict.py       # Main prediction pipeline task
│   │           └── validate.py      # Physical constraint checker
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── graph_builder.py         # pymatgen Structure → PyG Data
│   │   ├── models/
│   │   │   ├── base.py              # Abstract base model
│   │   │   ├── cgcnn.py             # CGCNN implementation
│   │   │   └── pinn_gnn.py          # Your PINN-augmented GNN (main model)
│   │   ├── constraints/
│   │   │   ├── symmetry.py          # Symmetry invariance checks
│   │   │   ├── thermodynamic.py     # Formation energy bounds
│   │   │   └── quantum.py           # Band gap non-negativity, quantum limits
│   │   ├── inference.py             # Model loading + inference pipeline
│   │   ├── uncertainty.py           # MC dropout / ensemble uncertainty
│   │   └── checkpoints/             # .pth model weights (gitignored, pulled from S3)
│   │
│   ├── migrations/                  # Alembic migration files
│   ├── tests/
│   │   ├── test_api/
│   │   ├── test_ml/
│   │   └── test_workers/
│   ├── Dockerfile
│   ├── Dockerfile.worker
│   ├── requirements.txt
│   └── pyproject.toml
│
├── frontend/
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── router.tsx
│   │   │
│   │   ├── api/                     # API client layer
│   │   │   ├── client.ts            # Axios instance with auth interceptors
│   │   │   ├── structures.ts
│   │   │   ├── jobs.ts
│   │   │   └── predictions.ts
│   │   │
│   │   ├── components/
│   │   │   ├── ui/                  # shadcn/ui primitives
│   │   │   ├── StructureUploader/   # Drag-drop CIF upload + validation
│   │   │   ├── StructureViewer/     # 3Dmol.js wrapper component
│   │   │   ├── PropertyCard/        # Single predicted property display
│   │   │   ├── PredictionDashboard/ # Full results view
│   │   │   ├── JobStatusPoller/     # Polls /jobs/{id} until complete
│   │   │   └── ConstraintBadge/     # Physical validity indicator
│   │   │
│   │   ├── pages/
│   │   │   ├── Landing.tsx
│   │   │   ├── Dashboard.tsx        # User's job history
   │   │   ├── NewPrediction.tsx    # Upload + configure prediction job
│   │   │   ├── Results.tsx          # Full prediction results
│   │   │   └── Auth/
│   │   │       ├── Login.tsx
│   │   │       └── Register.tsx
│   │   │
│   │   ├── store/
│   │   │   └── auth.ts              # Zustand auth store
│   │   │
│   │   └── types/
│   │       └── index.ts             # TypeScript interfaces (Structure, Job, Prediction)
│   │
│   ├── public/
│   ├── index.html
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   └── tsconfig.json
│
├── docker-compose.yml               # Local dev: all services
├── docker-compose.prod.yml          # Production overrides
├── nginx/
│   └── nginx.conf
└── .github/
    └── workflows/
        ├── test.yml                 # Run tests on PR
        └── deploy.yml               # Deploy on main merge
```

---

## Part 3 — Step-by-Step Build Order

Build in exactly this order. Each phase produces something runnable.

---

### Phase 1 — Local Dev Environment (Day 1–2)

**Step 1: Repository setup**

```bash
mkdir scandium-labs && cd scandium-labs
git init
python -m venv .venv && source .venv/bin/activate

pip install fastapi uvicorn[standard] sqlalchemy alembic pydantic-settings \
            psycopg2-binary redis celery python-jose[cryptography] \
            passlib[bcrypt] python-multipart boto3 httpx

pip install pymatgen torch torch-geometric matminer ase chgnet
```

**Step 2: docker-compose.yml for local services**

```yaml
version: '3.9'
services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: scandium
      POSTGRES_USER: scandium
      POSTGRES_PASSWORD: scandium_dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  api:
    build: ./backend
    command: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    volumes:
      - ./backend:/app
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://scandium:scandium_dev@db:5432/scandium
      REDIS_URL: redis://redis:6379/0
      SECRET_KEY: dev-secret-key-change-in-prod
      AWS_BUCKET: scandium-dev
    depends_on:
      - db
      - redis

  worker:
    build:
      context: ./backend
      dockerfile: Dockerfile.worker
    command: celery -A app.workers.celery_app worker --loglevel=info -Q predictions
    volumes:
      - ./backend:/app
    environment:
      DATABASE_URL: postgresql://scandium:scandium_dev@db:5432/scandium
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - db
      - redis

volumes:
  postgres_data:
```

```bash
docker-compose up -d db redis
# Run API locally for hot reload during development
```

---

### Phase 2 — Core API (Week 1)

**Step 3: FastAPI application factory**

```python
# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import auth, structures, jobs, predictions, users
from app.database import engine, Base

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Scandium Labs API",
    description="Physics-constrained AI for materials property prediction",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(structures.router, prefix="/structures", tags=["structures"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
app.include_router(users.router, prefix="/users", tags=["users"])

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}
```

**Step 4: Structure upload endpoint**

```python
# backend/app/api/routes/structures.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api.deps import get_db, get_current_user
from app.services.structure_parser import parse_structure_file
from app.services.storage import upload_to_s3
from app.models.structure import Structure
import uuid

router = APIRouter()

ALLOWED_FORMATS = {".cif", ".poscar", ".json", ".vasp"}

@router.post("/")
async def upload_structure(
    file: UploadFile = File(...),
    name: str = None,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    # Validate file extension
    suffix = "." + file.filename.split(".")[-1].lower()
    if suffix not in ALLOWED_FORMATS:
        raise HTTPException(400, f"Unsupported format. Allowed: {ALLOWED_FORMATS}")

    contents = await file.read()

    # Parse with pymatgen — validates the structure is real
    try:
        parsed = parse_structure_file(contents, suffix)
    except Exception as e:
        raise HTTPException(422, f"Invalid structure file: {str(e)}")

    # Upload raw file to S3
    s3_key = f"structures/{current_user.id}/{uuid.uuid4()}{suffix}"
    upload_to_s3(contents, s3_key)

    # Save to DB
    structure = Structure(
        user_id=current_user.id,
        name=name or file.filename,
        formula=parsed["formula"],
        space_group=parsed["space_group"],
        n_atoms=parsed["n_atoms"],
        source_format=suffix.lstrip("."),
        s3_key=s3_key,
        pymatgen_json=parsed["structure_json"]
    )
    db.add(structure)
    db.commit()
    db.refresh(structure)
    return structure
```

**Step 5: Structure parser service**

```python
# backend/app/services/structure_parser.py
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp import Poscar
import io, json

def parse_structure_file(contents: bytes, suffix: str) -> dict:
    """
    Parse CIF, POSCAR, or pymatgen JSON into a normalized dict.
    Returns formula, space group, atom count, and the full pymatgen JSON.
    """
    if suffix == ".cif":
        parser = CifParser(io.StringIO(contents.decode("utf-8")))
        structure = parser.get_structures()[0]
    elif suffix in (".poscar", ".vasp"):
        poscar = Poscar.from_str(contents.decode("utf-8"))
        structure = poscar.structure
    elif suffix == ".json":
        structure = Structure.from_dict(json.loads(contents))
    else:
        raise ValueError(f"Unknown format: {suffix}")

    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    sga = SpacegroupAnalyzer(structure)

    return {
        "formula": structure.composition.reduced_formula,
        "space_group": sga.get_space_group_symbol(),
        "n_atoms": len(structure),
        "structure_json": structure.as_dict()
    }
```

---

### Phase 3 — ML Pipeline (Week 2–3)

**Step 6: Crystal graph builder**

```python
# backend/ml/graph_builder.py
import torch
from torch_geometric.data import Data
from pymatgen.core import Structure
import numpy as np

# Atomic features: atomic number, electronegativity, atomic radius, group, period
ELEMENT_FEATURES = { ... }  # Pre-built lookup table, ~100 elements

def structure_to_graph(structure: Structure, cutoff: float = 8.0) -> Data:
    """
    Convert a pymatgen Structure to a PyTorch Geometric Data object.
    
    Nodes = atoms (with elemental features)
    Edges = bonds within cutoff radius (with distance + direction features)
    """
    # Node features
    node_features = []
    for site in structure:
        el = site.specie.symbol
        node_features.append(ELEMENT_FEATURES.get(el, [0]*9))
    
    x = torch.tensor(node_features, dtype=torch.float)

    # Edges: all pairs within cutoff, with periodic boundary conditions
    center_indices, neighbor_indices, _, distances = \
        structure.get_neighbor_list(r=cutoff, numerical_tol=1e-8)
    
    edge_index = torch.tensor(
        [center_indices, neighbor_indices], dtype=torch.long
    )
    
    # Edge features: distance (Gaussian-expanded) + unit vector
    edge_attr = gaussian_expand_distances(distances)

    # Global state: unit cell volume, density
    global_feat = torch.tensor([
        [structure.volume, structure.density, len(structure)]
    ], dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        global_feat=global_feat,
        formula=structure.composition.reduced_formula
    )

def gaussian_expand_distances(distances: np.ndarray, n_filters=40, cutoff=8.0) -> torch.Tensor:
    """
    Expand scalar distances into a fixed-length vector using a Gaussian filter bank.
    Standard technique from CGCNN/SchNet.
    """
    centers = np.linspace(0, cutoff, n_filters)
    width = cutoff / n_filters
    expanded = np.exp(-((distances[:, None] - centers[None, :]) ** 2) / (2 * width**2))
    return torch.tensor(expanded, dtype=torch.float)
```

**Step 7: PINN-augmented GNN model**

```python
# backend/ml/models/pinn_gnn.py
import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data

class PhysicsConstrainedGNN(nn.Module):
    """
    CGCNN-style graph network augmented with physics-informed loss constraints.
    
    Architecture:
      - Embedding layer: node features → hidden dim
      - N x Crystal Graph Conv layers
      - Global pooling → crystal-level representation
      - Output heads: one per property (multi-task)
    
    Training adds penalty terms for:
      - Band gap negativity (quantum constraint)
      - Formation energy physical bounds
      - Symmetry-equivalent structure equivalence
    """

    def __init__(
        self,
        node_feat_dim: int = 9,
        edge_feat_dim: int = 40,
        hidden_dim: int = 128,
        n_conv_layers: int = 4,
        n_properties: int = 4,     # band_gap, formation_energy, bulk_modulus, shear_modulus
        dropout: float = 0.1
    ):
        super().__init__()

        # Initial embedding
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)

        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            CGConv(hidden_dim, dim=edge_feat_dim, batch_norm=True)
            for _ in range(n_conv_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

        # Pooling → crystal representation
        # Concatenate mean and sum pooling for richer global descriptor
        self.crystal_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # Separate output head for each property (multi-task)
        self.property_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 2, 32),
                nn.SiLU(),
                nn.Linear(32, 1)
            )
            for _ in range(n_properties)
        ])

        # Uncertainty head (outputs log variance for each property)
        self.uncertainty_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1)
            for _ in range(n_properties)
        ])

    def forward(self, data: Data):
        x, edge_index, edge_attr, batch = \
            data.x, data.edge_index, data.edge_attr, data.batch

        # Embed node features
        x = self.node_embedding(x)
        x = self.activation(x)

        # Message passing
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)
            x = self.activation(x)
            x = self.dropout(x)

        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_sum = global_add_pool(x, batch)
        x_global = torch.cat([x_mean, x_sum], dim=-1)

        # Crystal-level representation
        crystal_repr = self.crystal_proj(x_global)

        # Predictions + uncertainties for each property
        predictions = [head(crystal_repr).squeeze(-1) for head in self.property_heads]
        log_vars = [head(crystal_repr).squeeze(-1) for head in self.uncertainty_heads]

        return torch.stack(predictions, dim=-1), torch.stack(log_vars, dim=-1)


class PINNLoss(nn.Module):
    """
    Physics-Informed loss combining:
    1. Prediction loss (MSE with uncertainty weighting)
    2. Physical constraint penalties
    """

    PROPERTY_NAMES = ["band_gap", "formation_energy", "bulk_modulus", "shear_modulus"]

    def __init__(self, lambda_physics: float = 0.1):
        super().__init__()
        self.lambda_physics = lambda_physics

    def forward(self, predictions, log_vars, targets, mask):
        """
        predictions: (batch, n_properties)
        log_vars:    (batch, n_properties) — log variance for uncertainty
        targets:     (batch, n_properties) — DFT ground truth
        mask:        (batch, n_properties) — 1 where target exists, 0 where missing
        """
        # 1. Gaussian NLL loss (handles uncertainty and missing values)
        precision = torch.exp(-log_vars)
        prediction_loss = (
            mask * (precision * (predictions - targets) ** 2 + log_vars)
        ).sum() / mask.sum().clamp(min=1)

        # 2. Physics constraints
        band_gap = predictions[:, 0]
        formation_energy = predictions[:, 1]

        # Constraint 2a: Band gap must be non-negative
        band_gap_penalty = torch.relu(-band_gap).mean()

        # Constraint 2b: Formation energy for known stable materials should be < 0
        # (only penalise very positive values — above 3 eV/atom is clearly unphysical)
        form_e_penalty = torch.relu(formation_energy - 3.0).mean()

        # Constraint 2c: Bulk modulus must be positive (materials resist compression)
        bulk_mod = predictions[:, 2]
        bulk_penalty = torch.relu(-bulk_mod).mean()

        physics_loss = band_gap_penalty + form_e_penalty + bulk_penalty

        total_loss = prediction_loss + self.lambda_physics * physics_loss

        return total_loss, {
            "prediction_loss": prediction_loss.item(),
            "physics_loss": physics_loss.item(),
            "band_gap_penalty": band_gap_penalty.item(),
        }
```

**Step 8: Inference pipeline**

```python
# backend/ml/inference.py
import torch
from pymatgen.core import Structure
from ml.graph_builder import structure_to_graph
from ml.models.pinn_gnn import PhysicsConstrainedGNN
from ml.constraints.quantum import validate_quantum_constraints
import numpy as np

PROPERTY_NAMES = ["band_gap", "formation_energy", "bulk_modulus", "shear_modulus"]
PROPERTY_UNITS = {"band_gap": "eV", "formation_energy": "eV/atom",
                  "bulk_modulus": "GPa", "shear_modulus": "GPa"}

class MaterialsPredictor:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = PhysicsConstrainedGNN()
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        self.model.to(self.device)

        # Normalization stats (computed from training set)
        self.means = torch.tensor(state["property_means"]).to(self.device)
        self.stds = torch.tensor(state["property_stds"]).to(self.device)

    @torch.no_grad()
    def predict(self, structure: Structure, n_mc_samples: int = 20) -> list[dict]:
        """
        Run inference with MC dropout for uncertainty estimation.
        Returns list of property prediction dicts.
        """
        graph = structure_to_graph(structure).to(self.device)

        # Monte Carlo dropout: run N times with dropout ON for uncertainty
        self.model.train()  # enables dropout
        samples = []
        for _ in range(n_mc_samples):
            preds, _ = self.model(graph)
            samples.append(preds.cpu())
        self.model.eval()

        samples = torch.stack(samples, dim=0)  # (n_samples, n_properties)
        mean_pred = samples.mean(0)
        std_pred = samples.std(0)

        # Denormalize
        mean_denorm = (mean_pred * self.stds + self.means).numpy()
        std_denorm = (std_pred * self.stds).numpy()

        results = []
        for i, prop in enumerate(PROPERTY_NAMES):
            value = float(mean_denorm[i])
            uncertainty = float(std_denorm[i])

            # Physical constraint validation
            constraint_result = validate_quantum_constraints(prop, value)

            results.append({
                "property_name": prop,
                "value": value,
                "unit": PROPERTY_UNITS[prop],
                "uncertainty": uncertainty,
                "physical_valid": constraint_result["valid"],
                "constraint_details": constraint_result
            })

        return results
```

---

### Phase 4 — Celery Worker (Week 3)

**Step 9: Prediction task**

```python
# backend/app/workers/tasks/predict.py
from app.workers.celery_app import celery_app
from app.database import SessionLocal
from app.models.job import Job, JobStatus
from app.models.prediction import Prediction
from app.services.storage import download_from_s3
from ml.inference import MaterialsPredictor
from pymatgen.core import Structure
import json, logging

logger = logging.getLogger(__name__)

# Loaded once at worker startup — not on every task
predictor = MaterialsPredictor(checkpoint_path="ml/checkpoints/pinn_gnn_v1.pth")

@celery_app.task(
    bind=True,
    max_retries=2,
    soft_time_limit=120,  # 2 minute timeout
    queue="predictions"
)
def run_prediction(self, job_id: str):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found")
            return

        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        db.commit()

        # Load structure from DB (already parsed at upload time)
        structure_dict = job.structure.pymatgen_json
        structure = Structure.from_dict(structure_dict)

        # Run inference
        results = predictor.predict(structure)

        # Save predictions
        for r in results:
            pred = Prediction(
                job_id=job_id,
                property_name=r["property_name"],
                value=r["value"],
                unit=r["unit"],
                uncertainty=r["uncertainty"],
                physical_valid=r["physical_valid"],
                constraint_details=r["constraint_details"]
            )
            db.add(pred)

        job.status = JobStatus.COMPLETE
        job.completed_at = datetime.utcnow()
        db.commit()

    except Exception as exc:
        logger.exception(f"Prediction failed for job {job_id}")
        job.status = JobStatus.FAILED
        job.error_message = str(exc)
        db.commit()
        raise self.retry(exc=exc)
    finally:
        db.close()
```

---

### Phase 5 — Frontend (Week 4–5)

**Step 10: Vite + React setup**

```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install @tanstack/react-query axios zustand react-router-dom
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npm install recharts 3dmol
npm install @radix-ui/react-dialog @radix-ui/react-progress lucide-react
```

**Step 11: API client with auth interceptor**

```typescript
// frontend/src/api/client.ts
import axios from "axios";
import { useAuthStore } from "@/store/auth";

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
  timeout: 30000,
});

// Attach JWT to every request
client.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Auto-logout on 401
client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout();
      window.location.href = "/login";
    }
    return Promise.reject(error);
  }
);

export default client;
```

**Step 12: Job status poller (React Query)**

```typescript
// frontend/src/components/JobStatusPoller/index.tsx
import { useQuery } from "@tanstack/react-query";
import { getJob, getPredictions } from "@/api/jobs";
import { PredictionDashboard } from "@/components/PredictionDashboard";
import { Progress } from "@/components/ui/progress";

interface Props { jobId: string; }

export function JobStatusPoller({ jobId }: Props) {
  const { data: job } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId),
    // Poll every 2 seconds until complete or failed
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "complete" || status === "failed" ? false : 2000;
    },
  });

  const { data: predictions } = useQuery({
    queryKey: ["predictions", jobId],
    queryFn: () => getPredictions(jobId),
    enabled: job?.status === "complete",
  });

  if (!job) return <div>Loading...</div>;

  if (job.status === "queued" || job.status === "running") {
    return (
      <div className="space-y-4">
        <p className="text-sm text-gray-500">
          {job.status === "queued" ? "Queued..." : "Running inference..."}
        </p>
        <Progress value={job.status === "running" ? 60 : 20} className="w-full" />
      </div>
    );
  }

  if (job.status === "failed") {
    return <div className="text-red-500">Prediction failed: {job.error_message}</div>;
  }

  return predictions ? <PredictionDashboard predictions={predictions} /> : null;
}
```

**Step 13: 3D structure viewer wrapper**

```typescript
// frontend/src/components/StructureViewer/index.tsx
import { useEffect, useRef } from "react";
// @ts-ignore
import * as $3Dmol from "3dmol";

interface Props {
  cifString: string;
  style?: React.CSSProperties;
}

export function StructureViewer({ cifString, style }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || !cifString) return;

    const viewer = $3Dmol.createViewer(containerRef.current, {
      backgroundColor: "0x0f0f0f",
    });

    viewer.addModel(cifString, "cif");
    viewer.setStyle({}, {
      sphere: { scale: 0.3, colorscheme: "Jmol" },
      stick: { radius: 0.15, colorscheme: "Jmol" },
    });
    viewer.addUnitCell();
    viewer.zoomTo();
    viewer.render();

    // Enable rotation on mouse drag
    viewer.setHoverable({}, true,
      (atom: any) => viewer.addLabel(atom.elem, { position: atom, backgroundColor: "white" }),
      () => viewer.removeAllLabels()
    );

    return () => viewer.clear();
  }, [cifString]);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "400px", position: "relative", ...style }}
    />
  );
}
```

---

### Phase 6 — Training Your Model (Parallel with Frontend)

**Step 14: Download training data**

```python
# scripts/download_training_data.py
from mp_api.client import MPRester
import json

API_KEY = "your_materials_project_api_key"

with MPRester(API_KEY) as mpr:
    # Pull ~50k materials covering semiconductors, metals, insulators
    docs = mpr.summary.search(
        fields=[
            "material_id", "formula_pretty", "structure",
            "band_gap", "formation_energy_per_atom",
            "bulk_modulus", "shear_modulus",
            "is_stable"
        ],
        num_chunks=100
    )

# Save raw data
with open("data/materials_project_50k.json", "w") as f:
    json.dump([d.dict() for d in docs], f)

print(f"Downloaded {len(docs)} materials")
```

**Step 15: Training script**

```python
# scripts/train.py
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from ml.models.pinn_gnn import PhysicsConstrainedGNN, PINNLoss
from ml.graph_builder import structure_to_graph
from pymatgen.core import Structure
import json, numpy as np
from tqdm import tqdm

# Load and split dataset (80/10/10)
with open("data/materials_project_50k.json") as f:
    raw = json.load(f)

# Filter: require at least band_gap and formation_energy
data = [r for r in raw if r.get("band_gap") is not None
        and r.get("formation_energy_per_atom") is not None]

np.random.shuffle(data)
n = len(data)
train = data[:int(0.8*n)]
val = data[int(0.8*n):int(0.9*n)]

# Build PyG graphs
def make_graph(record):
    structure = Structure.from_dict(record["structure"])
    graph = structure_to_graph(structure)
    graph.y = torch.tensor([
        record.get("band_gap", float("nan")),
        record.get("formation_energy_per_atom", float("nan")),
        record.get("bulk_modulus", {}).get("voigt", float("nan")),
        record.get("shear_modulus", {}).get("voigt", float("nan")),
    ])
    return graph

train_graphs = [make_graph(r) for r in tqdm(train, desc="Building train graphs")]
val_graphs = [make_graph(r) for r in tqdm(val, desc="Building val graphs")]

train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=64)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhysicsConstrainedGNN().to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
criterion = PINNLoss(lambda_physics=0.1)

best_val_loss = float("inf")

for epoch in range(100):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        mask = ~torch.isnan(batch.y)
        targets = torch.nan_to_num(batch.y)

        optimizer.zero_grad()
        preds, log_vars = model(batch)
        loss, loss_details = criterion(preds, log_vars, targets, mask.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            mask = ~torch.isnan(batch.y)
            targets = torch.nan_to_num(batch.y)
            preds, log_vars = model(batch)
            loss, _ = criterion(preds, log_vars, targets, mask.float())
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1} | Train: {train_loss/len(train_loader):.4f} | Val: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
        }, "ml/checkpoints/pinn_gnn_v1.pth")
        print(f"  → Saved best model (val_loss={val_loss:.4f})")
```

---

## Part 4 — Physical Constraint Validation Layer

This is your scientific differentiator. Every prediction passes through this before being shown to users.

```python
# backend/ml/constraints/quantum.py

PROPERTY_CONSTRAINTS = {
    "band_gap": {
        "min": 0.0,        # Cannot be negative (quantum mechanics)
        "max": 20.0,       # No known material exceeds ~14 eV, 20 is conservative
        "unit": "eV"
    },
    "formation_energy": {
        "min": -5.0,       # Most stable materials are above -4 eV/atom
        "max": 5.0,        # Very unstable, but possible for some compounds
        "unit": "eV/atom"
    },
    "bulk_modulus": {
        "min": 0.0,        # Must be positive (stable material)
        "max": 1000.0,     # Diamond is ~440 GPa — 1000 is very conservative
        "unit": "GPa"
    },
    "shear_modulus": {
        "min": 0.0,
        "max": 800.0,
        "unit": "GPa"
    }
}

def validate_quantum_constraints(property_name: str, value: float) -> dict:
    """
    Check if a predicted value obeys known physical limits.
    Returns validation result with explanation.
    """
    if property_name not in PROPERTY_CONSTRAINTS:
        return {"valid": True, "message": "No constraints defined for this property"}

    c = PROPERTY_CONSTRAINTS[property_name]
    violations = []

    if value < c["min"]:
        violations.append(
            f"Value {value:.3f} {c['unit']} is below physical minimum {c['min']} {c['unit']}"
        )
    if value > c["max"]:
        violations.append(
            f"Value {value:.3f} {c['unit']} exceeds physical maximum {c['max']} {c['unit']}"
        )

    # Confidence: how far from the bounds (normalized)
    range_size = c["max"] - c["min"]
    margin_from_min = (value - c["min"]) / range_size
    margin_from_max = (c["max"] - value) / range_size
    confidence = min(margin_from_min, margin_from_max)  # 0 = at boundary, 0.5 = center

    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "confidence": float(np.clip(confidence, 0, 1)),
        "bounds": {"min": c["min"], "max": c["max"], "unit": c["unit"]}
    }
```

---

## Part 5 — Deployment

### Docker Configuration

**backend/Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**backend/Dockerfile.worker:**
```dockerfile
FROM python:3.11-slim

# Needs CUDA for GPU inference
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["celery", "-A", "app.workers.celery_app", "worker", "--loglevel=info", "-Q", "predictions", "-c", "2"]
```

### GitHub Actions CI/CD

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r backend/requirements.txt
      - run: pytest backend/tests/ -v

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to EC2
        env:
          EC2_HOST: ${{ secrets.EC2_HOST }}
          EC2_KEY: ${{ secrets.EC2_PRIVATE_KEY }}
        run: |
          echo "$EC2_KEY" > key.pem && chmod 600 key.pem
          ssh -i key.pem ubuntu@$EC2_HOST "
            cd /opt/scandium-labs &&
            git pull origin main &&
            docker-compose -f docker-compose.prod.yml up -d --build
          "
```

---

## Part 6 — Properties to Predict (MVP Scope)

Focus on these four properties for MVP. Each has abundant training data in Materials Project and clear commercial value.

| Property | Unit | Materials Project Coverage | Commercial Use |
|---|---|---|---|
| Band gap | eV | 154k+ materials | Solar cells, LEDs, semiconductors |
| Formation energy | eV/atom | 154k+ materials | Stability / synthesizability |
| Bulk modulus | GPa | ~30k materials | Structural materials, aerospace |
| Shear modulus | GPa | ~30k materials | Structural materials, aerospace |

**Post-MVP additions (in this order):**
- Thermal conductivity (battery thermal management)
- Magnetic moment (data storage, quantum computing)
- Ionic conductivity (solid-state electrolytes — your biggest commercial target)
- Optical absorption spectrum (solar cell efficiency prediction)

---

## Part 7 — Build Timeline

| Week | What You Build | Milestone |
|---|---|---|
| 1 | FastAPI skeleton, DB schema, auth, structure upload | Can upload a CIF file via API |
| 2 | ML graph builder, CGCNN baseline, dataset download | Baseline model predicting band gaps |
| 3 | PINN-augmented GNN, physical constraints, Celery worker | End-to-end prediction pipeline working |
| 4 | React frontend: upload + job submission + polling | Browser → API → ML → results |
| 5 | 3D viewer, results dashboard, constraint badges | Full user flow works |
| 6 | Model training on 50k Materials Project structures | Trained model with published RMSE |
| 7 | EC2 deployment, nginx, CI/CD | Live URL you can share |
| 8 | Buffer: polish, edge cases, test coverage | Demo-ready product |

---

## Part 8 — Immediate Next Steps (This Week)

Do these in order. Do not skip ahead.

1. `pip install pymatgen` — if you hit errors, fix them before anything else
2. Get a Materials Project API key at materialsproject.org — it's free and instant
3. Download 1,000 materials with their band gaps (the code is in the ML stack section above)
4. Run the structure parser on one CIF file from the dataset
5. Build one graph with `structure_to_graph()` and print its shape
6. Train a minimal CGCNN for 5 epochs — confirm loss goes down
7. Only then start the FastAPI skeleton

The ML pipeline has to work before the API wraps it. Build inside-out: core ML first, API second, frontend last.

---

## Appendix — Key Environment Variables

```env
# Backend
DATABASE_URL=postgresql://user:pass@localhost:5432/scandium
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-256-bit-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=ap-south-1
AWS_BUCKET=scandium-materials

MP_API_KEY=your-materials-project-api-key
MODEL_CHECKPOINT=ml/checkpoints/pinn_gnn_v1.pth

# Frontend
VITE_API_URL=http://localhost:8000
```

---

## Appendix — Benchmark Targets (Know If Your Model Is Good)

These are published RMSE values from the literature. Beat these on Materials Project holdout set and you have a publishable result.

| Model | Band Gap RMSE (eV) | Formation Energy RMSE (eV/atom) |
|---|---|---|
| CGCNN (baseline) | 0.388 | 0.039 |
| MEGNet | 0.330 | 0.028 |
| SchNet | 0.345 | 0.035 |
| **Your target** | **< 0.300** | **< 0.025** |
| CHGNet (SOTA) | 0.270 | 0.018 |

If your PINN constraints genuinely improve out-of-distribution performance, you do not need to beat CHGNet on in-distribution accuracy. You need to show that for materials compositionally distant from the training set, your physically-constrained model degrades more gracefully than the unconstrained baseline. That is the paper.
