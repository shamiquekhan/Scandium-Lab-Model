# Scandium Labs MVP — Complete Build

## Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker + Docker Compose
- PostgreSQL 15 (or Docker)

### Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Initialize database
alembic upgrade head

# Run API server
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Worker Setup (requires GPU or CUDA simulation)

```bash
celery -A app.workers.celery_app worker --loglevel=info -Q predictions
```

### All Services (Docker Compose)

```bash
docker-compose up
# Frontend: http://localhost:5173
# API: http://localhost:8000
# Nginx: http://localhost
```

---

## Architecture Overview

**Backend Stack:**
- FastAPI (async REST API)
- SQLAlchemy 2.0 (ORM)
- PostgreSQL (primary DB)
- Redis (job queue + caching)
- Celery (async task worker)

**ML Stack:**
- PyTorch 2.2 + PyG (graph neural networks)
- Pymatgen (crystal structure parsing)
- e3nn (SE(3)-equivariant convolutions)

**Frontend:**
- React 18 + TypeScript
- Vite (build tool)
- TailwindCSS (styling)
- React Query (server state)
- Zustand (client state)

**DevOps:**
- Docker + Docker Compose
- GitHub Actions (CI/CD)
- Nginx (reverse proxy)

---

## Key Files

### Backend
- `backend/app/main.py` — FastAPI application entry point
- `backend/app/workers/celery_app.py` — Celery configuration
- `backend/app/workers/tasks/predict.py` — Prediction task (loads model, runs inference)
- `backend/ml/models/cgcnn_baseline.py` — CGCNN baseline model
- `backend/ml/models/equivariant_model.py` — SE(3)-equivariant GNN with e3nn
- `backend/ml/models/pinn_loss.py` — Physics-informed loss with lambda annealing
- `backend/ml/graph_builder.py` — Crystal graph construction (handles periodic images)
- `backend/ml/training/train_baseline.py` — Training harness with compositional OOD split
- `backend/ml/training/train_equivariant.py` — Equivariant model training with PINN loss
- `backend/ml/data/ood_splits.py` — OOD split strategies (compositional, element, property, density)
- `backend/ml/evaluation/eval_ood.py` — OOD benchmark runner
- `backend/ml/evaluation/report_ood.py` — Report generator

### Frontend
- `frontend/src/App.tsx` — Root component
- `frontend/src/router.tsx` — Route definitions
- `frontend/src/api/client.ts` — Axios client with auth interceptor
- `frontend/src/pages/Landing.tsx` — Home page
- `frontend/src/pages/NewPrediction.tsx` — Structure upload + job submission
- `frontend/src/pages/Results.tsx` — Results viewer with job status polling
- `frontend/src/components/JobStatusPoller.tsx` — Polls job status until complete
- `frontend/src/store/auth.ts` — Zustand auth store

### Tests
- `backend/tests/test_api.py` — API endpoint tests
- `backend/tests/test_ml_model.py` — Model forward pass tests
- `backend/tests/test_graph_builder.py` — Graph construction tests
- `backend/tests/test_pinn_loss.py` — Loss function tests
- `backend/tests/test_ood_splits.py` — OOD split validation
- `backend/tests/test_training.py` — Training harness smoke tests

---

## Technical Highlights

### 1. Periodic Image Handling
The graph builder correctly uses `Structure.get_neighbor_list()` which returns image offsets. Edge vectors are computed including lattice shifts:
```python
shift = np.dot(np.array(img), lattice.matrix)
vec = (neighbor_coords + shift) - center_coords
```

### 2. PINN Loss with Lambda Annealing
- Starts with `lambda=0` (data-only loss)
- Linearly increases to `lambda=0.1` over warmup steps
- Physics penalties: band gap non-negativity, formation energy bounds, bulk modulus positive

### 3. SE(3)-Equivariance
- Uses e3nn spherical harmonics for rotationally-invariant edge features
- Model guaranteed equivariant by construction (not just by loss penalty)
- Includes optional rotation-equivariance penalty during training

### 4. OOD Evaluation
Four split strategies:
- **Compositional**: hold out entire chemical systems
- **Element holdout**: hold out lanthanides, transition metals, etc.
- **Property extrema**: hold out high band gap / unusual properties
- **Density anomaly**: hold out low/high density materials

Generates RMSE degradation curves + ECE uncertainty calibration metrics.

### 5. Uncertainty Calibration (ECE)
- Model predicts log variance per property
- ECE measures if predicted uncertainty matches actual error
- Reported per OOD split to detect overconfidence on novel data

---

## Training & Evaluation Commands

```bash
# Train CGCNN baseline (compositional split)
python backend/ml/training/train_baseline.py

# Train equivariant model (PINN loss + rotations)
python backend/ml/training/train_equivariant.py

# Run OOD benchmark
python backend/ml/evaluation/eval_ood.py ml/checkpoints/equivariant_best.pth data/materials_project_50k.json

# Generate report
python backend/ml/evaluation/report_ood.py evaluations/ood_benchmark.json
```

---

## Tests

```bash
# Run all backend tests
pytest backend/tests/ -v

# With coverage
pytest backend/tests/ --cov=backend/app --cov-report=html

# Frontend type check
cd frontend && npm run type-check
```

---

## CI/CD

- **test.yml**: Runs unit tests, linting, mypy, TypeScript checks on PR/push
- **deploy.yml**: Builds Docker images on merge to main

---

## Next Steps (Post-MVP)

1. **Auth endpoints**: Register, login, token refresh
2. **Structure upload**: POST /structures with multipart file handling
3. **Job routes**: POST /jobs, GET /jobs/{id}, real-time status updates
4. **Prediction retrieval**: GET /predictions?job_id={id}
5. **3D viewer integration**: Full 3Dmol.js component
6. **Model selection UI**: Choose between baseline/equivariant models
7. **Batch API**: Submit multiple structures in one request
8. **Ray Tune hyperparameter optimization**: Lambda annealing schedule, cutoff radius, n_conv_layers
9. **Tensorboard logging**: Training curves, loss components
10. **Production deployment**: AWS ECR, ECS, RDS, ElastiCache

---

## Deployment Checklist

- [ ] Data downloaded from Materials Project (MP_API_KEY set)
- [ ] Models trained: `equivariant_best.pth` in `ml/checkpoints/`
- [ ] OOD benchmark run and report generated
- [ ] Environment variables configured (DB, Redis, AWS, API key)
- [ ] Database migrations applied
- [ ] Docker images built and tested
- [ ] Nginx config deployed
- [ ] SSL certificates (if production)

---

## References

- Materials Project API: https://materialsproject.org/api
- PyG Documentation: https://pytorch-geometric.readthedocs.io/
- e3nn Docs: https://docs.e3nn.org/
- FastAPI: https://fastapi.tiangolo.com/
- Docker: https://docs.docker.com/

