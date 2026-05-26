# Scandium Labs: PIGNet V2 Model

This repository contains the production-grade, physics-informed Graph Neural Network (PIGNet V2) architecture for Scandium Labs. It is designed to predict crystal material properties (Band Gap, Formation Energy, Energy Above Hull) using 56-dimensional 3-body angular featurization.

## Hardware Requirements
This model is designed for high-performance training. 
- **Minimum**: NVIDIA RTX 3090 / 4090 (24GB VRAM)
- **Recommended**: NVIDIA A100 (40GB/80GB VRAM) or equivalent.
- **RAM**: 64GB+ (the full dataset graph tensors require ~10-12GB of system RAM to load).

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/shamiquekhan/Scandium-Lab-Model.git
   cd Scandium-Lab-Model/backend
   ```

2. **Setup the Python Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install torch-geometric pymatgen mp-api python-dotenv tqdm pydantic fastapi uvicorn
   ```

3. **Configure the Materials Project API**
   Create a `.env` file in the `backend/` directory and add your Materials Project API key:
   ```env
   MP_API_KEY="your_api_key_here"
   ```

## Production Training Pipeline

### Step 1: Data Ingestion (150k Structures)
You must download and featurize the dataset. This script pulls all stable/near-stable structures from the Materials Project and runs the V2 angular featurization.
```bash
# Run this inside the backend/ directory
python scripts/ingest_mp_150k.py --n 150000 --out data/processed_v2
```
*Note: This process takes ~2-3 hours and will generate ~12GB of `.pt` files. The dataset is excluded from git via `.gitignore`.*

### Step 2: Full Deep Ensemble Training
Once the data is ingested, start the 5-seed ensemble training. This provides robust uncertainty quantification via Monte Carlo dropout and deep ensembling.
```bash
python scripts/train_ensemble.py --data_dir data/processed_v2
```
*Note: This will train 5 models sequentially for 300 epochs each. Expected runtime on an A100 is ~40-50 hours total. Checkpoints will be saved in `backend/checkpoints/`.*

### Step 3: Conformal Calibration
After all 5 seeds finish training, run the calibration script to generate guaranteed conformal prediction intervals for the API layer.
```bash
python scripts/calibrate_conformal.py --data_dir data/processed_v2
```

## Architecture Notes
- **PIGNetV2**: Located in `scandium/models/pignet_v2.py`. Features attention-gated messages and dual global pooling.
- **Normalizer**: Targets are zero-mean, unit-variance scaled to stabilize gradients.
- **Physics Loss**: Softplus constraints ensure strictly positive predictions for Band Gap and Energy Above Hull.

Happy Training! 🚀
