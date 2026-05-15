from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from ..services.predictor import predictor

router = APIRouter(prefix="/api", tags=["inference"])

class CifBody(BaseModel):
    cif_content: str
    uncertainty: bool = True

@router.post("/predict")
async def predict_structure(body: CifBody):
    """
    POST /api/predict
    Body: { "cif_content": "data_...", "uncertainty": true }
    Returns predictions, uncertainty, violations, structure_info
    """
    try:
        result = predictor.predict_from_cif(body.cif_content, body.uncertainty)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

@router.post("/predict/file")
async def predict_cif_file(file: UploadFile = File(...)):
    """Upload a .cif file directly."""
    content = (await file.read()).decode("utf-8")
    try:
        return predictor.predict_from_cif(content)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

@router.get("/health")
async def health():
    import torch
    return {
        "status":         "ok",
        "model_loaded":   predictor.model is not None,
        "device":         predictor.device,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name":       torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }