from pydantic import BaseModel
from typing import Optional


class PropertyPrediction(BaseModel):
    value: float
    unit:  str
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None
    physics_valid: bool


class PredictResponse(BaseModel):
    material_id:   Optional[str]
    formula:       Optional[str]
    predictions:   dict[str, PropertyPrediction]
    inference_time_ms: float
    model_version: str
    physics_valid_overall: bool
    violations: list[str]
    warnings:   list[str]


class BatchScreenResponse(BaseModel):
    total_submitted: int
    total_completed: int
    results: list[PredictResponse]
    batch_time_seconds: float