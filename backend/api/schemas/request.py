from pydantic import BaseModel, Field
from typing import Optional


class StructureInput(BaseModel):
    """CIF or JSON string representing a crystal structure."""
    structure_data: str     = Field(..., description="CIF file content OR pymatgen Structure JSON string")
    format:         str     = Field("cif", description="'cif' or 'json'")
    material_id:    Optional[str] = Field(None, description="Optional identifier for tracking")


class PredictRequest(BaseModel):
    structure: StructureInput
    properties: list[str] = Field(
        default=["band_gap", "formation_energy"],
        description="List of properties to predict"
    )


class BatchScreenRequest(BaseModel):
    structures: list[StructureInput] = Field(..., max_items=10000)
    properties: list[str] = ["band_gap", "formation_energy"]
    filter_stable: bool = True     # only return energy_above_hull < 0.1 eV/atom
    band_gap_range: Optional[tuple[float, float]] = None