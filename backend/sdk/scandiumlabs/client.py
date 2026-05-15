"""
Scandium Labs Python SDK.

Usage:
    pip install scandiumlabs

    from scandiumlabs import Client
    client = Client(api_key="sl_xxxx", base_url="https://api.scandiumlabs.ai")

    result = client.predict("LiFePO4.cif", properties=["band_gap", "formation_energy"])
    print(result.band_gap.value)   # 3.71 eV
"""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class PropertyResult:
    value: float
    unit:  str
    physics_valid: bool
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None

    def __repr__(self) -> str:
        return f"{self.value:.4f} {self.unit} (valid={self.physics_valid})"


@dataclass
class PredictionResult:
    material_id:    Optional[str]
    formula:        str
    properties:     dict[str, PropertyResult]
    inference_ms:   float
    model_version:  str
    physics_valid:  bool
    violations:     list[str]
    warnings:       list[str]

    def __getattr__(self, name: str) -> PropertyResult:
        if name in self.properties:
            return self.properties[name]
        raise AttributeError(f"Property '{name}' not in predictions. "
                             f"Available: {list(self.properties.keys())}")


class Client:
    """Scandium Labs API client."""

    DEFAULT_BASE_URL = "https://api.scandiumlabs.ai"

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.headers  = {"X-API-Key": api_key, "Content-Type": "application/json"}
        self.timeout  = timeout

    def predict(
        self,
        structure_path: str,
        properties: list[str] = None,
        material_id: str = None,
    ) -> PredictionResult:
        """
        Predict properties for a crystal structure.

        Args:
            structure_path: Path to a .cif file.
            properties:     List of properties to predict.
            material_id:    Optional string identifier.

        Returns:
            PredictionResult with .band_gap, .formation_energy etc.
        """
        properties = properties or ["band_gap", "formation_energy"]
        path       = Path(structure_path)
        fmt        = "cif" if path.suffix.lower() == ".cif" else "json"
        data       = path.read_text()

        payload = {
            "structure": {
                "structure_data": data,
                "format": fmt,
                "material_id": material_id or path.stem,
            },
            "properties": properties,
        }

        with httpx.Client(timeout=self.timeout) as http:
            resp = http.post(
                f"{self.base_url}/predict/",
                headers=self.headers,
                json=payload,
            )
            resp.raise_for_status()

        return self._parse_response(resp.json())

    def predict_mp(self, material_id: str, properties: list[str] = None) -> PredictionResult:
        """Predict for a Materials Project entry by mp-id."""
        from mp_api.client import MPRester
        import os
        with MPRester(os.environ["MP_API_KEY"]) as mpr:
            docs = mpr.summary.search(material_ids=[material_id], fields=["structure"])
        struct_json = json.dumps(docs[0].structure.as_dict())

        payload = {
            "structure": {"structure_data": struct_json, "format": "json", "material_id": material_id},
            "properties": properties or ["band_gap", "formation_energy"],
        }
        with httpx.Client(timeout=self.timeout) as http:
            resp = http.post(f"{self.base_url}/predict/", headers=self.headers, json=payload)
            resp.raise_for_status()
        return self._parse_response(resp.json())

    def _parse_response(self, data: dict) -> PredictionResult:
        props = {
            k: PropertyResult(
                value=v["value"],
                unit=v["unit"],
                physics_valid=v["physics_valid"],
                confidence_lower=v.get("confidence_lower"),
                confidence_upper=v.get("confidence_upper"),
            )
            for k, v in data["predictions"].items()
        }
        return PredictionResult(
            material_id=data.get("material_id"),
            formula=data.get("formula", ""),
            properties=props,
            inference_ms=data["inference_time_ms"],
            model_version=data["model_version"],
            physics_valid=data["physics_valid_overall"],
            violations=data.get("violations", []),
            warnings=data.get("warnings", []),
        )