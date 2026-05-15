"""
Model version registry.
Tracks which checkpoint is active, its training metadata, and performance metrics.
Enables safe rollback: set ACTIVE_MODEL_VERSION in .env to switch models.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict


REGISTRY_PATH = Path("backend/checkpoints/version_registry.json")


@dataclass
class ModelVersion:
    version:       str
    checkpoint:    str
    val_bg_mae:    float
    val_fe_mae:    float
    n_train:       int
    data_source:   str
    n_ensemble:    int
    is_active:     bool = False
    notes:         str  = ""


class VersionRegistry:
    def __init__(self):
        self.versions: dict[str, ModelVersion] = {}
        self._load()

    def _load(self) -> None:
        if REGISTRY_PATH.exists():
            with open(REGISTRY_PATH) as f:
                raw = json.load(f)
            self.versions = {k: ModelVersion(**v) for k, v in raw.items()}

    def _save(self) -> None:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_PATH, "w") as f:
            json.dump({k: asdict(v) for k, v in self.versions.items()}, f, indent=2)

    def register(self, v: ModelVersion) -> None:
        self.versions[v.version] = v
        self._save()
        print(f"[Registry] Registered model version '{v.version}'")

    def activate(self, version: str) -> None:
        for v in self.versions.values():
            v.is_active = False
        if version not in self.versions:
            raise KeyError(f"Version '{version}' not in registry.")
        self.versions[version].is_active = True
        self._save()
        print(f"[Registry] Activated model version '{version}'")

    def get_active(self) -> ModelVersion | None:
        for v in self.versions.values():
            if v.is_active:
                return v
        return None

    def list(self) -> list[ModelVersion]:
        return sorted(self.versions.values(), key=lambda v: v.version)
