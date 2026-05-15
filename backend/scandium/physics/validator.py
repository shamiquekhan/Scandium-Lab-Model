"""
Post-prediction physics validator.
Runs after model inference to flag any remaining physical inconsistencies.
Returns a structured validation report per prediction.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ValidationReport:
    is_valid: bool = True
    violations: list[str] = field(default_factory=list)
    warnings: list[str]   = field(default_factory=list)


class PhysicsValidator:
    """
    Validates a single prediction dict for physical consistency.
    Called by the API on every prediction before returning to the client.
    """

    # Known physical bounds for common material families
    BAND_GAP_MAX_EV      = 14.0   # widest known gap (diamond-like) ~12 eV
    BULK_MODULUS_MAX_GPA = 1000.0 # diamond ~440 GPa, theoretical limit higher
    SHEAR_MODULUS_MAX_GPA = 600.0
    FORMATION_ENERGY_MIN = -10.0  # eV/atom — no known compound below this
    FORMATION_ENERGY_MAX =  5.0   # positive but extremely high is suspicious

    def validate(self, preds: dict[str, float]) -> ValidationReport:
        report = ValidationReport()

        if "band_gap" in preds:
            bg = preds["band_gap"]
            if bg < 0.0:
                report.is_valid = False
                report.violations.append(
                    f"Band gap is negative ({bg:.4f} eV) — physically impossible"
                )
            elif bg > self.BAND_GAP_MAX_EV:
                report.warnings.append(
                    f"Band gap ({bg:.2f} eV) exceeds typical maximum — verify structure"
                )

        if "formation_energy" in preds:
            fe = preds["formation_energy"]
            if fe < self.FORMATION_ENERGY_MIN:
                report.is_valid = False
                report.violations.append(
                    f"Formation energy ({fe:.4f} eV/atom) below physical minimum"
                )
            elif fe > self.FORMATION_ENERGY_MAX:
                report.warnings.append(
                    f"Formation energy ({fe:.4f} eV/atom) is very high — likely unstable"
                )

        if "energy_above_hull" in preds:
            hull = preds["energy_above_hull"]
            if hull < 0.0:
                report.is_valid = False
                report.violations.append(
                    f"Energy above hull is negative ({hull:.4f} eV/atom) — impossible by definition"
                )

        if "bulk_modulus" in preds:
            bm = preds["bulk_modulus"]
            if bm < 0.0:
                report.is_valid = False
                report.violations.append(f"Bulk modulus is negative ({bm:.2f} GPa)")
            elif bm > self.BULK_MODULUS_MAX_GPA:
                report.warnings.append(f"Bulk modulus ({bm:.2f} GPa) exceeds diamond — verify")

        if "shear_modulus" in preds:
            sm = preds["shear_modulus"]
            if sm < 0.0:
                report.is_valid = False
                report.violations.append(f"Shear modulus is negative ({sm:.2f} GPa)")

        return report