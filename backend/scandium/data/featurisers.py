"""
Atom and bond featurisers.
Each atom becomes a 92-dimensional feature vector encoding
element-level physics: electronegativity, atomic radius,
valence electrons, period, group, and one-hot atomic number.
"""

from __future__ import annotations
import numpy as np
from pymatgen.core import Element


# Precomputed element lookup table (atomic number → features)
# Columns: electronegativity, covalent_radius_Å, valence_electrons, period, group
_ELEMENT_TABLE: dict[int, list[float]] = {}

def _build_table() -> None:
    for Z in range(1, 95):
        try:
            el = Element.from_Z(Z)
            en   = float(el.X)                  if el.X is not None else 0.0
            cr   = float(el.atomic_radius or 0) * 100  # Å → pm for normalisation
            val  = float(el.nvalence())          if hasattr(el, "nvalence") else 0.0
            per  = float(el.row)
            grp  = float(el.group if el.group else 18)
            _ELEMENT_TABLE[Z] = [en, cr, val, per, grp]
        except Exception:
            _ELEMENT_TABLE[Z] = [0.0, 0.0, 0.0, 0.0, 0.0]

_build_table()

MAX_Z = 94


class AtomFeaturiser:
    """Converts atomic number to fixed-length feature vector."""

    FEATURE_DIM: int = MAX_Z + 5   # one-hot (94) + 5 physics features = 99

    def featurise(self, atomic_number: int) -> np.ndarray:
        onehot = np.zeros(MAX_Z, dtype=np.float32)
        if 1 <= atomic_number <= MAX_Z:
            onehot[atomic_number - 1] = 1.0

        physics = np.array(_ELEMENT_TABLE.get(atomic_number, [0.0] * 5), dtype=np.float32)
        # Normalise physics features to [0, 1] approximate range
        physics[0] /= 4.0    # electronegativity max ~4
        physics[1] /= 300.0  # covalent radius max ~300 pm
        physics[2] /= 18.0   # valence electrons max 18
        physics[3] /= 7.0    # period max 7
        physics[4] /= 18.0   # group max 18

        return np.concatenate([onehot, physics])