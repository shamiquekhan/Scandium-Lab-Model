"""
Download crystal structures and DFT properties from Materials Project.
Usage:
    python data/scripts/download_mp.py --n_materials 50000 --output data/raw/mp_data.json
"""

import argparse
import json
import os
from pathlib import Path

from mp_api.client import MPRester
from tqdm import tqdm


PROPERTIES = [
    "material_id",
    "formula_pretty",
    "structure",
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "bulk_modulus",
    "shear_modulus",
    "total_magnetization",
    "is_stable",
    "symmetry",
    "elements",
    "nelements",
    "nsites",
    "volume",
    "density",
]


def download(api_key: str, n: int, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[MP] Connecting to Materials Project API...")
    records = []

    with MPRester(api_key) as mpr:
        # Pull semiconductors and insulators (band_gap > 0) — most relevant
        docs = mpr.summary.search(
            band_gap=(0.01, 10.0),
            energy_above_hull=(0, 0.1),   # stable or near-stable
            fields=PROPERTIES,
            num_chunks=None,
            chunk_size=1000,
        )

        print(f"[MP] Retrieved {len(docs)} documents. Serialising...")
        for doc in tqdm(docs[:n]):
            try:
                rec = {
                    "material_id": doc.material_id,
                    "formula": doc.formula_pretty,
                    "band_gap": float(doc.band_gap) if doc.band_gap is not None else None,
                    "formation_energy_per_atom": (
                        float(doc.formation_energy_per_atom)
                        if doc.formation_energy_per_atom is not None else None
                    ),
                    "energy_above_hull": (
                        float(doc.energy_above_hull)
                        if doc.energy_above_hull is not None else None
                    ),
                    "is_stable": doc.is_stable,
                    "nsites": doc.nsites,
                    "nelements": doc.nelements,
                    "structure_json": doc.structure.as_dict(),
                }

                # Optional: moduli (not always present)
                if hasattr(doc, "bulk_modulus") and doc.bulk_modulus:
                    rec["bulk_modulus"] = float(doc.bulk_modulus.get("voigt", 0) or 0)
                if hasattr(doc, "shear_modulus") and doc.shear_modulus:
                    rec["shear_modulus"] = float(doc.shear_modulus.get("voigt", 0) or 0)

                records.append(rec)
            except Exception as e:
                print(f"[WARN] Skipping {doc.material_id}: {e}")

    with open(output_path, "w") as f:
        json.dump(records, f)

    print(f"[MP] Saved {len(records)} records -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_materials", type=int, default=50000)
    parser.add_argument("--output", type=str, default="data/raw/mp_data.json")
    args = parser.parse_args()

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise ValueError("Set MP_API_KEY in your .env file")

    download(api_key, args.n_materials, args.output)