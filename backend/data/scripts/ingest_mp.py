"""
Ingest crystal structures and target properties from the Materials Project.
Saves raw JSON files to backend/data/raw/.
Run: python -m data.scripts.ingest_mp --n 5000 --elements Li Fe P O Si
"""
import os, json, argparse
from pathlib import Path
from tqdm import tqdm
from mp_api.client import MPRester

RAW_DIR = Path("backend/data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "material_id", "structure", "band_gap",
    "formation_energy_per_atom", "energy_above_hull",
    "is_stable", "nsites", "symmetry",
]

def ingest(n: int, elements: list[str] = None):
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise ValueError("Set MP_API_KEY environment variable")

    with MPRester(api_key) as mpr:
        kwargs = {"fields": FIELDS}
        if elements:
            kwargs["elements"] = elements
        docs = mpr.materials.summary.search(**kwargs)

    saved = 0
    for doc in tqdm(docs[:n], desc="Ingesting structures"):
        try:
            record = {
                "material_id": doc.material_id,
                "structure": doc.structure.as_dict(),
                "band_gap": float(doc.band_gap),
                "formation_energy_per_atom": float(doc.formation_energy_per_atom),
                "energy_above_hull": float(doc.energy_above_hull),
                "is_stable": bool(doc.is_stable),
                "nsites": doc.nsites,
            }
            path = RAW_DIR / f"{doc.material_id}.json"
            with open(path, "w") as f:
                json.dump(record, f)
            saved += 1
        except Exception as e:
            print(f"Skip {doc.material_id}: {e}")

    print(f"Saved {saved} structures to {RAW_DIR}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--elements", nargs="*")
    args = p.parse_args()
    ingest(args.n, args.elements)
