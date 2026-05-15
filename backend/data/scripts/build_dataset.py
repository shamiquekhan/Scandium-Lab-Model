"""Pre-process all raw JSONs to PyG .pt files. Run once before training."""
import json, torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pymatgen.core import Structure
from scandium.data.featurize import structure_to_graph

RAW = Path("backend/data/raw")
OUT = Path("backend/data/processed")
OUT.mkdir(exist_ok=True)

def process_one(fpath):
    stem = fpath.stem
    out_path = OUT / f"{stem}.pt"
    if out_path.exists(): return  # skip if already processed
    try:
        with open(fpath) as f: rec = json.load(f)
        struct = Structure.from_dict(rec["structure"])
        label = {k: rec[k] for k in ["band_gap", "formation_energy_per_atom", "energy_above_hull"]}
        data = structure_to_graph(struct, label=label)
        data.material_id = rec["material_id"]
        torch.save(data, out_path)
    except Exception as e:
        print(f"Failed {stem}: {e}")

def main():
    files = list(RAW.glob("*.json"))
    print(f"Processing {len(files)} files...")
    with ProcessPoolExecutor(max_workers=8) as exe:
        list(tqdm(exe.map(process_one, files), total=len(files)))

if __name__ == "__main__":
    main()