"""
Generate synthetic crystal structures for pipeline testing.
Run: python -m data.scripts.synthetic_data --n 200
"""
import json, random, argparse
import numpy as np
from pathlib import Path
from pymatgen.core import Structure, Lattice, Element

RAW_DIR = Path("backend/data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Preset templates: (elements, wyckoff_coords, lattice_params)
TEMPLATES = [
    ("Si",  ["Si", "Si"],  [[0.0,0.0,0.0],[0.25,0.25,0.25]], (5.43,5.43,5.43,90,90,90), 1.12, 0.0),
    ("NaCl", ["Na", "Cl"], [[0.0,0.0,0.0],[0.5,0.5,0.5]],  (5.64,5.64,5.64,90,90,90), 8.5, -3.37),
    ("GaN",  ["Ga", "N"],  [[0.333,0.667,0.0],[0.333,0.667,0.377]], (3.19,3.19,5.19,90,90,120), 3.4, -1.24),
    ("TiO2", ["Ti", "O", "O"], [[0.0,0.0,0.0],[0.305,0.305,0.0],[-0.305,-0.305,0.0]], (4.59,4.59,2.96,90,90,90), 3.0, -3.18),
]

def make_structure(template, noise=0.05):
    name, species, coords, lat_params, bg, fe = template
    lp = [p * (1 + random.uniform(-noise, noise)) for p in lat_params]
    lattice = Lattice.from_parameters(*lp)
    noisy_coords = [[c + random.uniform(-0.02, 0.02) for c in coord] for coord in coords]
    structure = Structure(lattice, species, noisy_coords)
    return structure, bg, fe

def generate(n: int):
    for i in range(n):
        tpl = random.choice(TEMPLATES)
        structure, bg, fe = make_structure(tpl, noise=0.08)
        record = {
            "material_id": f"syn-{i:04d}",
            "structure": structure.as_dict(),
            "band_gap": max(0.0, bg + random.gauss(0, 0.3)),
            "formation_energy_per_atom": fe + random.gauss(0, 0.1),
            "energy_above_hull": max(0.0, random.gauss(0.05, 0.05)),
            "is_stable": True,
            "nsites": structure.num_sites,
        }
        with open(RAW_DIR / f"syn-{i:04d}.json", "w") as f:
            json.dump(record, f)
    print(f"Generated {n} synthetic structures -> {RAW_DIR}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()
    generate(args.n)
