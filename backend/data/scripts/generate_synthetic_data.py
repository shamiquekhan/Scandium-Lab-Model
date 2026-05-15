import json
import random
from pymatgen.core import Structure, Lattice

def generate_synthetic_data(n=100, output_path="data/raw/mp_data.json"):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    records = []
    elements = ["Si", "C", "Fe", "Cu", "Al", "O", "Na", "Cl", "Li", "P"]
    
    for i in range(n):
        el1 = random.choice(elements)
        el2 = random.choice(elements)
        
        # Create a simple cubic structure with random lattice parameter
        a = random.uniform(3.0, 6.0)
        lattice = Lattice.cubic(a)
        
        try:
            struct = Structure(lattice, [el1, el2], [[0, 0, 0], [0.5, 0.5, 0.5]])
            
            # Synthetic physical properties
            band_gap = random.uniform(0.0, 5.0)
            formation_energy = random.uniform(-3.0, 0.5)
            energy_above_hull = random.uniform(0.0, 0.2)
            bulk_modulus = random.uniform(10.0, 300.0)
            shear_modulus = random.uniform(5.0, 150.0)
            
            rec = {
                "material_id": f"synth-{i}",
                "formula": struct.composition.reduced_formula,
                "band_gap": band_gap,
                "formation_energy_per_atom": formation_energy,
                "energy_above_hull": energy_above_hull,
                "is_stable": energy_above_hull < 0.05,
                "nsites": 2,
                "nelements": len(set([el1, el2])),
                "structure_json": struct.as_dict(),
                "bulk_modulus": bulk_modulus,
                "shear_modulus": shear_modulus,
            }
            records.append(rec)
        except Exception as e:
            print(f"Skipping index {i}: {e}")
            
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
        
    print(f"Generated {len(records)} synthetic records at {output_path}")

if __name__ == "__main__":
    generate_synthetic_data(100)
