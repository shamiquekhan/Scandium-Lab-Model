"""
Download and featurise 150,000 Materials Project structures for v2 training.

Runtime:  ~2–3 hours depending on API rate limits and internet speed.
Output:   data/processed_v2/train.pt  val.pt  test.pt
Storage:  ~8 GB after featurisation.

Usage:
    python scripts/ingest_mp_150k.py --n 150000 --out data/processed_v2
"""

from __future__ import annotations
import argparse
import os
import random
import json
from pathlib import Path

import torch
from mp_api.client import MPRester
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scandium.data.featurize_v2 import structure_to_graph_v2


# ── MP query filters ──────────────────────────────────────────────────────────
# These reproduce the training distribution of CGCNN and ALIGNN for comparability.
QUERY_FIELDS = [
    "material_id", "formula_pretty", "structure",
    "band_gap", "formation_energy_per_atom", "energy_above_hull",
    "is_stable", "nsites",
]


def download_and_featurise(
    api_key:    str,
    n_target:   int,
    output_dir: str,
    cutoff:     float = 6.0,
    max_nbrs:   int   = 12,
    seed:       int   = 42,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[Ingest] Connecting to Materials Project...")
    print(f"         Target: {n_target:,} structures  cutoff={cutoff} Å  max_nbrs={max_nbrs}")

    graphs = []
    skipped = 0

    with MPRester(api_key) as mpr:
        # Pull all materials with DFT band gap and formation energy available
        # energy_above_hull < 0.2 eV/atom: filters to experimentally realistic compounds
        docs = mpr.summary.search(
            energy_above_hull=(0.0, 0.2),
            fields=QUERY_FIELDS,
            chunk_size=1000,
        )
        total_available = len(docs)
        print(f"[Ingest] {total_available:,} documents available from MP.")

        # Subsample if more available than needed
        if total_available > n_target:
            random.seed(seed)
            docs = random.sample(docs, n_target)
            print(f"[Ingest] Subsampled to {len(docs):,} documents.")

        print(f"[Ingest] Featurising structures (this takes a while)...")
        for doc in tqdm(docs):
            try:
                # Skip if any required label is missing
                if doc.band_gap is None or doc.formation_energy_per_atom is None:
                    skipped += 1
                    continue
                if doc.energy_above_hull is None:
                    skipped += 1
                    continue

                label = {
                    "band_gap":                  float(doc.band_gap),
                    "formation_energy_per_atom": float(doc.formation_energy_per_atom),
                    "energy_above_hull":         float(doc.energy_above_hull),
                }

                graph = structure_to_graph_v2(
                    doc.structure, label=label,
                    cutoff=cutoff, max_neighbors=max_nbrs,
                )
                graph.material_id = doc.material_id
                graphs.append(graph)

            except Exception as e:
                skipped += 1
                # Don't print every skip — just count

    print(f"[Ingest] Built {len(graphs):,} graphs. Skipped {skipped:,}.")

    # ── Stratified split ─────────────────────────────────────────────────────
    random.seed(seed)
    random.shuffle(graphs)

    n_total = len(graphs)
    n_train = int(n_total * 0.80)
    n_val   = int(n_total * 0.10)

    train_graphs = graphs[:n_train]
    val_graphs   = graphs[n_train:n_train + n_val]
    test_graphs  = graphs[n_train + n_val:]

    # ── Save as .pt files ─────────────────────────────────────────────────────
    for split_name, split_data in [
        ("train", train_graphs),
        ("val",   val_graphs),
        ("test",  test_graphs),
    ]:
        path = out / f"{split_name}.pt"
        torch.save(split_data, path)
        print(f"  Saved {len(split_data):,} -> {path}  ({path.stat().st_size / 1e9:.2f} GB)")

    # ── Save split metadata ───────────────────────────────────────────────────
    meta = {
        "total": n_total, "train": n_train,
        "val": len(val_graphs), "test": len(test_graphs),
        "skipped": skipped, "cutoff": cutoff, "max_neighbors": max_nbrs,
        "edge_feat_dim": 56,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[Ingest] Done. Meta: {meta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",   type=int, default=150000)
    parser.add_argument("--out", type=str, default="data/processed_v2")
    args = parser.parse_args()

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: Set MP_API_KEY in your .env file.")

    download_and_featurise(api_key, args.n, args.out)
