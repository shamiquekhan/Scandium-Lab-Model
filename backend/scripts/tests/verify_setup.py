#!/usr/bin/env python
"""Verify structure parser and prepare for full E2E test."""
import json
import sys
sys.path.insert(0, 'backend')

from app.services.structure_parser import parse_structure_file

# Test parser with converted JSON
print("Testing structure parser...")
with open('backend/mp-23907-out.json', 'rb') as f:
    data = f.read()

result = parse_structure_file(data, '.json')
print(f"✓ Parser OK")
print(f"  Formula: {result['formula']}")
print(f"  Space Group: {result['space_group']}")
print(f"  Atoms: {result['n_atoms']}")
print(f"  Structure JSON size: {len(str(result['structure_json']))} chars")
print("\n✓ All parser checks passed. Ready for E2E test once Docker starts.")
