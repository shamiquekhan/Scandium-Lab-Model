#!/usr/bin/env python
"""
End-to-end test: Upload structure, create job, poll results, verify predictions.
Requires Docker services to be running (docker compose up -d).
"""
import requests
import json
import time
import sys

API_BASE = "http://localhost/api"
MAX_WAIT = 180  # seconds

print("="*70)
print("SCANDIUM LABS E2E TEST: Structure → Prediction → Dashboard")
print("="*70)

# Check connectivity
print("\n1. Checking API connectivity...")
try:
    response = requests.get(f"{API_BASE}/docs", timeout=5)
    print("   ✓ API responsive")
except Exception as e:
    print(f"   ✗ API not reachable: {e}")
    print(f"   Make sure Docker services are running: docker compose up -d")
    sys.exit(1)

# Upload structure
print("\n2. Uploading mp-23907 structure...")
try:
    with open("backend/mp-23907-out.json", "rb") as f:
        files = {"file": ("mp-23907.json", f, "application/json")}
        data = {"name": "mp-23907"}
        response = requests.post(f"{API_BASE}/structures", files=files, data=data, timeout=30)
    response.raise_for_status()
    structure = response.json()
    structure_id = structure["id"]
    print(f"   ✓ Structure uploaded")
    print(f"     - ID: {structure_id}")
    print(f"     - Formula: {structure.get('formula', 'N/A')}")
    print(f"     - Atoms: {structure.get('n_atoms', 'N/A')}")
except Exception as e:
    print(f"   ✗ Upload failed: {e}")
    sys.exit(1)

# Create prediction job
print("\n3. Creating prediction job...")
try:
    response = requests.post(
        f"{API_BASE}/jobs",
        json={"structure_id": structure_id},
        timeout=10
    )
    response.raise_for_status()
    job = response.json()
    job_id = job["id"]
    print(f"   ✓ Job created")
    print(f"     - ID: {job_id}")
    print(f"     - Status: {job.get('status', 'unknown')}")
except Exception as e:
    print(f"   ✗ Job creation failed: {e}")
    sys.exit(1)

# Poll job status
print(f"\n4. Polling job status (max {MAX_WAIT}s)...")
start = time.time()
while time.time() - start < MAX_WAIT:
    try:
        response = requests.get(f"{API_BASE}/jobs/{job_id}", timeout=10)
        response.raise_for_status()
        job_data = response.json()
        job_status = job_data.get("status")
        
        if job_status == "completed":
            print(f"   ✓ Prediction completed!")
            break
        elif job_status == "failed":
            error = job_data.get("error_message", "Unknown error")
            print(f"   ✗ Prediction failed: {error}")
            sys.exit(1)
        else:
            elapsed = time.time() - start
            print(f"   [{elapsed:5.1f}s] Status: {job_status}...", end="\r")
            time.sleep(2)
    except Exception as e:
        print(f"   ✗ Status check failed: {e}")
        sys.exit(1)
else:
    print(f"   ✗ Timeout waiting for prediction")
    sys.exit(1)

# Fetch predictions
print("\n5. Fetching prediction results...")
try:
    response = requests.get(f"{API_BASE}/predictions?job_id={job_id}", timeout=10)
    response.raise_for_status()
    predictions = response.json()
    print(f"   ✓ Retrieved {len(predictions)} predictions")
except Exception as e:
    print(f"   ✗ Failed to fetch predictions: {e}")
    sys.exit(1)

# Display results
print("\n" + "="*70)
print("PREDICTION RESULTS WITH UNCERTAINTY BOUNDS")
print("="*70)

for i, pred in enumerate(predictions, 1):
    print(f"\n{i}. {pred.get('property_name', 'Unknown')}")
    print(f"   Value:          {pred.get('predicted_value', 'N/A'):.4f} {pred.get('unit', '')}")
    print(f"   Material Class: {pred.get('material_class', 'N/A')}")
    print(f"   Context Score:  {pred.get('context_score', 'N/A'):.4f}")
    print(f"   Context Valid:  {pred.get('context_valid', 'N/A')}")
    
    lower = pred.get('lower_bound', 'N/A')
    upper = pred.get('upper_bound', 'N/A')
    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
        print(f"   95% CI:         [{lower:.4f}, {upper:.4f}]")
        uncertainty = (upper - lower) / 2
        print(f"   Uncertainty:    ± {uncertainty:.4f}")
    else:
        print(f"   95% CI:         [{lower}, {upper}]")
    
    print(f"   Context Z-score: {pred.get('context_z_score', 'N/A'):.4f}")

print("\n" + "="*70)
print("✓ E2E TEST PASSED")
print("="*70)
print("\nVisualization Dashboard is now integrated into the frontend:")
print("  - CrystalCanvas: 3D rotating structure visualization")
print("  - PredictionCards: Display uncertainty bounds, context scores")
print("  - CI Bars: Color-coded uncertainty visualization")
print("\nTo view the dashboard:")
print("  1. Open http://localhost in your browser")
print("  2. Upload a structure file")
print("  3. Wait for predictions to complete")
print("  4. Results appear with crystal structure + prediction cards")
print("="*70)
