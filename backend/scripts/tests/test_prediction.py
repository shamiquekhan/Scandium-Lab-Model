import requests
import json
import time

API_BASE = "http://localhost/api"

# Read the converted structure JSON
with open("backend/mp-23907-out.json", "r") as f:
    structure_data = json.load(f)

# Upload structure and get job ID
print("Uploading structure...")
with open("backend/mp-23907-out.json", "rb") as f:
    files = {"file": ("mp-23907.json", f)}
    data = {"name": "mp-23907"}
    upload_response = requests.post(f"{API_BASE}/structures", files=files, data=data)
upload_response.raise_for_status()
structure_id = upload_response.json()["id"]
print(f"✓ Structure uploaded, ID: {structure_id}")

# Create prediction job
print("Creating prediction job...")
job_response = requests.post(
    f"{API_BASE}/jobs",
    json={"structure_id": structure_id}
)
job_response.raise_for_status()
job_id = job_response.json()["id"]
job_status = job_response.json()["status"]
print(f"✓ Job created, ID: {job_id}, Status: {job_status}")

# Poll job status
print("Waiting for prediction...")
max_wait = 120
start = time.time()
while time.time() - start < max_wait:
    status_response = requests.get(f"{API_BASE}/jobs/{job_id}")
    status_response.raise_for_status()
    job_data = status_response.json()
    job_status = job_data.get("status")
    
    if job_status == "completed":
        print(f"✓ Prediction completed!")
        break
    elif job_status == "failed":
        error = job_data.get("error", "Unknown error")
        print(f"✗ Prediction failed: {error}")
        exit(1)
    else:
        print(f"  Status: {job_status}...", end="\r")
        time.sleep(2)
else:
    print(f"✗ Timeout waiting for prediction (>{max_wait}s)")
    exit(1)

# Get predictions
print("\nFetching predictions...")
pred_response = requests.get(f"{API_BASE}/predictions?job_id={job_id}")
pred_response.raise_for_status()
predictions = pred_response.json()

print("\n" + "="*60)
print("PREDICTION RESULTS")
print("="*60)

for i, pred in enumerate(predictions, 1):
    print(f"\nProperty {i}: {pred.get('property_name', 'Unknown')}")
    print(f"  Value:          {pred.get('predicted_value', 'N/A')}")
    print(f"  Unit:           {pred.get('unit', 'N/A')}")
    print(f"  Material Class: {pred.get('material_class', 'N/A')}")
    print(f"  Context Score:  {pred.get('context_score', 'N/A'):.3f}")
    print(f"  Context Valid:  {pred.get('context_valid', 'N/A')}")
    print(f"  Uncertainty (95% CI):")
    print(f"    Lower Bound:  {pred.get('lower_bound', 'N/A')}")
    print(f"    Upper Bound:  {pred.get('upper_bound', 'N/A')}")
    if pred.get('constraint_details'):
        print(f"  Constraint Details:")
        for key, val in pred.get('constraint_details', {}).items():
            print(f"    {key}: {val}")

print("\n" + "="*60)
