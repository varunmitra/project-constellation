#!/usr/bin/env python3
"""
Verify if federated learning worked for completed jobs
Checks if weights were uploaded and if aggregation is possible
"""

import sys
import json
from pathlib import Path

# Try to import requests, with helpful error message
try:
    import requests
except ImportError:
    print("❌ Error: 'requests' module not found")
    print("")
    print("💡 To fix this, run:")
    print("   source venv/bin/activate")
    print("   pip install requests")
    print("")
    print("Or install globally:")
    print("   pip3 install requests")
    sys.exit(1)

SERVER_URL = "http://localhost:8000"
AUTH_HEADER = {"Authorization": "Bearer constellation-token"}

def main():
    print("🔍 Verifying Federated Learning Status")
    print("=" * 60)
    
    # Check for completed jobs
    response = requests.get(f"{SERVER_URL}/jobs", headers=AUTH_HEADER)
    if response.status_code != 200:
        print("❌ Failed to get jobs")
        return
    
    jobs = response.json()
    completed_jobs = [j for j in jobs if j.get('status') == 'completed']
    
    print(f"\n📊 Found {len(completed_jobs)} completed job(s)")
    
    if not completed_jobs:
        print("⚠️ No completed jobs to verify")
        return
    
    # Check each completed job
    for job in completed_jobs[-3:]:  # Check last 3 jobs
        job_id = job['id']
        job_name = job['name']
        
        print(f"\n{'='*60}")
        print(f"Job: {job_name}")
        print(f"ID: {job_id}")
        print('='*60)
        
        # Check for federated updates
        project_root = Path(__file__).parent
        updates_dir = project_root / "federated_updates"
        
        if not updates_dir.exists():
            print("❌ federated_updates directory doesn't exist")
            print("   This means weights were NOT uploaded")
            continue
        
        # Find updates for this job
        update_files = list(updates_dir.glob(f"*_{job_id[:8]}*.json"))
        if not update_files:
            # Try to find by assignment ID
            update_files = [f for f in updates_dir.glob("*.json") if job_id in f.read_text()]
        
        if update_files:
            print(f"✅ Found {len(update_files)} weight update(s)")
            for update_file in update_files:
                with open(update_file) as f:
                    data = json.load(f)
                print(f"   - Device: {data.get('device_id', 'unknown')}")
                print(f"     Samples: {data.get('sample_count', 0)}")
                print(f"     Accuracy: {data.get('accuracy', 0):.2f}%")
                print(f"     Loss: {data.get('loss', 0):.4f}")
                print(f"     Layers: {len(data.get('model_weights', {}))}")
        else:
            print("❌ No weight updates found for this job")
            print("   Weights were NOT uploaded")
        
        # Check for aggregated model
        models_dir = project_root / "federated_models"
        if models_dir.exists():
            model_files = list(models_dir.glob(f"*{job_id}*.pth"))
            if model_files:
                print(f"✅ Found aggregated model: {model_files[0].name}")
            else:
                print("⚠️ No aggregated model found")
                print("   Run: POST /federated/aggregate/{job_id}")
        else:
            print("⚠️ federated_models directory doesn't exist")
            print("   No aggregation has been done yet")
    
    print(f"\n{'='*60}")
    print("📋 Summary")
    print('='*60)
    
    updates_dir = Path("federated_updates")
    if updates_dir.exists():
        update_count = len(list(updates_dir.glob("*.json")))
        print(f"✅ Weight updates: {update_count} file(s)")
    else:
        print("❌ Weight updates: 0 (directory doesn't exist)")
    
    models_dir = Path("federated_models")
    if models_dir.exists():
        model_count = len(list(models_dir.glob("*.pth")))
        print(f"✅ Aggregated models: {model_count} file(s)")
    else:
        print("❌ Aggregated models: 0 (directory doesn't exist)")
    
    print("\n💡 To make models intelligent:")
    print("   1. Train multiple devices on same job")
    print("   2. Weights will be uploaded automatically")
    print("   3. Call: POST /federated/aggregate/{job_id}")
    print("   4. Aggregated model combines all devices' knowledge")

if __name__ == "__main__":
    main()

