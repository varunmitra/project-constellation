#!/usr/bin/env python3
"""
Aggregate models from multiple completed training jobs
"""
import sys
import json
import requests
from pathlib import Path

SERVER_URL = "https://project-constellation.onrender.com"
AUTH_TOKEN = "constellation-token"

def get_completed_jobs():
    """Get all completed jobs"""
    response = requests.get(
        f"{SERVER_URL}/jobs",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"}
    )
    response.raise_for_status()
    jobs = response.json()
    return [job for job in jobs if job.get("status") == "completed"]

def aggregate_job_models(job_id):
    """Aggregate models for a specific job"""
    response = requests.post(
        f"{SERVER_URL}/federated/aggregate/{job_id}",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"}
    )
    if response.status_code == 200:
        return response.json()
    else:
        print(f"⚠️  Job {job_id}: {response.status_code} - {response.text}")
        return None

def main():
    print("🔄 Aggregating Models from Completed Training Jobs")
    print("=" * 60)
    
    # Get completed jobs
    completed_jobs = get_completed_jobs()
    print(f"\n📊 Found {len(completed_jobs)} completed jobs")
    
    for job in completed_jobs:
        print(f"\n🔄 Aggregating models for job: {job['name']} (ID: {job['id']})")
        result = aggregate_job_models(job['id'])
        if result:
            print(f"✅ {result.get('status', 'Aggregated')}")
            if 'aggregated_model_path' in result:
                print(f"   Model saved: {result['aggregated_model_path']}")
        else:
            print(f"⚠️  Could not aggregate job {job['name']}")
    
    print("\n✅ Aggregation complete!")

if __name__ == "__main__":
    main()
