#!/usr/bin/env python3
"""
Download federated updates from Render and aggregate locally
This is faster than aggregating on Render's limited CPU
"""
import sys
import json
import requests
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List

SERVER_URL = "https://project-constellation.onrender.com"
AUTH_TOKEN = "constellation-token"

def get_completed_jobs():
    """Get all completed jobs"""
    print("📡 Fetching completed jobs from server...")
    response = requests.get(
        f"{SERVER_URL}/jobs",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"}
    )
    response.raise_for_status()
    jobs = response.json()
    completed = [job for job in jobs if job.get("status") == "completed"]
    print(f"✅ Found {len(completed)} completed jobs")
    return completed

def get_federated_updates_for_job(job_id: str):
    """Get federated updates for a specific job"""
    print(f"\n📡 Fetching federated updates for job {job_id}...")
    
    # Get device training assignments for this job
    response = requests.get(
        f"{SERVER_URL}/jobs/{job_id}",
        headers={"Authorization": f"Bearer {AUTH_TOKEN}"}
    )
    response.raise_for_status()
    job = response.json()
    
    # We need to get the federated updates - they're stored in federated_updates directory
    # Since we can't directly access files, we'll need to use the aggregation endpoint
    # or fetch from the database. For now, let's try to get updates via a workaround
    
    # Actually, the updates are stored locally in federated_updates/ directory
    # Let's check if we have them locally first
    local_updates_dir = Path("federated_updates")
    updates = []
    
    if local_updates_dir.exists():
        for update_file in local_updates_dir.glob("*.json"):
            try:
                with open(update_file, 'r') as f:
                    update_data = json.load(f)
                    if update_data.get("job_id") == job_id:
                        updates.append(update_data)
                        print(f"  ✅ Found update: {update_file.name}")
            except Exception as e:
                print(f"  ⚠️  Error reading {update_file}: {e}")
    
    return updates

def convert_json_weights_to_numpy(update_data: Dict) -> Dict:
    """Convert JSON model weights to numpy arrays"""
    model_weights = {}
    for key, value in update_data["model_weights"].items():
        if isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], list):
                arr = np.array(value, dtype=np.float32)
            else:
                arr = np.array(value)
                if arr.dtype in [np.int64, np.int32]:
                    model_weights[key] = arr.astype(np.int64)
                    continue
                else:
                    arr = arr.astype(np.float32)
            model_weights[key] = arr
        elif isinstance(value, (int, float)):
            if isinstance(value, int):
                model_weights[key] = np.int64(value)
            else:
                model_weights[key] = np.float32(value)
        else:
            model_weights[key] = np.array(value, dtype=np.float32)
    return model_weights

def aggregate_models_fedavg(device_updates: List[Dict]) -> Dict[str, np.ndarray]:
    """Aggregate models using Federated Averaging"""
    print(f"\n🔄 Aggregating {len(device_updates)} model updates...")
    
    # Calculate total samples
    total_samples = sum(update.get("sample_count", 1) for update in device_updates)
    print(f"📊 Total samples: {total_samples}")
    
    # Initialize aggregated weights
    aggregated_weights = {}
    first_update = device_updates[0]
    
    for layer_name in first_update["model_weights"].keys():
        first_weight = first_update["model_weights"][layer_name]
        if isinstance(first_weight, np.ndarray):
            aggregated_weights[layer_name] = np.zeros_like(first_weight, dtype=np.float32)
        else:
            aggregated_weights[layer_name] = np.zeros_like(np.array(first_weight), dtype=np.float32)
    
    # Weighted average
    for update in device_updates:
        weight_factor = update.get("sample_count", 1) / total_samples
        device_id = update.get("device_id", "unknown")
        sample_count = update.get("sample_count", 0)
        print(f"  📱 Device {device_id[:8]}...: {sample_count} samples ({weight_factor:.2%})")
        
        for layer_name, weights in update["model_weights"].items():
            if layer_name in aggregated_weights:
                if isinstance(weights, np.ndarray):
                    aggregated_weights[layer_name] += weight_factor * weights.astype(np.float32)
                else:
                    aggregated_weights[layer_name] += weight_factor * np.array(weights, dtype=np.float32)
    
    return aggregated_weights

def save_aggregated_model(weights: Dict[str, np.ndarray], job_id: str, job_name: str, metadata: Dict):
    """Save aggregated model"""
    models_dir = Path("federated_models")
    models_dir.mkdir(exist_ok=True)
    
    # Convert numpy arrays to PyTorch tensors
    aggregated_state_dict = {}
    for key, value in weights.items():
        if isinstance(value, np.ndarray):
            aggregated_state_dict[key] = torch.from_numpy(value)
        else:
            aggregated_state_dict[key] = torch.tensor(value)
    
    model_path = models_dir / f"aggregated_model_{job_id}.pth"
    
    torch.save({
        "model_state_dict": aggregated_state_dict,
        "job_id": job_id,
        "job_name": job_name,
        **metadata
    }, model_path)
    
    print(f"✅ Aggregated model saved: {model_path}")
    return str(model_path)

def main():
    print("🚀 Local Model Aggregation")
    print("=" * 60)
    
    # Get completed jobs
    completed_jobs = get_completed_jobs()
    
    if not completed_jobs:
        print("❌ No completed jobs found")
        return
    
    # Process each job
    for job in completed_jobs:
        job_id = job["id"]
        job_name = job["name"]
        
        print(f"\n{'='*60}")
        print(f"📋 Processing job: {job_name} (ID: {job_id})")
        print(f"{'='*60}")
        
        # Get federated updates for this job
        updates = get_federated_updates_for_job(job_id)
        
        if not updates:
            print(f"⚠️  No federated updates found for job {job_name}")
            continue
        
        # Convert JSON weights to numpy
        device_updates = []
        for update in updates:
            model_weights = convert_json_weights_to_numpy(update)
            device_updates.append({
                "device_id": update.get("device_id", "unknown"),
                "model_weights": model_weights,
                "sample_count": update.get("sample_count", 1000),
                "loss": update.get("loss", 0.0),
                "accuracy": update.get("accuracy", 0.0)
            })
        
        # Aggregate models
        aggregated_weights = aggregate_models_fedavg(device_updates)
        
        # Calculate metadata
        total_samples = sum(u["sample_count"] for u in device_updates)
        avg_accuracy = np.mean([u["accuracy"] for u in device_updates])
        avg_loss = np.mean([u["loss"] for u in device_updates])
        
        metadata = {
            "participating_devices": [u["device_id"] for u in device_updates],
            "total_samples": total_samples,
            "avg_accuracy": float(avg_accuracy),
            "avg_loss": float(avg_loss),
            "aggregation_strategy": "fedavg"
        }
        
        # Save aggregated model
        save_aggregated_model(aggregated_weights, job_id, job_name, metadata)
        
        print(f"✅ Completed aggregation for {job_name}")
        print(f"   📊 Total samples: {total_samples}")
        print(f"   📊 Average accuracy: {avg_accuracy:.2f}%")
        print(f"   📊 Average loss: {avg_loss:.4f}")
    
    print(f"\n{'='*60}")
    print("✅ All aggregations complete!")

if __name__ == "__main__":
    main()
