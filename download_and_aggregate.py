#!/usr/bin/env python3
"""
Download federated updates from Render and aggregate locally
This avoids memory issues on Render and is much faster
"""
import sys
import json
import requests
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
from datetime import datetime

SERVER_URL = "https://project-constellation.onrender.com"
AUTH_TOKEN = "constellation-token"
HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}"}

def get_completed_jobs():
    """Get all completed jobs"""
    print("📡 Fetching completed jobs from server...")
    try:
        response = requests.get(f"{SERVER_URL}/jobs", headers=HEADERS, timeout=30)
        response.raise_for_status()
        jobs = response.json()
        completed = [job for job in jobs if job.get("status") == "completed"]
        print(f"✅ Found {len(completed)} completed jobs")
        return completed
    except Exception as e:
        print(f"❌ Error fetching jobs: {e}")
        return []

def get_federated_updates_for_job(job_id: str):
    """Get list of federated update files for a job"""
    print(f"\n📡 Fetching federated updates for job {job_id}...")
    try:
        response = requests.get(
            f"{SERVER_URL}/federated/updates/{job_id}",
            headers=HEADERS,
            timeout=30
        )
        if response.status_code == 404:
            print(f"  ⚠️  No federated updates found for this job")
            return []
        response.raise_for_status()
        data = response.json()
        return data.get("update_files", [])
    except Exception as e:
        print(f"  ❌ Error fetching updates: {e}")
        return []

def download_federated_update(filename: str, local_dir: Path):
    """Download a federated update file"""
    local_file = local_dir / filename
    
    # Skip if already downloaded
    if local_file.exists():
        print(f"  ⏭️  Already downloaded: {filename}")
        return True
    
    try:
        print(f"  📥 Downloading: {filename}...")
        response = requests.get(
            f"{SERVER_URL}/federated/download/{filename}",
            headers=HEADERS,
            timeout=120  # Large files may take time
        )
        response.raise_for_status()
        
        # Save to local file
        with open(local_file, 'wb') as f:
            f.write(response.content)
        
        file_size_mb = local_file.stat().st_size / (1024 * 1024)
        print(f"  ✅ Downloaded: {filename} ({file_size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"  ❌ Error downloading {filename}: {e}")
        return False

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
    
    total_samples = sum(update.get("sample_count", 1) for update in device_updates)
    print(f"📊 Total samples: {total_samples}")
    
    aggregated_weights = {}
    first_update = device_updates[0]
    
    for layer_name in first_update["model_weights"].keys():
        first_weight = first_update["model_weights"][layer_name]
        if isinstance(first_weight, np.ndarray):
            aggregated_weights[layer_name] = np.zeros_like(first_weight, dtype=np.float32)
        else:
            aggregated_weights[layer_name] = np.zeros_like(np.array(first_weight), dtype=np.float32)
    
    for update in device_updates:
        weight_factor = update.get("sample_count", 1) / total_samples
        device_id = update.get("device_id", "unknown")
        sample_count = update.get("sample_count", 0)
        accuracy = update.get("accuracy", 0.0)
        loss = update.get("loss", 0.0)
        print(f"  📱 Device {device_id[:8]}...: {sample_count} samples ({weight_factor:.2%}), Acc: {accuracy:.2f}%, Loss: {loss:.4f}")
        
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
    
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✅ Aggregated model saved: {model_path}")
    print(f"   📦 File size: {file_size_mb:.2f} MB")
    return str(model_path)

def main():
    print("🚀 Download and Aggregate Models from Render")
    print("=" * 60)
    
    # Create local directory for downloads
    local_updates_dir = Path("federated_updates_downloaded")
    local_updates_dir.mkdir(exist_ok=True)
    
    # Get completed jobs
    completed_jobs = get_completed_jobs()
    if not completed_jobs:
        print("❌ No completed jobs found")
        return
    
    print(f"\n📋 Processing {len(completed_jobs)} completed jobs...")
    
    aggregated_count = 0
    
    for job in completed_jobs:
        job_id = job["id"]
        job_name = job["name"]
        
        print(f"\n{'='*60}")
        print(f"📋 Job: {job_name} (ID: {job_id})")
        print(f"{'='*60}")
        
        # Get list of federated update files
        update_files = get_federated_updates_for_job(job_id)
        
        if not update_files:
            print(f"⚠️  No federated updates found for {job_name}")
            continue
        
        print(f"📥 Found {len(update_files)} update file(s)")
        
        # Download all update files
        downloaded_files = []
        for update_info in update_files:
            filename = update_info["filename"]
            if download_federated_update(filename, local_updates_dir):
                downloaded_files.append(filename)
        
        if not downloaded_files:
            print(f"⚠️  No files downloaded for {job_name}")
            continue
        
        # Load and process downloaded files
        device_updates = []
        for filename in downloaded_files:
            file_path = local_updates_dir / filename
            try:
                with open(file_path, 'r') as f:
                    update_data = json.load(f)
                
                model_weights = convert_json_weights_to_numpy(update_data)
                device_updates.append({
                    "device_id": update_data.get("device_id", "unknown"),
                    "model_weights": model_weights,
                    "sample_count": update_data.get("sample_count", 1000),
                    "loss": update_data.get("loss", 0.0),
                    "accuracy": update_data.get("accuracy", 0.0)
                })
                
                # Free memory
                del update_data
                del model_weights
            except Exception as e:
                print(f"  ❌ Error processing {filename}: {e}")
                continue
        
        if not device_updates:
            print(f"⚠️  No valid updates to aggregate for {job_name}")
            continue
        
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
            "aggregation_strategy": "fedavg",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Save aggregated model
        save_aggregated_model(aggregated_weights, job_id, job_name, metadata)
        
        # Free memory
        del aggregated_weights
        del device_updates
        
        aggregated_count += 1
        print(f"\n✅ Completed aggregation for {job_name}")
    
    print(f"\n{'='*60}")
    print(f"✅ Aggregation complete! Processed {aggregated_count} jobs")
    print(f"\n📁 Aggregated models saved to: federated_models/")
    print(f"📁 Downloaded updates saved to: federated_updates_downloaded/")

if __name__ == "__main__":
    main()
