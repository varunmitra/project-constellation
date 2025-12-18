#!/usr/bin/env python3
"""
Aggregate models directly from local federated_updates files
Much faster than doing it on Render!
"""
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
from datetime import datetime

def load_federated_updates():
    """Load all federated update files"""
    updates_dir = Path("federated_updates")
    if not updates_dir.exists():
        print("❌ federated_updates directory not found")
        return []
    
    updates = []
    for update_file in updates_dir.glob("*.json"):
        try:
            with open(update_file, 'r') as f:
                data = json.load(f)
                updates.append((update_file, data))
        except Exception as e:
            print(f"⚠️  Error reading {update_file}: {e}")
    
    print(f"✅ Loaded {len(updates)} federated update files")
    return updates

def group_updates_by_job(updates):
    """Group updates by job_id"""
    jobs = {}
    for file_path, data in updates:
        job_id = data.get("job_id")
        if job_id:
            if job_id not in jobs:
                jobs[job_id] = []
            jobs[job_id].append((file_path, data))
    return jobs

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

def save_aggregated_model(weights: Dict[str, np.ndarray], job_id: str, metadata: Dict):
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
        **metadata
    }, model_path)
    
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✅ Aggregated model saved: {model_path}")
    print(f"   📦 File size: {file_size_mb:.2f} MB")
    return str(model_path)

def main():
    print("🚀 Local Model Aggregation (Direct from Files)")
    print("=" * 60)
    
    # Load all federated updates
    all_updates = load_federated_updates()
    if not all_updates:
        print("❌ No federated updates found")
        return
    
    # Group by job_id
    jobs = group_updates_by_job(all_updates)
    print(f"\n📋 Found {len(jobs)} unique jobs")
    
    # Process each job
    for job_id, updates in jobs.items():
        print(f"\n{'='*60}")
        print(f"📋 Processing job: {job_id}")
        print(f"   Found {len(updates)} device updates")
        print(f"{'='*60}")
        
        # Convert JSON weights to numpy
        device_updates = []
        for file_path, update_data in updates:
            model_weights = convert_json_weights_to_numpy(update_data)
            device_updates.append({
                "device_id": update_data.get("device_id", "unknown"),
                "model_weights": model_weights,
                "sample_count": update_data.get("sample_count", 1000),
                "loss": update_data.get("loss", 0.0),
                "accuracy": update_data.get("accuracy", 0.0)
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
            "aggregation_strategy": "fedavg",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Save aggregated model
        save_aggregated_model(aggregated_weights, job_id, metadata)
        
        print(f"\n✅ Completed aggregation for job {job_id}")
        print(f"   📊 Total samples: {total_samples}")
        print(f"   📊 Average accuracy: {avg_accuracy:.2f}%")
        print(f"   📊 Average loss: {avg_loss:.4f}")
    
    print(f"\n{'='*60}")
    print("✅ All aggregations complete!")
    print(f"\n📁 Aggregated models saved to: federated_models/")

if __name__ == "__main__":
    main()
