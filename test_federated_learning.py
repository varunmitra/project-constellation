#!/usr/bin/env python3
"""
Test script for Federated Learning Integration
Tests the complete flow: training -> upload weights -> aggregate -> verify
"""

import sys
import json
import time
import requests
from pathlib import Path
import torch
import numpy as np

SERVER_URL = "http://localhost:8000"
AUTH_HEADER = {"Authorization": "Bearer constellation-token"}

def print_step(step_num, description):
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print('='*60)

def create_test_job():
    """Create a test training job"""
    print_step(1, "Creating Test Training Job")
    
    job_data = {
        "name": "Federated Learning Test",
        "model_name": "Test Model",
        "model_type": "text_classification",
        "dataset": "synthetic",
        "total_epochs": 2,  # Small for quick testing
        "config": {
            "vocab_size": 1000,
            "seq_length": 50,
            "num_samples": 500,
            "num_classes": 4,
            "batch_size": 32,
            "learning_rate": 0.001
        }
    }
    
    response = requests.post(
        f"{SERVER_URL}/jobs",
        json=job_data,
        headers=AUTH_HEADER
    )
    
    if response.status_code == 200:
        job = response.json()
        print(f"✅ Job created: {job['name']} (ID: {job['id']})")
        return job['id']
    else:
        print(f"❌ Failed to create job: {response.status_code}")
        print(response.text)
        return None

def simulate_device_training(job_id, device_id, device_name):
    """Simulate a device training and uploading weights"""
    print_step(2, f"Simulating Training on {device_name}")
    
    # Register device
    device_data = {
        "name": device_name,
        "device_type": "test-device",
        "os_version": "Test OS",
        "cpu_cores": 4,
        "memory_gb": 8,
        "gpu_available": False,
        "gpu_memory_gb": 0
    }
    
    response = requests.post(
        f"{SERVER_URL}/devices/register",
        json=device_data,
        headers=AUTH_HEADER
    )
    
    if response.status_code != 200:
        print(f"⚠️ Device registration failed, using existing device")
        registered_device_id = device_id
    else:
        registered_device_id = response.json()["id"]
        print(f"✅ Device registered: {registered_device_id}")
    
    # Get job assignment
    response = requests.get(
        f"{SERVER_URL}/devices/{registered_device_id}/next-job",
        headers=AUTH_HEADER
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to get job assignment: {response.status_code}")
        return None
    
    assignment_data = response.json()
    assignment_id = assignment_data.get("assignment_id")
    
    if not assignment_id:
        print(f"❌ No assignment ID received")
        return None
    
    print(f"✅ Got assignment: {assignment_id}")
    
    # Simulate training progress
    total_epochs = 2
    for epoch in range(total_epochs):
        progress = ((epoch + 1) / total_epochs) * 100
        response = requests.post(
            f"{SERVER_URL}/devices/{registered_device_id}/training/{assignment_id}/progress",
            params={"progress": progress, "current_epoch": epoch + 1},
            headers=AUTH_HEADER
        )
        print(f"   Epoch {epoch + 1}/{total_epochs}: {progress:.1f}%")
        time.sleep(0.5)
    
    # Create mock model weights
    model_weights = {
        "embedding.weight": np.random.randn(1000, 128).tolist(),
        "lstm.weight_ih_l0": np.random.randn(256, 128).tolist(),
        "lstm.weight_hh_l0": np.random.randn(256, 64).tolist(),
        "fc.weight": np.random.randn(4, 64).tolist(),
        "fc.bias": np.random.randn(4).tolist()
    }
    
    # Upload model weights
    update_data = {
        "assignment_id": assignment_id,
        "model_weights": model_weights,
        "sample_count": 500 + np.random.randint(-50, 50),  # Vary sample count
        "loss": 0.5 + np.random.random() * 0.3,
        "accuracy": 70.0 + np.random.random() * 20.0,
        "checkpoint_path": f"checkpoints/device_{device_id}_epoch_2.pth"
    }
    
    response = requests.post(
        f"{SERVER_URL}/devices/{registered_device_id}/federated-update",
        json=update_data,
        headers=AUTH_HEADER
    )
    
    if response.status_code == 200:
        print(f"✅ Model weights uploaded successfully")
        return assignment_id
    else:
        print(f"❌ Failed to upload weights: {response.status_code}")
        print(response.text)
        return None

def complete_training(device_id, assignment_id):
    """Mark training as completed"""
    response = requests.post(
        f"{SERVER_URL}/devices/{device_id}/training/{assignment_id}/complete",
        json={"checkpoint_path": f"checkpoints/device_{device_id}.pth"},
        headers=AUTH_HEADER
    )
    
    if response.status_code == 200:
        print(f"✅ Training marked as completed")
        return True
    else:
        print(f"⚠️ Failed to mark complete: {response.status_code}")
        return False

def aggregate_models(job_id):
    """Aggregate models from all devices"""
    print_step(3, "Aggregating Models")
    
    response = requests.post(
        f"{SERVER_URL}/federated/aggregate/{job_id}",
        headers=AUTH_HEADER
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Aggregation successful!")
        print(f"   Model path: {result['model_path']}")
        print(f"   Participating devices: {result['participating_devices']}")
        print(f"   Total samples: {result['total_samples']}")
        return result['model_path']
    else:
        print(f"❌ Aggregation failed: {response.status_code}")
        print(response.text)
        return None

def verify_aggregated_model(job_id, model_path):
    """Verify the aggregated model exists and is valid"""
    print_step(4, "Verifying Aggregated Model")
    
    # Check if file exists
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"✅ Model file exists: {model_path}")
    
    # Load and verify model
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"✅ Model loaded successfully")
        print(f"   Job ID: {checkpoint.get('job_id')}")
        print(f"   Participating devices: {len(checkpoint.get('participating_devices', []))}")
        print(f"   Total samples: {checkpoint.get('total_samples')}")
        print(f"   Average accuracy: {checkpoint.get('avg_accuracy', 0):.2f}%")
        print(f"   Average loss: {checkpoint.get('avg_loss', 0):.4f}")
        
        # Verify model weights exist
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"✅ Model weights found: {len(state_dict)} layers")
            for key in list(state_dict.keys())[:3]:
                print(f"   - {key}: shape {state_dict[key].shape}")
            return True
        else:
            print(f"❌ No model_state_dict in checkpoint")
            return False
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def get_aggregated_model(job_id):
    """Get aggregated model via API"""
    print_step(5, "Getting Aggregated Model via API")
    
    response = requests.get(
        f"{SERVER_URL}/federated/model/{job_id}",
        headers=AUTH_HEADER
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Retrieved aggregated model")
        print(f"   Model path: {result['model_path']}")
        return result['model_path']
    else:
        print(f"❌ Failed to get model: {response.status_code}")
        print(response.text)
        return None

def main():
    print("\n" + "="*60)
    print("🧪 FEDERATED LEARNING TEST")
    print("="*60)
    
    # Check server is running
    try:
        response = requests.get(f"{SERVER_URL}/health")
        if response.status_code != 200:
            print("❌ Server is not running or not healthy")
            return
    except:
        print("❌ Cannot connect to server. Make sure it's running on", SERVER_URL)
        return
    
    print("✅ Server is running")
    
    # Step 1: Create job
    job_id = create_test_job()
    if not job_id:
        return
    
    # Step 2: Simulate multiple devices training
    device_assignments = []
    for i in range(3):  # Simulate 3 devices
        device_id = f"test-device-{i}"
        device_name = f"Test Device {i+1}"
        assignment_id = simulate_device_training(job_id, device_id, device_name)
        if assignment_id:
            device_assignments.append((device_id, assignment_id))
        time.sleep(1)
    
    if not device_assignments:
        print("❌ No devices completed training")
        return
    
    # Complete training for all devices
    print("\n📋 Completing training for all devices...")
    for device_id, assignment_id in device_assignments:
        complete_training(device_id, assignment_id)
    
    # Step 3: Aggregate models
    model_path = aggregate_models(job_id)
    if not model_path:
        return
    
    # Step 4: Verify aggregated model
    if not verify_aggregated_model(job_id, model_path):
        return
    
    # Step 5: Get model via API
    api_model_path = get_aggregated_model(job_id)
    
    print("\n" + "="*60)
    print("✅ FEDERATED LEARNING TEST COMPLETE!")
    print("="*60)
    print("\n📊 Summary:")
    print(f"   - Job ID: {job_id}")
    print(f"   - Devices trained: {len(device_assignments)}")
    print(f"   - Aggregated model: {model_path}")
    print(f"   - Model verified: ✅")
    print("\n🎉 Federated learning is working correctly!")

if __name__ == "__main__":
    main()

