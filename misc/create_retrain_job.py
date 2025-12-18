#!/usr/bin/env python3
"""
Create a training job for retraining with fixed deterministic tokenization
"""

import requests
import json

SERVER_URL = "https://project-constellation.onrender.com"
AUTH_HEADER = {"Authorization": "Bearer constellation-token"}

def create_retrain_job():
    """Create a training job for retraining with fixed tokenization"""
    
    job_data = {
        "name": "Retrained Model - Fixed Deterministic Tokenization",
        "model_type": "text_classification",
        "dataset": "synthetic",  # Use 'synthetic' for now (engine.py uses random data)
        "total_epochs": 20,  # More epochs for better training
        "config": {
            "vocab_size": 10000,
            "seq_length": 100,
            "num_samples": 2000,  # More samples = better training
            "num_classes": 4,  # AG News: World, Sports, Business, Sci/Tech
            "batch_size": 32,
            "learning_rate": 0.001
        }
    }
    
    try:
        response = requests.post(
            f"{SERVER_URL}/jobs",
            json=job_data,
            headers=AUTH_HEADER,
            timeout=30
        )
        response.raise_for_status()
        job = response.json()
        
        print("✅ Retraining Job Created!")
        print(f"   Job ID: {job['id']}")
        print(f"   Name: {job['name']}")
        print(f"   Epochs: {job['total_epochs']}")
        print(f"   Dataset: {job['dataset']}")
        print(f"   Status: {job['status']}")
        print()
        print("📝 Note:")
        print("   - The training code already has deterministic hashing fixes")
        print("   - Desktop app will automatically pick up this job")
        print("   - Training will start within 10 seconds")
        print()
        print("🚀 Next Steps:")
        print("   1. Make sure desktop app is running:")
        print("      cd desktop-swift && ./build/Constellation")
        print("   2. Watch terminal for training logs")
        print("   3. After training completes:")
        print("      python3 download_and_aggregate.py")
        print("   4. Test the new model:")
        print("      python3 invoke_model.py --text \"Breaking news about technology\"")
        
        return job['id']
    except Exception as e:
        print(f"❌ Failed to create job: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    create_retrain_job()

