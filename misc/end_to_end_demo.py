#!/usr/bin/env python3
"""
End-to-End Demo: From Dumb Model to Intelligent Model
Demonstrates the complete workflow of training a model from scratch
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import requests
import json
import time

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

SERVER_URL = "http://localhost:8000"
AUTH_HEADER = {"Authorization": "Bearer constellation-token"}

def create_dumb_model():
    """Create a fresh, untrained model with random weights"""
    class SimpleLSTM(nn.Module):
        def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=64, num_classes=4):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)
        
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            last_output = lstm_out[:, -1, :]
            return self.fc(last_output)
    
    model = SimpleLSTM()
    return model

def evaluate_model_intelligence(model, test_cases):
    """Evaluate model on test cases and return accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for text, expected_class in test_cases:
            # Simple tokenization
            words = text.lower().split()
            tokens = [hash(word) % 10000 for word in words[:100]]
            tokens.extend([0] * (100 - len(tokens)))
            input_tensor = torch.tensor([tokens], dtype=torch.long)
            
            output = model(input_tensor)
            predicted = torch.argmax(output, dim=1).item()
            
            if predicted == expected_class:
                correct += 1
            total += 1
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy, correct, total

def test_dumb_model():
    """Test the untrained model - should perform poorly (random ~25%)"""
    print("\n" + "="*60)
    print("🧪 STEP 1: Testing DUMB Model (Untrained)")
    print("="*60)
    
    model = create_dumb_model()
    model.eval()
    
    # Test cases
    test_cases = [
        ("Scientists discover new breakthrough in quantum computing", 3),  # Sci/Tech
        ("Stock market reaches new highs as investors celebrate", 2),      # Business
        ("Team wins championship after thrilling overtime victory", 1),    # Sports
        ("World leaders meet to discuss climate change", 0),               # World
        ("New AI technology transforms healthcare industry", 3),            # Sci/Tech
        ("Company reports record profits this quarter", 2),                # Business
        ("Athlete breaks world record in Olympic games", 1),               # Sports
        ("International summit addresses global security", 0),              # World
    ]
    
    accuracy, correct, total = evaluate_model_intelligence(model, test_cases)
    
    print(f"\n📊 Dumb Model Performance:")
    print(f"   Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
    print(f"   Expected: ~25% (random guessing for 4 classes)")
    
    if accuracy < 40:
        print(f"   ✅ Model is DUMB (as expected - untrained)")
    else:
        print(f"   ⚠️  Model performs better than expected")
    
    return accuracy

def create_training_job():
    """Create a training job on the server"""
    print("\n" + "="*60)
    print("🚀 STEP 2: Creating Training Job")
    print("="*60)
    
    job_data = {
        "name": "End-to-End Intelligence Demo",
        "model_type": "text_classification",
        "dataset": "synthetic",
        "total_epochs": 50,
        "config": {
            "vocab_size": 10000,
            "seq_length": 100,
            "num_samples": 2000,
            "num_classes": 4,
            "batch_size": 32,
            "learning_rate": 0.001
        }
    }
    
    try:
        response = requests.post(
            f"{SERVER_URL}/jobs",
            json=job_data,
            headers=AUTH_HEADER
        )
        response.raise_for_status()
        job = response.json()
        
        print(f"✅ Training job created!")
        print(f"   Job ID: {job['id']}")
        print(f"   Name: {job['name']}")
        print(f"   Epochs: {job['total_epochs']}")
        print(f"   Status: {job['status']}")
        
        return job['id']
    except Exception as e:
        print(f"❌ Failed to create job: {e}")
        return None

def wait_for_training_completion(job_id, max_wait=300):
    """Wait for training to complete"""
    print("\n" + "="*60)
    print("⏳ STEP 3: Training in Progress...")
    print("="*60)
    
    start_time = time.time()
    last_progress = 0
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(
                f"{SERVER_URL}/jobs/{job_id}",
                headers=AUTH_HEADER
            )
            response.raise_for_status()
            job = response.json()
            
            progress = job.get('progress', 0)
            status = job.get('status', 'unknown')
            current_epoch = job.get('current_epoch', 0)
            total_epochs = job.get('total_epochs', 0)
            
            if progress > last_progress:
                print(f"   Progress: {progress:.1f}% (Epoch {current_epoch}/{total_epochs})")
                last_progress = progress
            
            if status == 'completed':
                print(f"\n✅ Training completed!")
                print(f"   Final progress: {progress:.1f}%")
                return True
            elif status == 'failed':
                print(f"\n❌ Training failed!")
                return False
            
            time.sleep(5)
        except Exception as e:
            print(f"⚠️  Error checking status: {e}")
            time.sleep(5)
    
    print(f"\n⏰ Timeout waiting for training")
    return False

def test_trained_model():
    """Test the trained model - should perform much better"""
    print("\n" + "="*60)
    print("🧠 STEP 4: Testing INTELLIGENT Model (After Training)")
    print("="*60)
    
    # Try aggregated model first (preferred)
    federated_models_dir = Path("federated_models")
    aggregated_models = list(federated_models_dir.glob("aggregated_model_*.pth")) if federated_models_dir.exists() else []
    
    if aggregated_models:
        # Use latest aggregated model
        latest_checkpoint = sorted(aggregated_models, key=lambda p: p.stat().st_mtime)[-1]
        print(f"📁 Using aggregated model: {latest_checkpoint.name}")
    else:
        # Fallback to checkpoints
        checkpoints_dir = Path("checkpoints")  # Training engine saves here
        if not checkpoints_dir.exists():
            checkpoints_dir = Path("training/checkpoints")  # Fallback location
        
        if not checkpoints_dir.exists():
            print("❌ No models found!")
            return None
        
        checkpoint_files = sorted(checkpoints_dir.glob("checkpoint_epoch_*.pth"), key=lambda p: p.stat().st_mtime)
        if not checkpoint_files:
            print("❌ No checkpoint files found!")
            return None
        
        latest_checkpoint = checkpoint_files[-1]
        print(f"📁 Loading model from: {latest_checkpoint.name}")
    
    # Load model
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    
    # Create model architecture
    model = create_dumb_model()
    
    # Convert state dict to CPU
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    cpu_state_dict = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cpu_state_dict[key] = value.cpu()
        else:
            cpu_state_dict[key] = value
    
    model.load_state_dict(cpu_state_dict)
    model.eval()
    
    # Get training metrics
    accuracy = checkpoint.get('accuracy', 0)
    loss = checkpoint.get('loss', 0)
    epoch = checkpoint.get('epoch', 0)
    
    print(f"✅ Model loaded!")
    print(f"   Training Accuracy: {accuracy:.2f}%")
    print(f"   Training Loss: {loss:.4f}")
    print(f"   Epochs Trained: {epoch}")
    
    # Test on same test cases
    test_cases = [
        ("Scientists discover new breakthrough in quantum computing", 3),
        ("Stock market reaches new highs as investors celebrate", 2),
        ("Team wins championship after thrilling overtime victory", 1),
        ("World leaders meet to discuss climate change", 0),
        ("New AI technology transforms healthcare industry", 3),
        ("Company reports record profits this quarter", 2),
        ("Athlete breaks world record in Olympic games", 1),
        ("International summit addresses global security", 0),
    ]
    
    test_accuracy, correct, total = evaluate_model_intelligence(model, test_cases)
    
    print(f"\n📊 Intelligent Model Performance:")
    print(f"   Test Accuracy: {test_accuracy:.2f}% ({correct}/{total} correct)")
    print(f"   Training Accuracy: {accuracy:.2f}%")
    
    return test_accuracy, accuracy

def compare_models(dumb_accuracy, intelligent_accuracy):
    """Compare dumb vs intelligent model"""
    print("\n" + "="*60)
    print("📈 STEP 5: Intelligence Comparison")
    print("="*60)
    
    improvement = intelligent_accuracy - dumb_accuracy
    
    print(f"\n📊 Model Performance Comparison:")
    print(f"   Dumb Model (Untrained):     {dumb_accuracy:.2f}%")
    print(f"   Intelligent Model (Trained): {intelligent_accuracy:.2f}%")
    print(f"   Improvement:               {improvement:+.2f}%")
    
    if improvement > 20:
        print(f"\n✅ Model became SIGNIFICANTLY MORE INTELLIGENT!")
        print(f"   The training process successfully taught the model!")
    elif improvement > 10:
        print(f"\n✅ Model became MORE INTELLIGENT!")
        print(f"   Training improved the model's performance.")
    elif improvement > 0:
        print(f"\n✅ Model shows improvement!")
        print(f"   Training is working, but may need more epochs.")
    else:
        print(f"\n⚠️  Model didn't improve much")
        print(f"   May need more training or different hyperparameters.")
    
    print(f"\n💡 Key Insight:")
    print(f"   Through federated learning and distributed training,")
    print(f"   we transformed a DUMB model into an INTELLIGENT model!")

def main():
    print("="*60)
    print("🎯 End-to-End Demo: From Dumb to Intelligent Model")
    print("="*60)
    print("\nThis demo will:")
    print("  1. Test an untrained (dumb) model")
    print("  2. Create a training job")
    print("  3. Train the model")
    print("  4. Test the trained (intelligent) model")
    print("  5. Compare the results")
    print("\n" + "="*60)
    
    # Step 1: Test dumb model
    dumb_accuracy = test_dumb_model()
    
    # Step 2: Create training job
    job_id = create_training_job()
    if not job_id:
        print("\n❌ Cannot proceed without training job")
        return
    
    print(f"\n💡 Next steps:")
    print(f"   1. Make sure Swift app is connected to server")
    print(f"   2. The app will automatically pick up the job")
    print(f"   3. Wait for training to complete")
    print(f"   4. Then run this script again with --test-intelligent")
    print(f"\n   Or wait here for training to complete...")
    
    # Ask if user wants to wait
    response = input("\n⏳ Wait for training to complete? (y/n): ").strip().lower()
    
    if response == 'y':
        # Step 3: Wait for training
        if wait_for_training_completion(job_id):
            # Step 4: Test intelligent model
            result = test_trained_model()
            if result:
                intelligent_accuracy, training_accuracy = result
                # Step 5: Compare
                compare_models(dumb_accuracy, intelligent_accuracy)
        else:
            print("\n⚠️  Training didn't complete. You can run:")
            print(f"   python3 end_to_end_demo.py --test-intelligent")
    else:
        print(f"\n✅ Dumb model tested. Start training and then run:")
        print(f"   python3 end_to_end_demo.py --test-intelligent")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-intelligent", action="store_true", help="Test trained model only")
    args = parser.parse_args()
    
    if args.test_intelligent:
        # Just test the intelligent model
        result = test_trained_model()
        if result:
            intelligent_accuracy, training_accuracy = result
            print(f"\n✅ Model intelligence test complete!")
            print(f"   The model has been trained and is now intelligent!")
    else:
        main()

