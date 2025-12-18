#!/usr/bin/env python3
"""
Invoke the trained model for text classification
Loads the aggregated model and makes predictions on new text
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import hashlib
import requests
import json
import numpy as np
from datetime import datetime
from typing import Optional

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

SERVER_URL = "https://project-constellation.onrender.com"
AUTH_TOKEN = "constellation-token"
HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}"}

def load_model_architecture():
    """Load the same model architecture used during training"""
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
    
    return SimpleLSTM

def load_trained_model(model_path: str, device='cpu'):
    """Load a trained model from checkpoint"""
    print(f"📁 Loading model from: {model_path}")
    
    # Load checkpoint - force CPU and handle MPS tensor conversion
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except TypeError as e:
        if 'MPS' in str(e) or 'float64' in str(e):
            # Try loading with weights_only=False and map to CPU
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        else:
            raise
    
    # Get model configuration
    config = checkpoint.get('config', {})
    vocab_size = config.get('vocab_size', 10000)
    num_classes = config.get('num_classes', 4)
    
    # Create model architecture
    ModelClass = load_model_architecture()
    model = ModelClass(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=64,
        num_classes=num_classes
    )
    
    # Load weights - convert all tensors to CPU first
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Convert MPS tensors to CPU tensors
    cpu_state_dict = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cpu_state_dict[key] = value.cpu()
        else:
            cpu_state_dict[key] = value
    
    model.load_state_dict(cpu_state_dict)
    model.eval()
    model.to('cpu')  # Force CPU for inference
    
    # Get training metrics (check both naming conventions)
    # Aggregated models use avg_accuracy/avg_loss, individual checkpoints use accuracy/loss
    accuracy = checkpoint.get('accuracy') or checkpoint.get('avg_accuracy', 0)
    loss = checkpoint.get('loss') or checkpoint.get('avg_loss', 0)
    
    print(f"✅ Model loaded successfully")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Loss: {loss:.4f}")
    
    # Show additional metadata if available
    if 'participating_devices' in checkpoint:
        devices = checkpoint.get('participating_devices', [])
        total_samples = checkpoint.get('total_samples', 0)
        print(f"   Devices: {len(devices)}, Total samples: {total_samples}")
    
    # Warn if model metrics look suspicious
    if accuracy == 0.0 and loss == 0.0:
        print(f"⚠️  Warning: Model shows 0% accuracy and 0.0 loss.")
        print(f"   This may indicate:")
        print(f"   1. Model was trained on synthetic random data (not real text)")
        print(f"   2. Model metrics weren't saved properly")
        print(f"   3. Tokenization mismatch between training and inference")
    
    return model, vocab_size, num_classes

def deterministic_hash(word: str, vocab_size: int) -> int:
    """
    Deterministic hash function that matches training tokenization
    Uses MD5 hash to ensure consistency across Python runs
    
    NOTE: If the model was trained with Python's hash() function (non-deterministic),
    this may not match. The model may need to be retrained with deterministic hashing.
    """
    # Use MD5 hash for deterministic tokenization
    # This ensures the same word always gets the same token ID
    hash_obj = hashlib.md5(word.encode('utf-8'))
    hash_int = int(hash_obj.hexdigest(), 16)
    return hash_int % vocab_size

def simple_tokenizer(text: str, vocab_size: int = 10000, seq_length: int = 100):
    """
    Simple tokenizer that converts text to token IDs
    Uses deterministic hashing to match training tokenization
    """
    words = text.lower().split()
    tokens = []
    for word in words[:seq_length]:
        # Use deterministic hash to match training
        token_id = deterministic_hash(word, vocab_size)
        tokens.append(token_id)
    
    # Pad or truncate to seq_length
    if len(tokens) < seq_length:
        tokens.extend([0] * (seq_length - len(tokens)))
    else:
        tokens = tokens[:seq_length]
    
    return torch.tensor([tokens], dtype=torch.long)

def predict(model, text: str, vocab_size: int, num_classes: int, class_names=None, debug=False):
    """Make a prediction on input text"""
    # Tokenize text
    input_tensor = simple_tokenizer(text, vocab_size)
    
    if debug:
        # Show tokenization for debugging
        words = text.lower().split()[:10]  # First 10 words
        print(f"🔍 Debug: First 10 words tokenized:")
        for word in words:
            token_id = deterministic_hash(word, vocab_size)
            print(f"   '{word}' -> token_id: {token_id}")
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Get class name
    if class_names is None:
        class_names = ['World', 'Sports', 'Business', 'Sci/Tech']  # AG News categories
    
    return {
        'predicted_class': predicted_class,
        'class_name': class_names[predicted_class],
        'confidence': confidence,
        'probabilities': {
            class_names[i]: probabilities[0][i].item() 
            for i in range(num_classes)
        }
    }

def get_latest_trained_job_from_server():
    """Get the most recently completed job from the server"""
    try:
        response = requests.get(f"{SERVER_URL}/jobs", headers=HEADERS, timeout=30)
        response.raise_for_status()
        jobs = response.json()
        
        # Filter completed jobs
        completed_jobs = [job for job in jobs if job.get("status") == "completed"]
        
        if not completed_jobs:
            return None
        
        # Sort by completed_at timestamp (most recent first)
        def get_completed_time(job):
            completed_at = job.get("completed_at")
            if completed_at:
                try:
                    # Parse ISO format timestamp
                    return datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
                except:
                    return datetime.min
            return datetime.min
        
        latest_job = max(completed_jobs, key=get_completed_time)
        return latest_job
    except Exception as e:
        print(f"⚠️  Error fetching jobs from server: {e}")
        return None

def download_and_aggregate_latest_job(job_id: str, job_name: str) -> Optional[str]:
    """Download federated updates for a job and aggregate them locally"""
    print(f"📡 Fetching federated updates for job: {job_name} ({job_id[:8]}...)")
    
    # Get federated updates for this job
    try:
        response = requests.get(
            f"{SERVER_URL}/federated/updates/{job_id}",
            headers=HEADERS,
            timeout=30
        )
        if response.status_code == 404:
            print(f"  ⚠️  No federated updates found for this job")
            return None
        response.raise_for_status()
        data = response.json()
        update_files = data.get("update_files", [])
    except Exception as e:
        print(f"  ❌ Error fetching updates: {e}")
        return None
    
    if not update_files:
        print(f"  ⚠️  No update files found")
        return None
    
    print(f"  📥 Found {len(update_files)} update file(s)")
    
    # Create local directory for downloads
    local_updates_dir = Path("federated_updates_temp")
    local_updates_dir.mkdir(exist_ok=True)
    
    # Download all update files
    device_updates = []
    for update_info in update_files:
        filename = update_info["filename"]
        local_file = local_updates_dir / filename
        
        if not local_file.exists():
            try:
                print(f"  📥 Downloading: {filename}...")
                response = requests.get(
                    f"{SERVER_URL}/federated/download/{filename}",
                    headers=HEADERS,
                    timeout=120
                )
                response.raise_for_status()
                with open(local_file, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"  ❌ Error downloading {filename}: {e}")
                continue
        
        # Load and convert to numpy
        try:
            with open(local_file, 'r') as f:
                update_data = json.load(f)
            
            # Convert JSON weights to numpy arrays
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
            
            device_updates.append({
                "device_id": update_data.get("device_id", "unknown"),
                "model_weights": model_weights,
                "sample_count": update_data.get("sample_count", 1000),
                "loss": update_data.get("loss", 0.0),
                "accuracy": update_data.get("accuracy", 0.0)
            })
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {e}")
            continue
    
    if not device_updates:
        print(f"  ⚠️  No valid updates to aggregate")
        return None
    
    # Aggregate using Federated Averaging
    print(f"  🔄 Aggregating {len(device_updates)} model updates...")
    total_samples = sum(update.get("sample_count", 1) for update in device_updates)
    
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
        for layer_name, weights in update["model_weights"].items():
            if layer_name in aggregated_weights:
                if isinstance(weights, np.ndarray):
                    aggregated_weights[layer_name] += weight_factor * weights.astype(np.float32)
                else:
                    aggregated_weights[layer_name] += weight_factor * np.array(weights, dtype=np.float32)
    
    # Save aggregated model
    models_dir = Path("federated_models")
    models_dir.mkdir(exist_ok=True)
    
    aggregated_state_dict = {}
    for key, value in aggregated_weights.items():
        if isinstance(value, np.ndarray):
            aggregated_state_dict[key] = torch.from_numpy(value)
        else:
            aggregated_state_dict[key] = torch.tensor(value)
    
    model_path = models_dir / f"aggregated_model_{job_id}.pth"
    
    total_samples = sum(u["sample_count"] for u in device_updates)
    avg_accuracy = np.mean([u["accuracy"] for u in device_updates])
    avg_loss = np.mean([u["loss"] for u in device_updates])
    
    torch.save({
        "model_state_dict": aggregated_state_dict,
        "job_id": job_id,
        "job_name": job_name,
        "participating_devices": [u["device_id"] for u in device_updates],
        "total_samples": total_samples,
        "avg_accuracy": float(avg_accuracy),
        "avg_loss": float(avg_loss),
        "aggregation_strategy": "fedavg",
        "timestamp": datetime.utcnow().isoformat()
    }, model_path)
    
    print(f"  ✅ Aggregated model saved: {model_path.name}")
    print(f"     Accuracy: {avg_accuracy:.2f}%, Loss: {avg_loss:.4f}, Samples: {total_samples}")
    
    # Cleanup temp files
    import shutil
    if local_updates_dir.exists():
        shutil.rmtree(local_updates_dir)
    
    return str(model_path)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Invoke trained model for text classification")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--debug", action="store_true", help="Show debug information (tokenization)")
    parser.add_argument("--from-server", action="store_true", help="Fetch the latest trained model from server")
    
    args = parser.parse_args()
    
    # Fetch from server if requested
    if args.from_server:
        print("📡 Fetching latest trained model from server...")
        latest_job = get_latest_trained_job_from_server()
        
        if not latest_job:
            print("❌ No completed jobs found on server")
            print("   Use --model to specify a local model, or train a job first")
            return
        
        job_id = latest_job.get("id")
        job_name = latest_job.get("name", "unnamed")
        completed_at = latest_job.get("completed_at", "unknown")
        
        print(f"✅ Found latest job: {job_name}")
        print(f"   Completed: {completed_at}")
        
        # Check if we already have this model locally
        models_dir = Path("federated_models")
        existing_model = models_dir / f"aggregated_model_{job_id}.pth"
        
        if existing_model.exists():
            print(f"📁 Using existing local model: {existing_model.name}")
            args.model = str(existing_model)
        else:
            # Download and aggregate
            model_path = download_and_aggregate_latest_job(job_id, job_name)
            if model_path:
                args.model = model_path
            else:
                print("❌ Failed to download and aggregate model from server")
                return
    
    # Find model if not provided
    if not args.model:
        # Try aggregated models first (preferred)
        federated_models_dir = Path("federated_models")
        if federated_models_dir.exists():
            aggregated_models = list(federated_models_dir.glob("aggregated_model_*.pth"))
            if aggregated_models:
                # Use latest aggregated model
                latest_aggregated = sorted(aggregated_models, key=lambda p: p.stat().st_mtime)[-1]
                args.model = str(latest_aggregated)
                print(f"📁 Using latest aggregated model: {latest_aggregated.name}")
            else:
                # Try checkpoints directory
                checkpoints_dir = Path("checkpoints")
                if not checkpoints_dir.exists():
                    checkpoints_dir = Path("training/checkpoints")
                
                if checkpoints_dir.exists():
                    checkpoint_files = sorted(checkpoints_dir.glob("checkpoint_epoch_*.pth"), key=lambda p: p.stat().st_mtime)
                    if checkpoint_files:
                        args.model = str(checkpoint_files[-1])
                        print(f"📁 Using latest checkpoint: {checkpoint_files[-1].name}")
                    else:
                        print("❌ No model found!")
                        print("   Specify model path with --model")
                        print("   Or ensure federated_models/ or checkpoints/ directory has model files")
                        return
                else:
                    print("❌ No model found!")
                    print("   Specify model path with --model")
                    return
        else:
            # Try checkpoints directory
            checkpoints_dir = Path("checkpoints")
            if not checkpoints_dir.exists():
                checkpoints_dir = Path("training/checkpoints")
            
            if checkpoints_dir.exists():
                checkpoint_files = sorted(checkpoints_dir.glob("checkpoint_epoch_*.pth"), key=lambda p: p.stat().st_mtime)
                if checkpoint_files:
                    args.model = str(checkpoint_files[-1])
                    print(f"📁 Using latest checkpoint: {checkpoint_files[-1].name}")
                else:
                    print("❌ No model found!")
                    print("   Specify model path with --model")
                    return
            else:
                print("❌ No model found!")
                print("   Specify model path with --model")
                return
    
    # Load model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        model, vocab_size, num_classes = load_trained_model(args.model, device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Interactive mode
    if args.interactive:
        print("\n" + "="*60)
        print("🤖 Interactive Text Classification")
        print("="*60)
        print("Enter text to classify (or 'quit' to exit)")
        print()
        
        class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        
        while True:
            try:
                text = input("\n📝 Enter text: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not text:
                    continue
                
                result = predict(model, text, vocab_size, num_classes, class_names, debug=args.debug)
                
                print(f"\n🎯 Prediction: {result['class_name']}")
                print(f"📊 Confidence: {result['confidence']*100:.2f}%")
                print(f"\n📈 All Probabilities:")
                for class_name, prob in result['probabilities'].items():
                    bar = "█" * int(prob * 50)
                    print(f"   {class_name:12s}: {prob*100:5.2f}% {bar}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    # Single prediction mode
    elif args.text:
        print("\n" + "="*60)
        print("🤖 Text Classification")
        print("="*60)
        
        result = predict(model, args.text, vocab_size, num_classes, debug=args.debug)
        
        print(f"\n📝 Input Text: {args.text}")
        print(f"\n🎯 Prediction: {result['class_name']}")
        print(f"📊 Confidence: {result['confidence']*100:.2f}%")
        print(f"\n📈 All Probabilities:")
        for class_name, prob in result['probabilities'].items():
            bar = "█" * int(prob * 50)
            print(f"   {class_name:12s}: {prob*100:5.2f}% {bar}")
    
    else:
        print("\n" + "="*60)
        print("📖 Usage Examples")
        print("="*60)
        print("\n1. Single prediction:")
        print(f"   python3 invoke_model.py --text \"Breaking news about technology\"")
        print("\n2. Interactive mode:")
        print(f"   python3 invoke_model.py --interactive")
        print("\n3. Specify model:")
        print(f"   python3 invoke_model.py --model path/to/model.pth --text \"Your text here\"")

if __name__ == "__main__":
    main()

