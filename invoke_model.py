#!/usr/bin/env python3
"""
Invoke the trained model for text classification
Loads the aggregated model and makes predictions on new text
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

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
    
    # Get training metrics
    accuracy = checkpoint.get('accuracy', 0)
    loss = checkpoint.get('loss', 0)
    
    print(f"✅ Model loaded successfully")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Loss: {loss:.4f}")
    
    return model, vocab_size, num_classes

def simple_tokenizer(text: str, vocab_size: int = 10000, seq_length: int = 100):
    """
    Simple tokenizer that converts text to token IDs
    In production, you'd use the same tokenizer from training
    """
    # Simple hash-based tokenization
    # In real scenario, use the same tokenizer as training
    words = text.lower().split()
    tokens = []
    for word in words[:seq_length]:
        # Hash word to vocab ID (simple approach)
        token_id = hash(word) % vocab_size
        tokens.append(token_id)
    
    # Pad or truncate to seq_length
    if len(tokens) < seq_length:
        tokens.extend([0] * (seq_length - len(tokens)))
    else:
        tokens = tokens[:seq_length]
    
    return torch.tensor([tokens], dtype=torch.long)

def predict(model, text: str, vocab_size: int, num_classes: int, class_names=None):
    """Make a prediction on input text"""
    # Tokenize text
    input_tensor = simple_tokenizer(text, vocab_size)
    
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

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Invoke trained model for text classification")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
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
                
                result = predict(model, text, vocab_size, num_classes, class_names)
                
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
        
        result = predict(model, args.text, vocab_size, num_classes)
        
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

