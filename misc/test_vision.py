#!/usr/bin/env python3
"""
Test vision/image classification model with descriptive results
Shows class predictions with confidence scores
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import argparse
import requests
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def load_model_architecture(num_classes=10):
    """Load ResNet18 model architecture"""
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def preprocess_image(image_path: str, image_size: int = 224):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        # Assume it's already a PIL Image
        image = image_path.convert('RGB')
    
    # Apply transforms
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension

def create_test_image(color='red', size=224):
    """Create a simple test image for testing"""
    # Create a solid color image
    color_map = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'purple': (128, 0, 128),
        'orange': (255, 165, 0),
        'pink': (255, 192, 203),
        'cyan': (0, 255, 255),
        'brown': (165, 42, 42),
        'gray': (128, 128, 128)
    }
    
    rgb = color_map.get(color.lower(), (128, 128, 128))
    image = Image.new('RGB', (size, size), rgb)
    return image

def predict_image(model, image_tensor, num_classes=10):
    """Predict image class with detailed results"""
    model.eval()
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Get top 3 predictions
    top3_probs, top3_indices = torch.topk(probabilities[0], min(3, num_classes))
    
    predictions = []
    for idx, prob in zip(top3_indices, top3_probs):
        predictions.append({
            'class': int(idx.item()),
            'confidence': float(prob.item())
        })
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'top_predictions': predictions,
        'all_probabilities': probabilities[0].tolist()
    }

def load_trained_model(model_path: str):
    """Load trained vision model"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Try to get config from checkpoint first
    config = checkpoint.get('config', {})
    num_classes = config.get('num_classes')
    image_size = config.get('image_size', 224)
    
    # Check if this is actually a vision model by looking at state_dict keys
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Detect model type from state_dict keys
    is_vision_model = any(key.startswith(('conv1', 'layer1', 'bn1')) for key in state_dict.keys())
    is_text_model = any(key.startswith(('embedding', 'lstm')) for key in state_dict.keys())
    
    if is_text_model and not is_vision_model:
        raise ValueError(f"This checkpoint contains a text classification model (LSTM), not a vision model. "
                        f"Use test_sentiment.py instead.")
    
    # If config not in checkpoint, try to infer from model weights
    if num_classes is None:
        if 'fc.weight' in state_dict:
            # Infer num_classes from the weight shape
            num_classes = state_dict['fc.weight'].shape[0]
            print(f"ℹ️  Inferred num_classes={num_classes} from model weights")
        else:
            num_classes = 10  # Default fallback for vision
            print(f"⚠️  Could not infer num_classes, using default: {num_classes}")
    
    # Create model
    model = load_model_architecture(num_classes)
    
    # Load weights
    cpu_state_dict = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cpu_state_dict[key] = value.cpu()
        else:
            cpu_state_dict[key] = value
    
    # Try to load state dict, handling partial matches
    try:
        model.load_state_dict(cpu_state_dict, strict=True)
    except RuntimeError as e:
        # If strict loading fails, try partial loading
        print(f"⚠️  Strict loading failed: {e}")
        print("   Attempting partial loading...")
        model.load_state_dict(cpu_state_dict, strict=False)
        print("   ⚠️  Partial loading successful - some weights may be missing")
    
    model.eval()
    model.to('cpu')
    
    return model, num_classes, image_size

def main():
    parser = argparse.ArgumentParser(description="Test vision/image classification model")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--color", type=str, help="Create test image with color (red, green, blue, etc.)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Find model
    if not args.model:
        # Try to find vision classification job and its model
        try:
            response = requests.get(
                "http://localhost:8000/jobs",
                headers={"Authorization": "Bearer constellation-token"}
            )
            if response.status_code == 200:
                jobs = response.json()
                vision_jobs = [j for j in jobs if j.get('model_type') == 'vision' and j.get('status') == 'completed']
                if vision_jobs:
                    job = vision_jobs[0]
                    job_id = job['id']
                    config = job.get('config', {})
                    if isinstance(config, str):
                        config = json.loads(config)
                    
                    # Look for model with this job ID
                    federated_models_dir = Path("federated_models")
                    if federated_models_dir.exists():
                        model_path = federated_models_dir / f"aggregated_model_{job_id}.pth"
                        if model_path.exists():
                            args.model = str(model_path)
                            print(f"📁 Using vision model: {model_path.name}")
                            print(f"   Job: {job['name']}")
                            print(f"   Classes: {config.get('num_classes', 'unknown')}")
                        else:
                            print(f"⚠️  Model not found for job {job_id}, using latest...")
                            aggregated_models = list(federated_models_dir.glob("aggregated_model_*.pth"))
                            if aggregated_models:
                                args.model = str(sorted(aggregated_models, key=lambda p: p.stat().st_mtime)[-1])
                                print(f"📁 Using latest model: {Path(args.model).name}")
                            else:
                                print("❌ No model found!")
                                return
                    else:
                        print("❌ federated_models directory not found!")
                        return
                else:
                    # Fallback to latest model
                    federated_models_dir = Path("federated_models")
                    if federated_models_dir.exists():
                        aggregated_models = list(federated_models_dir.glob("aggregated_model_*.pth"))
                        if aggregated_models:
                            args.model = str(sorted(aggregated_models, key=lambda p: p.stat().st_mtime)[-1])
                            print(f"📁 Using latest model: {Path(args.model).name}")
                        else:
                            print("❌ No model found!")
                            return
                    else:
                        print("❌ No model found!")
                        return
        except Exception as e:
            print(f"⚠️  Could not query server: {e}")
            # Fallback to latest model
            federated_models_dir = Path("federated_models")
            if federated_models_dir.exists():
                aggregated_models = list(federated_models_dir.glob("aggregated_model_*.pth"))
                if aggregated_models:
                    args.model = str(sorted(aggregated_models, key=lambda p: p.stat().st_mtime)[-1])
                    print(f"📁 Using latest model: {Path(args.model).name}")
                else:
                    print("❌ No model found!")
                    return
            else:
                print("❌ No model found!")
                return
    
    # Load model
    try:
        model, num_classes, image_size = load_trained_model(args.model)
        print("✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Interactive mode
    if args.interactive:
        print("="*60)
        print("🖼️  Vision Classification - Interactive Mode")
        print("="*60)
        print(f"Model: {num_classes} classes")
        print("Enter image path or color name (or 'quit' to exit)\n")
        
        while True:
            try:
                user_input = input("📝 Enter image path or color: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                # Check if it's a color name
                if user_input.lower() in ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'brown', 'gray']:
                    test_image = create_test_image(user_input.lower(), image_size)
                    image_tensor = preprocess_image(test_image, image_size)
                    print(f"🎨 Created test image: {user_input}")
                else:
                    # Assume it's a file path
                    if not Path(user_input).exists():
                        print(f"❌ File not found: {user_input}")
                        continue
                    image_tensor = preprocess_image(user_input, image_size)
                    print(f"📁 Loaded image: {user_input}")
                
                result = predict_image(model, image_tensor, num_classes)
                
                print(f"\n🎯 Predicted Class: {result['predicted_class']}")
                print(f"📊 Confidence: {result['confidence']*100:.1f}%")
                print(f"\n📈 Top 3 Predictions:")
                for i, pred in enumerate(result['top_predictions'], 1):
                    bar = "█" * int(pred['confidence'] * 50)
                    print(f"   {i}. Class {pred['class']:2d}: {pred['confidence']*100:5.1f}% {bar}")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")
                import traceback
                traceback.print_exc()
    
    # Single prediction mode
    elif args.image or args.color:
        print("="*60)
        print("🖼️  Vision Classification")
        print("="*60)
        
        try:
            if args.color:
                test_image = create_test_image(args.color.lower(), image_size)
                image_tensor = preprocess_image(test_image, image_size)
                print(f"🎨 Created test image: {args.color}")
            else:
                if not Path(args.image).exists():
                    print(f"❌ File not found: {args.image}")
                    return
                image_tensor = preprocess_image(args.image, image_size)
                print(f"📁 Loaded image: {args.image}")
            
            result = predict_image(model, image_tensor, num_classes)
            
            print(f"\n🎯 Predicted Class: {result['predicted_class']}")
            print(f"📊 Confidence: {result['confidence']*100:.1f}%")
            print(f"\n📈 Top 3 Predictions:")
            for i, pred in enumerate(result['top_predictions'], 1):
                bar = "█" * int(pred['confidence'] * 50)
                print(f"   {i}. Class {pred['class']:2d}: {pred['confidence']*100:5.1f}% {bar}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("="*60)
        print("🖼️  Vision Classification Test")
        print("="*60)
        print("\nUsage:")
        print("  python3 test_vision.py --image path/to/image.jpg")
        print("  python3 test_vision.py --color red")
        print("  python3 test_vision.py --interactive")
        print("\nExample:")
        print("  python3 test_vision.py --color red")
        print("  python3 test_vision.py --color blue")
        print("  python3 test_vision.py --image my_image.jpg")

if __name__ == "__main__":
    main()

