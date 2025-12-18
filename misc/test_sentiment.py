#!/usr/bin/env python3
"""
Test sentiment analysis model with descriptive results
Shows sentiment predictions with confidence scores
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def load_model_architecture(num_classes=3):
    """Load sentiment analysis model architecture"""
    class SentimentLSTM(nn.Module):
        def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=64, num_classes=3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, num_classes)
        
        def forward(self, x):
            embedded = self.embedding(x)
            lstm_out, _ = self.lstm(embedded)
            last_output = lstm_out[:, -1, :]
            return self.fc(last_output)
    
    return SentimentLSTM

def simple_tokenizer(text: str, vocab_size: int = 10000, seq_length: int = 100):
    """Simple tokenizer for text"""
    words = text.lower().split()
    tokens = [hash(word) % vocab_size for word in words[:seq_length]]
    tokens.extend([0] * (seq_length - len(tokens)))
    return torch.tensor([tokens], dtype=torch.long)

def predict_sentiment(model, text: str, vocab_size: int = 10000):
    """Predict sentiment with detailed results"""
    model.eval()
    input_tensor = simple_tokenizer(text, vocab_size)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Determine number of classes from model output
    num_classes = probabilities.shape[1]
    
    if num_classes == 3:
        sentiments = ['Negative', 'Neutral', 'Positive']
    elif num_classes == 2:
        sentiments = ['Negative', 'Positive']
    elif num_classes == 4:
        sentiments = ['World', 'Sports', 'Business', 'Sci/Tech']  # AG News
    else:
        sentiments = [f'Class {i}' for i in range(num_classes)]
    
    sentiment_scores = {
        sentiments[i]: probabilities[0][i].item() 
        for i in range(num_classes)
    }
    
    result = {
        'sentiment': sentiments[predicted_class],
        'confidence': confidence,
        'scores': sentiment_scores
    }
    
    # Add raw_scores only if we have the expected sentiment labels
    if num_classes == 3 and all(s in sentiment_scores for s in ['Negative', 'Neutral', 'Positive']):
        result['raw_scores'] = {
            'negative': sentiment_scores['Negative'],
            'neutral': sentiment_scores['Neutral'],
            'positive': sentiment_scores['Positive']
        }
    elif num_classes == 2 and all(s in sentiment_scores for s in ['Negative', 'Positive']):
        result['raw_scores'] = {
            'negative': sentiment_scores.get('Negative', 0),
            'positive': sentiment_scores.get('Positive', 0)
        }
    
    return result

def load_trained_model(model_path: str):
    """Load trained sentiment model"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Try to get config from checkpoint first
    config = checkpoint.get('config', {})
    num_classes = config.get('num_classes')
    vocab_size = config.get('vocab_size', 10000)
    
    # If config not in checkpoint, try to infer from model weights
    if num_classes is None:
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if 'fc.weight' in state_dict:
            # Infer num_classes from the weight shape
            num_classes = state_dict['fc.weight'].shape[0]
            print(f"ℹ️  Inferred num_classes={num_classes} from model weights")
        else:
            num_classes = 3  # Default fallback
            print(f"⚠️  Could not infer num_classes, using default: {num_classes}")
    
    # Create model
    ModelClass = load_model_architecture(num_classes)
    model = ModelClass(vocab_size=vocab_size, embed_dim=128, hidden_dim=64, num_classes=num_classes)
    
    # Load weights
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    cpu_state_dict = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cpu_state_dict[key] = value.cpu()
        else:
            cpu_state_dict[key] = value
    
    model.load_state_dict(cpu_state_dict)
    model.eval()
    model.to('cpu')
    
    return model, vocab_size

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test sentiment analysis model")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Find model
    if not args.model:
        from pathlib import Path
        import requests
        
        # Try to find sentiment analysis job and its model
        try:
            response = requests.get(
                "http://localhost:8000/jobs",
                headers={"Authorization": "Bearer constellation-token"}
            )
            if response.status_code == 200:
                jobs = response.json()
                sentiment_jobs = [j for j in jobs if 'sentiment' in j.get('name', '').lower() and j.get('status') == 'completed']
                if sentiment_jobs:
                    job = sentiment_jobs[0]
                    job_id = job['id']
                    config = job.get('config', {})
                    if isinstance(config, str):
                        import json
                        config = json.loads(config)
                    
                    # Look for model with this job ID
                    federated_models_dir = Path("federated_models")
                    if federated_models_dir.exists():
                        model_path = federated_models_dir / f"aggregated_model_{job_id}.pth"
                        if model_path.exists():
                            args.model = str(model_path)
                            print(f"📁 Using sentiment model: {model_path.name}")
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
        model, vocab_size = load_trained_model(args.model)
        print("✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Interactive mode
    if args.interactive:
        print("="*60)
        print("😊 Sentiment Analysis - Interactive Mode")
        print("="*60)
        print("Enter text to analyze (or 'quit' to exit)\n")
        
        while True:
            try:
                text = input("📝 Enter text: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not text:
                    continue
                
                result = predict_sentiment(model, text, vocab_size)
                
                print(f"\n🎯 Sentiment: {result['sentiment']}")
                print(f"📊 Confidence: {result['confidence']*100:.1f}%")
                print(f"\n📈 Detailed Scores:")
                for sentiment, score in result['scores'].items():
                    bar = "█" * int(score * 50)
                    emoji = "😊" if sentiment == "Positive" else "😐" if sentiment == "Neutral" else "😞"
                    print(f"   {emoji} {sentiment:8s}: {score*100:5.1f}% {bar}")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")
    
    # Single prediction mode
    elif args.text:
        print("="*60)
        print("😊 Sentiment Analysis")
        print("="*60)
        
        result = predict_sentiment(model, args.text, vocab_size)
        
        print(f"\n📝 Input Text: {args.text}")
        print(f"\n🎯 Sentiment: {result['sentiment']}")
        print(f"📊 Confidence: {result['confidence']*100:.1f}%")
        print(f"\n📈 Detailed Scores:")
        for sentiment, score in result['scores'].items():
            bar = "█" * int(score * 50)
            emoji = "😊" if sentiment == "Positive" else "😐" if sentiment == "Neutral" else "😞"
            print(f"   {emoji} {sentiment:8s}: {score*100:5.1f}% {bar}")
    
    else:
        print("="*60)
        print("😊 Sentiment Analysis Test")
        print("="*60)
        print("\nUsage:")
        print("  python3 test_sentiment.py --text \"Your text here\"")
        print("  python3 test_sentiment.py --interactive")
        print("\nExample:")
        print("  python3 test_sentiment.py --text \"I love this product! It's amazing!\"")
        print("  python3 test_sentiment.py --text \"This is terrible, worst experience ever\"")
        print("  python3 test_sentiment.py --text \"It's okay, nothing special\"")

if __name__ == "__main__":
    main()

