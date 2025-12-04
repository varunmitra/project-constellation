#!/usr/bin/env python3
"""
Model Evaluation Script for AG News Classification
Evaluates the current model performance and training progress
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
from pathlib import Path

# Add training directory to path
sys.path.append('training')
from ag_news_trainer import AGNewsTrainer, AGNewsDataset

def load_latest_checkpoint():
    """Load the latest model checkpoint"""
    checkpoint_dir = Path('training/checkpoints')
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        return None, None
    
    # Get the latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
    print(f"📁 Loading checkpoint: {latest_checkpoint}")
    
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    return checkpoint, latest_checkpoint

def evaluate_model_performance():
    """Evaluate the current model performance"""
    print("🔍 Evaluating Model Performance...")
    print("=" * 50)
    
    # Load the latest checkpoint
    checkpoint, checkpoint_path = load_latest_checkpoint()
    if checkpoint is None:
        return
    
    # Extract epoch number
    epoch_num = int(checkpoint_path.stem.split('_')[-1])
    print(f"📊 Evaluating model from Epoch {epoch_num}")
    
    # Load test data
    test_data_path = 'training/data/ag_news_test.csv'
    if not os.path.exists(test_data_path):
        print(f"❌ Test data not found at {test_data_path}")
        return
    
    test_df = pd.read_csv(test_data_path)
    print(f"📈 Test dataset size: {len(test_df)} samples")
    
    # Initialize trainer and model with default config
    config = {
        'vocab_size': 10000,
        'embed_dim': 128,
        'hidden_dim': 256,
        'num_classes': 4,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32
    }
    trainer = AGNewsTrainer(config)
    trainer.create_model()  # Initialize the model
    
    # Load model architecture
    model = trainer.model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare test data
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    # Create test dataset
    test_dataset = AGNewsDataset(test_texts, test_labels, trainer.tokenizer)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate model
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = correct / total
    print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 Correct: {correct}/{total}")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['World', 'Sports', 'Business', 'Sci/Tech']))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\n🔢 Confusion Matrix:")
    print(cm)
    
    # Training history analysis
    if 'training_history' in checkpoint:
        history = checkpoint['training_history']
        print(f"\n📈 Training History (Epochs 1-{epoch_num}):")
        
        if len(history) > 0:
            # Calculate average metrics
            avg_train_acc = np.mean([h.get('train_accuracy', 0) for h in history])
            avg_train_loss = np.mean([h.get('train_loss', 0) for h in history])
            avg_val_acc = np.mean([h.get('val_accuracy', 0) for h in history])
            avg_val_loss = np.mean([h.get('val_loss', 0) for h in history])
            
            print(f"  Average Training Accuracy: {avg_train_acc:.4f}")
            print(f"  Average Training Loss: {avg_train_loss:.4f}")
            print(f"  Average Validation Accuracy: {avg_val_acc:.4f}")
            print(f"  Average Validation Loss: {avg_val_loss:.4f}")
            
            # Check for overfitting
            if avg_train_acc - avg_val_acc > 0.1:
                print("⚠️  Warning: Potential overfitting detected (train acc >> val acc)")
            elif avg_val_acc - avg_train_acc > 0.05:
                print("✅ Good generalization (val acc > train acc)")
            else:
                print("✅ Balanced training (train acc ≈ val acc)")
    
    # Training progress analysis
    print(f"\n🚀 Training Progress Analysis:")
    print(f"  Current Epoch: {epoch_num}/100")
    print(f"  Progress: {epoch_num}%")
    
    if epoch_num >= 20:
        print("✅ Sufficient training epochs completed for initial evaluation")
    else:
        print("⏳ Still in early training phase")
    
    # Performance recommendations
    print(f"\n💡 Recommendations:")
    if accuracy > 0.85:
        print("🎉 Excellent performance! Model is well-trained.")
    elif accuracy > 0.75:
        print("✅ Good performance! Model shows solid learning.")
    elif accuracy > 0.65:
        print("📈 Decent performance, but more training may help.")
    else:
        print("⚠️  Low performance. Consider:")
        print("   - More training epochs")
        print("   - Learning rate adjustment")
        print("   - Model architecture changes")
    
    if epoch_num < 50:
        print(f"🔄 Consider continuing training to epoch 50+ for better performance")
    elif epoch_num < 100:
        print(f"⏳ Training is progressing well, continue to completion")
    else:
        print(f"🏁 Training complete! Final evaluation ready.")

if __name__ == "__main__":
    evaluate_model_performance()
