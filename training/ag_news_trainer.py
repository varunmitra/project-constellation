#!/usr/bin/env python3
"""
AG News Text Classification Trainer
A simple neural network for news article classification
Perfect for testing distributed training systems
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import requests
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AGNewsDataset(Dataset):
    """AG News Dataset for text classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Simple tokenization (split by spaces)
        tokens = text.split()[:self.max_length]
        
        # Convert to tensor
        token_ids = [hash(token) % 10000 for token in tokens]  # Simple hashing
        token_ids = token_ids + [0] * (self.max_length - len(token_ids))  # Padding
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class SimpleTextClassifier(nn.Module):
    """Simple neural network for text classification"""
    
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256, num_classes=4):
        super(SimpleTextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Global average pooling
        pooled = torch.mean(lstm_out, dim=1)
        dropped = self.dropout(pooled)
        output = self.classifier(dropped)
        return output

class AGNewsTrainer:
    """AG News Classification Trainer"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        # Class labels
        self.class_labels = ['World', 'Sports', 'Business', 'Sci/Tech']
        
    def download_dataset(self) -> Tuple[List[str], List[int]]:
        """Download and prepare AG News dataset"""
        logger.info("📥 Downloading AG News dataset...")
        
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        # Download training data
        train_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
        test_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
        
        train_file = "data/ag_news_train.csv"
        test_file = "data/ag_news_test.csv"
        
        if not os.path.exists(train_file):
            logger.info("Downloading training data...")
            response = requests.get(train_url)
            with open(train_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
        
        if not os.path.exists(test_file):
            logger.info("Downloading test data...")
            response = requests.get(test_url)
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
        
        # Load and combine data
        train_df = pd.read_csv(train_file, header=None, names=['label', 'title', 'description'])
        test_df = pd.read_csv(test_file, header=None, names=['label', 'title', 'description'])
        
        # Combine title and description
        train_df['text'] = train_df['title'] + ' ' + train_df['description']
        test_df['text'] = test_df['title'] + ' ' + test_df['description']
        
        # Convert labels to 0-based indexing
        train_df['label'] = train_df['label'] - 1
        test_df['label'] = test_df['label'] - 1
        
        # Combine train and test for distributed training
        all_texts = list(train_df['text']) + list(test_df['text'])
        all_labels = list(train_df['label']) + list(test_df['label'])
        
        logger.info(f"📊 Dataset loaded: {len(all_texts)} samples")
        logger.info(f"📊 Class distribution: {np.bincount(all_labels)}")
        
        return all_texts, all_labels
    
    def create_model(self):
        """Create the neural network model"""
        logger.info("🧠 Creating model...")
        
        self.model = SimpleTextClassifier(
            vocab_size=self.config.get('vocab_size', 10000),
            embed_dim=self.config.get('embed_dim', 128),
            hidden_dim=self.config.get('hidden_dim', 256),
            num_classes=4
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.get('learning_rate', 0.001)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def train(self, texts: List[str], labels: List[int], 
              epochs: int = 5, batch_size: int = 32, 
              test_size: float = 0.2) -> Dict:
        """Train the model"""
        logger.info("🚀 Starting training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        logger.info(f"📊 Training samples: {len(X_train)}")
        logger.info(f"📊 Test samples: {len(X_test)}")
        
        # Create datasets
        train_dataset = AGNewsDataset(X_train, y_train, None)
        test_dataset = AGNewsDataset(X_test, y_test, None)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        self.create_model()
        
        # Training loop
        best_accuracy = 0
        training_history = []
        
        for epoch in range(epochs):
            logger.info(f"📈 Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch + 1)
            
            # Evaluate
            test_metrics = self.evaluate(test_loader)
            
            # Log results
            logger.info(f"📊 Train - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.2f}%")
            logger.info(f"📊 Test  - Loss: {test_metrics['loss']:.4f}, Accuracy: {test_metrics['accuracy']:.2f}%")
            
            # Save best model
            if test_metrics['accuracy'] > best_accuracy:
                best_accuracy = test_metrics['accuracy']
                self.save_model('best_model.pth')
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'test_loss': test_metrics['loss'],
                'test_accuracy': test_metrics['accuracy']
            })
        
        # Final evaluation
        final_metrics = self.evaluate(test_loader)
        
        # Generate classification report
        report = classification_report(
            test_metrics['targets'], 
            test_metrics['predictions'],
            target_names=self.class_labels,
            output_dict=True
        )
        
        results = {
            'final_accuracy': final_metrics['accuracy'],
            'best_accuracy': best_accuracy,
            'training_history': training_history,
            'classification_report': report,
            'model_info': {
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'device': str(self.device),
                'config': self.config
            }
        }
        
        logger.info(f"✅ Training completed! Final accuracy: {final_metrics['accuracy']:.2f}%")
        logger.info(f"📊 Best accuracy: {best_accuracy:.2f}%")
        
        return results
    
    def save_model(self, path: str):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"💾 Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"📂 Model loaded from {path}")

def main():
    """Main training function"""
    # Configuration
    config = {
        'vocab_size': 10000,
        'embed_dim': 128,
        'hidden_dim': 256,
        'learning_rate': 0.001,
        'epochs': 5,
        'batch_size': 32
    }
    
    # Create trainer
    trainer = AGNewsTrainer(config)
    
    # Download dataset
    texts, labels = trainer.download_dataset()
    
    # Train model
    results = trainer.train(texts, labels, epochs=config['epochs'], batch_size=config['batch_size'])
    
    # Save results
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("🎉 Training completed successfully!")
    print(f"📊 Final accuracy: {results['final_accuracy']:.2f}%")
    print(f"📊 Best accuracy: {results['best_accuracy']:.2f}%")

if __name__ == "__main__":
    main()
