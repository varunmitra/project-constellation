#!/usr/bin/env python3
"""
Federated Learning Client
Runs on each device to participate in federated learning
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import requests
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedDataset(Dataset):
    """Dataset for federated learning"""
    
    def __init__(self, data_path: str, max_length: int = 128):
        self.data_path = data_path
        self.max_length = max_length
        self.texts, self.labels = self._load_data()
    
    def _load_data(self):
        """Load data from CSV file"""
        df = pd.read_csv(self.data_path)
        
        # Assuming CSV has 'text' and 'label' columns
        if 'text' in df.columns and 'label' in df.columns:
            texts = df['text'].tolist()
            labels = df['label'].tolist()
        else:
            # For AG News format: label, title, description
            texts = (df.iloc[:, 1] + " " + df.iloc[:, 2]).tolist()  # title + description
            labels = df.iloc[:, 0].tolist()
        
        return texts, labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Simple tokenization
        tokens = text.split()[:self.max_length]
        token_ids = [hash(token) % 10000 for token in tokens]
        token_ids = token_ids + [0] * (self.max_length - len(token_ids))
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class SimpleLSTM(nn.Module):
    """Simple LSTM model for text classification"""
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, 
                 hidden_dim: int = 256, num_classes: int = 4, max_length: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Use the last hidden state
        output = self.fc(self.dropout(hidden[-1]))
        return output

class FederatedLearningClient:
    """Client for participating in federated learning"""
    
    def __init__(self, device_id: str, server_url: str = "http://localhost:8000"):
        self.device_id = device_id
        self.server_url = server_url
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.current_round = None
        self.global_model_version = 0
        
    async def start_federated_training(self, job_config: Dict):
        """Start federated training on this device"""
        self.current_round = job_config.get("federated_round_id")
        data_path = job_config.get("device_data_path")
        
        if not data_path or not Path(data_path).exists():
            raise FileNotFoundError(f"Device data not found: {data_path}")
        
        logger.info(f"🚀 Starting federated training on device {self.device_id}")
        logger.info(f"📊 Data path: {data_path}")
        
        # Initialize model
        self.model = SimpleLSTM(
            vocab_size=job_config.get("vocab_size", 10000),
            embed_dim=job_config.get("embed_dim", 128),
            hidden_dim=job_config.get("hidden_dim", 256),
            num_classes=job_config.get("num_classes", 4),
            max_length=job_config.get("max_length", 128)
        )
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=job_config.get("learning_rate", 0.001)
        )
        
        # Load local data
        dataset = FederatedDataset(data_path, job_config.get("max_length", 128))
        dataloader = DataLoader(
            dataset, 
            batch_size=job_config.get("batch_size", 32), 
            shuffle=True
        )
        
        # Train locally
        local_epochs = job_config.get("local_epochs", 3)
        total_loss = 0.0
        correct = 0
        total = 0
        
        self.model.train()
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                pred = output.argmax(dim=1)
                epoch_correct += pred.eq(target).sum().item()
                epoch_total += target.size(0)
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / len(dataloader)
            accuracy = 100.0 * epoch_correct / epoch_total
            
            logger.info(f"Epoch {epoch+1}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            total_loss += avg_loss
            correct += epoch_correct
            total += epoch_total
        
        # Calculate final metrics
        final_loss = total_loss / local_epochs
        final_accuracy = 100.0 * correct / total
        
        # Save model weights
        await self._save_model_weights()
        
        # Send update to server
        await self._send_update(final_loss, final_accuracy, len(dataset))
        
        logger.info(f"✅ Local training completed. Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.2f}%")
    
    async def _save_model_weights(self):
        """Save model weights for aggregation"""
        if not self.model:
            return
        
        weights = {}
        for name, param in self.model.named_parameters():
            weights[name] = param.data.cpu().numpy()
        
        # Save to local file
        weights_path = f"federated_weights/{self.device_id}_{self.current_round}.json"
        Path("federated_weights").mkdir(exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_weights = {}
        for name, weights_array in weights.items():
            serializable_weights[name] = weights_array.tolist()
        
        with open(weights_path, 'w') as f:
            json.dump(serializable_weights, f)
        
        logger.info(f"💾 Model weights saved to {weights_path}")
    
    async def _send_update(self, loss: float, accuracy: float, sample_count: int):
        """Send model update to server"""
        try:
            response = requests.post(
                f"{self.server_url}/devices/{self.device_id}/federated-update",
                json={
                    "round_id": self.current_round,
                    "loss": loss,
                    "accuracy": accuracy,
                    "sample_count": sample_count,
                    "status": "ready"
                },
                headers={"Authorization": "Bearer constellation-token"}
            )
            response.raise_for_status()
            logger.info("📤 Model update sent to server")
            
        except Exception as e:
            logger.error(f"Failed to send update: {e}")
    
    async def receive_global_model(self, global_weights: Dict[str, List]):
        """Receive and apply global model weights"""
        if not self.model:
            logger.error("No model initialized")
            return
        
        logger.info("📥 Receiving global model weights")
        
        # Convert lists back to numpy arrays
        weights = {}
        for name, weights_list in global_weights.items():
            weights[name] = np.array(weights_list)
        
        # Apply global weights to local model
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in weights:
                    param.data = torch.from_numpy(weights[name])
        
        logger.info("✅ Global model weights applied")
    
    async def get_model_weights(self, round_id: str) -> Dict[str, List]:
        """Get current model weights for server"""
        if not self.model:
            return {}
        
        weights = {}
        for name, param in self.model.named_parameters():
            weights[name] = param.data.cpu().numpy().tolist()
        
        return weights

class FederatedClientService:
    """Service to handle federated learning requests"""
    
    def __init__(self, device_id: str, server_url: str = "http://localhost:8000"):
        self.device_id = device_id
        self.server_url = server_url
        self.client = FederatedLearningClient(device_id, server_url)
        self.is_training = False
    
    async def handle_federated_job(self, job_config: Dict):
        """Handle incoming federated training job"""
        if self.is_training:
            logger.warning("Already training, ignoring new job")
            return
        
        self.is_training = True
        try:
            await self.client.start_federated_training(job_config)
        finally:
            self.is_training = False
    
    async def handle_global_model(self, global_weights: Dict[str, List]):
        """Handle incoming global model"""
        await self.client.receive_global_model(global_weights)
    
    async def get_model_weights(self, round_id: str) -> Dict[str, List]:
        """Get model weights for aggregation"""
        return await self.client.get_model_weights(round_id)

# Example usage
async def main():
    device_id = "demo-device-001"
    client_service = FederatedClientService(device_id)
    
    # Simulate receiving a federated job
    job_config = {
        "federated_round_id": "fed_round_001",
        "device_data_path": "federated_data/demo-device-001_fed_round_001.csv",
        "local_epochs": 3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "vocab_size": 10000,
        "embed_dim": 128,
        "hidden_dim": 256,
        "num_classes": 4,
        "max_length": 128
    }
    
    await client_service.handle_federated_job(job_config)

if __name__ == "__main__":
    asyncio.run(main())
