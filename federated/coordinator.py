#!/usr/bin/env python3
"""
Federated Learning Coordinator
Manages distributed training across multiple devices
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import requests
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FederatedRound:
    """Represents a federated learning round"""
    round_id: str
    global_model_version: int
    participating_devices: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "active"  # active, completed, failed
    aggregated_model_path: Optional[str] = None
    convergence_threshold: float = 0.01
    max_rounds: int = 10

@dataclass
class DeviceUpdate:
    """Represents a model update from a device"""
    device_id: str
    round_id: str
    model_weights: Dict[str, np.ndarray]
    sample_count: int
    loss: float
    accuracy: float
    timestamp: datetime

class FederatedLearningCoordinator:
    """Coordinates federated learning across multiple devices"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.active_rounds: Dict[str, FederatedRound] = {}
        self.device_updates: Dict[str, List[DeviceUpdate]] = {}
        self.global_model_version = 0
        self.models_dir = Path("federated_models")
        self.models_dir.mkdir(exist_ok=True)
        
    async def start_federated_training(
        self, 
        job_id: str, 
        dataset_name: str,
        model_type: str,
        config: Dict,
        min_devices: int = 2,
        max_devices: int = 10
    ) -> str:
        """Start a new federated learning round"""
        
        # Get available devices
        devices = await self._get_available_devices()
        if len(devices) < min_devices:
            raise ValueError(f"Not enough devices available. Need {min_devices}, got {len(devices)}")
        
        # Select participating devices
        participating_devices = devices[:max_devices]
        
        # Create federated round
        round_id = f"fed_round_{int(time.time())}"
        round_info = FederatedRound(
            round_id=round_id,
            global_model_version=self.global_model_version,
            participating_devices=participating_devices,
            start_time=datetime.utcnow(),
            max_rounds=config.get("max_rounds", 10),
            convergence_threshold=config.get("convergence_threshold", 0.01)
        )
        
        self.active_rounds[round_id] = round_info
        self.device_updates[round_id] = []
        
        logger.info(f"🚀 Starting federated round {round_id} with {len(participating_devices)} devices")
        
        # Distribute data and start training
        await self._distribute_data(round_id, dataset_name, participating_devices)
        await self._start_device_training(round_id, job_id, model_type, config)
        
        return round_id
    
    async def _get_available_devices(self) -> List[str]:
        """Get list of available devices"""
        try:
            response = requests.get(f"{self.server_url}/devices")
            response.raise_for_status()
            devices = response.json()
            return [device["id"] for device in devices if device.get("is_active", False)]
        except Exception as e:
            logger.error(f"Failed to get devices: {e}")
            return []
    
    async def _distribute_data(self, round_id: str, dataset_name: str, devices: List[str]):
        """Distribute dataset across participating devices"""
        logger.info(f"📊 Distributing {dataset_name} dataset across {len(devices)} devices")
        
        # Load and split dataset
        dataset_path = f"training/data/{dataset_name}_train.csv"
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Read dataset
        import pandas as pd
        df = pd.read_csv(dataset_path)
        
        # Split data across devices
        chunk_size = len(df) // len(devices)
        for i, device_id in enumerate(devices):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < len(devices) - 1 else len(df)
            
            device_data = df.iloc[start_idx:end_idx]
            device_data_path = f"federated_data/{device_id}_{round_id}.csv"
            
            # Create directory if it doesn't exist
            Path("federated_data").mkdir(exist_ok=True)
            
            # Save device-specific data
            device_data.to_csv(device_data_path, index=False)
            
            logger.info(f"📱 Device {device_id}: {len(device_data)} samples ({start_idx}-{end_idx})")
    
    async def _start_device_training(self, round_id: str, job_id: str, model_type: str, config: Dict):
        """Start training on participating devices"""
        round_info = self.active_rounds[round_id]
        
        for device_id in round_info.participating_devices:
            try:
                # Create federated training job for this device
                fed_job = {
                    "name": f"Federated Round {round_id} - Device {device_id}",
                    "model_type": model_type,
                    "total_epochs": config.get("local_epochs", 3),
                    "config": {
                        **config,
                        "federated_round_id": round_id,
                        "device_data_path": f"federated_data/{device_id}_{round_id}.csv",
                        "is_federated": True
                    }
                }
                
                # Send job to device
                response = requests.post(
                    f"{self.server_url}/devices/{device_id}/federated-job",
                    json=fed_job,
                    headers={"Authorization": "Bearer constellation-token"}
                )
                response.raise_for_status()
                
                logger.info(f"✅ Started federated training on device {device_id}")
                
            except Exception as e:
                logger.error(f"Failed to start training on device {device_id}: {e}")
    
    async def collect_device_updates(self, round_id: str) -> bool:
        """Collect model updates from all participating devices"""
        round_info = self.active_rounds[round_id]
        expected_devices = len(round_info.participating_devices)
        
        # Check for updates from all devices
        for device_id in round_info.participating_devices:
            try:
                response = requests.get(
                    f"{self.server_url}/devices/{device_id}/federated-update/{round_id}",
                    headers={"Authorization": "Bearer constellation-token"}
                )
                
                if response.status_code == 200:
                    update_data = response.json()
                    if update_data.get("status") == "ready":
                        # Download model weights
                        model_weights = await self._download_model_weights(device_id, round_id)
                        
                        update = DeviceUpdate(
                            device_id=device_id,
                            round_id=round_id,
                            model_weights=model_weights,
                            sample_count=update_data.get("sample_count", 0),
                            loss=update_data.get("loss", 0.0),
                            accuracy=update_data.get("accuracy", 0.0),
                            timestamp=datetime.utcnow()
                        )
                        
                        self.device_updates[round_id].append(update)
                        logger.info(f"📥 Collected update from device {device_id}")
                
            except Exception as e:
                logger.warning(f"Failed to collect update from device {device_id}: {e}")
        
        return len(self.device_updates[round_id]) >= expected_devices
    
    async def _download_model_weights(self, device_id: str, round_id: str) -> Dict[str, np.ndarray]:
        """Download model weights from device"""
        try:
            response = requests.get(
                f"{self.server_url}/devices/{device_id}/model-weights/{round_id}",
                headers={"Authorization": "Bearer constellation-token"}
            )
            response.raise_for_status()
            
            # Convert weights to numpy arrays
            weights_data = response.json()
            model_weights = {}
            
            for layer_name, weights in weights_data.items():
                model_weights[layer_name] = np.array(weights)
            
            return model_weights
            
        except Exception as e:
            logger.error(f"Failed to download weights from device {device_id}: {e}")
            return {}
    
    async def aggregate_models(self, round_id: str) -> str:
        """Aggregate model updates using Federated Averaging"""
        round_info = self.active_rounds[round_id]
        updates = self.device_updates[round_id]
        
        if not updates:
            raise ValueError("No updates to aggregate")
        
        logger.info(f"🔄 Aggregating {len(updates)} model updates")
        
        # Calculate total samples
        total_samples = sum(update.sample_count for update in updates)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        
        # Get layer names from first update
        first_update = updates[0]
        for layer_name in first_update.model_weights.keys():
            aggregated_weights[layer_name] = np.zeros_like(first_update.model_weights[layer_name])
        
        # Weighted average of model weights
        for update in updates:
            weight_factor = update.sample_count / total_samples
            
            for layer_name, weights in update.model_weights.items():
                aggregated_weights[layer_name] += weight_factor * weights
        
        # Save aggregated model
        model_path = self.models_dir / f"global_model_round_{round_id}.pth"
        torch.save({
            "model_weights": aggregated_weights,
            "round_id": round_id,
            "global_version": self.global_model_version + 1,
            "participating_devices": [u.device_id for u in updates],
            "total_samples": total_samples,
            "timestamp": datetime.utcnow().isoformat()
        }, model_path)
        
        # Update global model version
        self.global_model_version += 1
        round_info.aggregated_model_path = str(model_path)
        
        logger.info(f"✅ Model aggregated and saved to {model_path}")
        return str(model_path)
    
    async def distribute_global_model(self, round_id: str):
        """Distribute the aggregated global model to all devices"""
        round_info = self.active_rounds[round_id]
        
        if not round_info.aggregated_model_path:
            raise ValueError("No aggregated model to distribute")
        
        logger.info(f"📤 Distributing global model to {len(round_info.participating_devices)} devices")
        
        # Load aggregated model
        model_data = torch.load(round_info.aggregated_model_path, map_location='cpu')
        model_weights = model_data["model_weights"]
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_weights = {}
        for layer_name, weights in model_weights.items():
            serializable_weights[layer_name] = weights.tolist()
        
        # Distribute to all devices
        for device_id in round_info.participating_devices:
            try:
                response = requests.post(
                    f"{self.server_url}/devices/{device_id}/global-model",
                    json={
                        "round_id": round_id,
                        "global_version": self.global_model_version,
                        "model_weights": serializable_weights
                    },
                    headers={"Authorization": "Bearer constellation-token"}
                )
                response.raise_for_status()
                
                logger.info(f"✅ Distributed global model to device {device_id}")
                
            except Exception as e:
                logger.error(f"Failed to distribute model to device {device_id}: {e}")
    
    async def run_federated_round(self, round_id: str) -> bool:
        """Run a complete federated learning round"""
        round_info = self.active_rounds[round_id]
        
        logger.info(f"🔄 Running federated round {round_id}")
        
        # Wait for all devices to complete training
        max_wait_time = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            if await self.collect_device_updates(round_id):
                break
            await asyncio.sleep(10)  # Check every 10 seconds
        
        if not self.device_updates[round_id]:
            logger.error(f"❌ No updates received for round {round_id}")
            return False
        
        # Aggregate models
        await self.aggregate_models(round_id)
        
        # Distribute global model
        await self.distribute_global_model(round_id)
        
        # Mark round as completed
        round_info.end_time = datetime.utcnow()
        round_info.status = "completed"
        
        logger.info(f"✅ Federated round {round_id} completed successfully")
        return True
    
    async def run_federated_learning(
        self, 
        job_id: str, 
        dataset_name: str,
        model_type: str,
        config: Dict
    ) -> str:
        """Run complete federated learning process"""
        
        logger.info("🚀 Starting federated learning process")
        
        # Start initial round
        round_id = await self.start_federated_training(job_id, dataset_name, model_type, config)
        
        # Run multiple rounds
        for round_num in range(config.get("max_rounds", 10)):
            logger.info(f"🔄 Starting round {round_num + 1}")
            
            success = await self.run_federated_round(round_id)
            if not success:
                logger.error(f"❌ Round {round_num + 1} failed")
                break
            
            # Check for convergence (simplified)
            if round_num > 0:
                # In a real implementation, you'd check model convergence
                logger.info(f"✅ Round {round_num + 1} completed")
        
        logger.info("🎉 Federated learning process completed")
        return round_id

# Example usage
async def main():
    coordinator = FederatedLearningCoordinator()
    
    config = {
        "local_epochs": 3,
        "max_rounds": 5,
        "convergence_threshold": 0.01,
        "learning_rate": 0.001,
        "batch_size": 32
    }
    
    round_id = await coordinator.run_federated_learning(
        job_id="fed_job_001",
        dataset_name="ag_news",
        model_type="text_classification",
        config=config
    )
    
    print(f"Federated learning completed: {round_id}")

if __name__ == "__main__":
    asyncio.run(main())
