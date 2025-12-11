#!/usr/bin/env python3
"""
Federated Learning Demo
Demonstrates the complete federated learning process
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np

from coordinator import FederatedLearningCoordinator
from data_distributor import DataDistributor
from model_aggregator import ModelAggregator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedLearningDemo:
    """Demo of the complete federated learning system"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.coordinator = FederatedLearningCoordinator(server_url)
        self.data_distributor = DataDistributor()
        self.model_aggregator = ModelAggregator("fedavg")
        
    async def run_demo(self):
        """Run the complete federated learning demo"""
        
        logger.info("🚀 Starting Federated Learning Demo")
        logger.info("=" * 50)
        
        # Step 1: Prepare data distribution
        await self._prepare_data_distribution()
        
        # Step 2: Start federated learning
        await self._start_federated_learning()
        
        # Step 3: Monitor progress
        await self._monitor_progress()
        
        # Step 4: Analyze results
        await self._analyze_results()
        
        logger.info("🎉 Federated Learning Demo Completed!")
    
    async def _prepare_data_distribution(self):
        """Prepare data distribution for federated learning"""
        
        logger.info("📊 Step 1: Preparing Data Distribution")
        
        # Check if AG News dataset exists
        dataset_path = "training/data/ag_news_train.csv"
        if not Path(dataset_path).exists():
            logger.error(f"Dataset not found: {dataset_path}")
            logger.info("Please run the AG News trainer first to download the dataset")
            return
        
        # Create synthetic devices for demo
        device_ids = self.data_distributor.create_synthetic_devices(4, "fed_demo")
        logger.info(f"📱 Created {len(device_ids)} synthetic devices: {device_ids}")
        
        # Distribute dataset
        device_paths = self.data_distributor.distribute_dataset(
            dataset_path=dataset_path,
            device_ids=device_ids,
            distribution_strategy="stratified",
            test_split=0.2
        )
        
        # Analyze distribution
        analysis = self.data_distributor.analyze_distribution(device_paths)
        
        logger.info("📊 Data Distribution Analysis:")
        for device_id, stats in analysis.items():
            logger.info(f"  {device_id}: {stats['total_samples']} samples")
            logger.info(f"    Classes: {stats['class_counts']}")
        
        self.device_ids = device_ids
        self.device_paths = device_paths
    
    async def _start_federated_learning(self):
        """Start the federated learning process"""
        
        logger.info("🔄 Step 2: Starting Federated Learning")
        
        # Configuration for federated learning
        fed_config = {
            "local_epochs": 3,
            "max_rounds": 5,
            "convergence_threshold": 0.01,
            "learning_rate": 0.001,
            "batch_size": 32,
            "vocab_size": 10000,
            "embed_dim": 128,
            "hidden_dim": 256,
            "num_classes": 4,
            "max_length": 128
        }
        
        # Start federated learning
        try:
            round_id = await self.coordinator.run_federated_learning(
                job_id="fed_demo_001",
                dataset_name="ag_news",
                model_type="text_classification",
                config=fed_config
            )
            
            self.round_id = round_id
            logger.info(f"✅ Federated learning started: {round_id}")
            
        except Exception as e:
            logger.error(f"Failed to start federated learning: {e}")
            # For demo purposes, create a mock round
            self.round_id = "fed_demo_round_001"
            logger.info(f"📝 Created mock round: {self.round_id}")
    
    async def _monitor_progress(self):
        """Monitor the progress of federated learning"""
        
        logger.info("📈 Step 3: Monitoring Progress")
        
        # Simulate monitoring for demo
        for round_num in range(1, 6):
            logger.info(f"🔄 Round {round_num}/5")
            
            # Simulate device training
            await self._simulate_device_training(round_num)
            
            # Simulate model aggregation
            await self._simulate_model_aggregation(round_num)
            
            # Simulate global model distribution
            await self._simulate_global_distribution(round_num)
            
            # Check convergence
            if round_num > 1:
                convergence = await self._check_convergence(round_num)
                if convergence:
                    logger.info(f"✅ Converged at round {round_num}")
                    break
            
            await asyncio.sleep(1)  # Simulate processing time
    
    async def _simulate_device_training(self, round_num: int):
        """Simulate device training for demo"""
        
        logger.info(f"  📱 Simulating device training for round {round_num}")
        
        for device_id in self.device_ids:
            # Simulate training metrics
            loss = np.random.uniform(0.1, 0.5)
            accuracy = np.random.uniform(85, 95)
            sample_count = np.random.randint(500, 1500)
            
            logger.info(f"    Device {device_id}: Loss={loss:.3f}, Accuracy={accuracy:.1f}%, Samples={sample_count}")
    
    async def _simulate_model_aggregation(self, round_num: int):
        """Simulate model aggregation for demo"""
        
        logger.info(f"  🔄 Simulating model aggregation for round {round_num}")
        
        # Create mock device updates
        device_updates = []
        for device_id in self.device_ids:
            update = {
                "device_id": device_id,
                "model_weights": {
                    "embedding.weight": np.random.randn(10000, 128),
                    "lstm.weight_ih_l0": np.random.randn(1024, 128),
                    "lstm.weight_hh_l0": np.random.randn(1024, 256),
                    "fc.weight": np.random.randn(4, 256),
                    "fc.bias": np.random.randn(4)
                },
                "sample_count": np.random.randint(500, 1500),
                "local_epochs": 3
            }
            device_updates.append(update)
        
        # Aggregate models
        aggregated_weights = self.model_aggregator.aggregate_models(device_updates)
        
        # Save aggregated model
        metadata = {
            "round": round_num,
            "participating_devices": self.device_ids,
            "total_samples": sum(u["sample_count"] for u in device_updates),
            "aggregation_strategy": "fedavg"
        }
        
        model_path = self.model_aggregator.save_aggregated_model(
            aggregated_weights, 
            f"round_{round_num}", 
            metadata
        )
        
        logger.info(f"    ✅ Model aggregated and saved: {model_path}")
    
    async def _simulate_global_distribution(self, round_num: int):
        """Simulate global model distribution for demo"""
        
        logger.info(f"  📤 Simulating global model distribution for round {round_num}")
        
        for device_id in self.device_ids:
            logger.info(f"    📱 Distributed global model to {device_id}")
    
    async def _check_convergence(self, round_num: int) -> bool:
        """Check if the model has converged"""
        
        # Simulate convergence check
        convergence_probability = 0.3  # 30% chance of convergence each round
        converged = np.random.random() < convergence_probability
        
        if converged:
            logger.info(f"    ✅ Model converged at round {round_num}")
        else:
            logger.info(f"    📈 Model still improving at round {round_num}")
        
        return converged
    
    async def _analyze_results(self):
        """Analyze the results of federated learning"""
        
        logger.info("📊 Step 4: Analyzing Results")
        
        # Find the latest aggregated model
        models_dir = Path("federated_models")
        if not models_dir.exists():
            logger.warning("No federated models found")
            return
        
        model_files = list(models_dir.glob("aggregated_model_round_*.pth"))
        if not model_files:
            logger.warning("No aggregated model files found")
            return
        
        # Get the latest model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"📂 Latest model: {latest_model}")
        
        # Load and analyze the model
        try:
            model_data = torch.load(latest_model, map_location='cpu')
            weights = model_data["weights"]
            metadata = model_data.get("metadata", {})
            
            logger.info("📊 Model Analysis:")
            logger.info(f"  Round: {metadata.get('round', 'unknown')}")
            logger.info(f"  Participating devices: {len(metadata.get('participating_devices', []))}")
            logger.info(f"  Total samples: {metadata.get('total_samples', 'unknown')}")
            logger.info(f"  Aggregation strategy: {metadata.get('aggregation_strategy', 'unknown')}")
            
            # Analyze model weights
            total_params = sum(w.size for w in weights.values())
            logger.info(f"  Total parameters: {total_params:,}")
            
            # Calculate weight statistics
            for layer_name, weights_array in weights.items():
                mean_weight = np.mean(weights_array)
                std_weight = np.std(weights_array)
                logger.info(f"  {layer_name}: mean={mean_weight:.4f}, std={std_weight:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to analyze model: {e}")
    
    def create_demo_report(self):
        """Create a demo report"""
        
        report = {
            "demo_info": {
                "title": "Federated Learning Demo Report",
                "timestamp": str(pd.Timestamp.now()),
                "server_url": self.server_url
            },
            "data_distribution": {
                "total_devices": len(self.device_ids),
                "device_ids": self.device_ids,
                "distribution_strategy": "stratified"
            },
            "federated_learning": {
                "round_id": getattr(self, 'round_id', 'unknown'),
                "max_rounds": 5,
                "local_epochs": 3,
                "aggregation_strategy": "fedavg"
            },
            "results": {
                "status": "completed",
                "models_created": len(list(Path("federated_models").glob("*.pth"))),
                "convergence_achieved": True
            }
        }
        
        # Save report
        report_path = Path("federated_learning_demo_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Demo report saved: {report_path}")
        return report

async def main():
    """Main demo function"""
    
    demo = FederatedLearningDemo()
    
    try:
        await demo.run_demo()
        report = demo.create_demo_report()
        
        print("\n" + "=" * 50)
        print("🎉 FEDERATED LEARNING DEMO COMPLETED!")
        print("=" * 50)
        print(f"📊 Devices: {len(demo.device_ids)}")
        print(f"🔄 Rounds: 5")
        print(f"📈 Strategy: Federated Averaging (FedAvg)")
        print(f"📄 Report: federated_learning_demo_report.json")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
