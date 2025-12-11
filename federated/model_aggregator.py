#!/usr/bin/env python3
"""
Model Aggregation Utility
Implements various aggregation strategies for federated learning
"""

import numpy as np
import torch
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelAggregator:
    """Aggregates model updates from multiple devices"""
    
    def __init__(self, aggregation_strategy: str = "fedavg"):
        self.aggregation_strategy = aggregation_strategy
        self.aggregation_history = []
    
    def aggregate_models(
        self, 
        device_updates: List[Dict], 
        global_model_weights: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate model updates from multiple devices
        
        Args:
            device_updates: List of device updates with model weights and sample counts
            global_model_weights: Previous global model weights (for momentum, etc.)
        
        Returns:
            Aggregated model weights
        """
        
        if not device_updates:
            raise ValueError("No device updates to aggregate")
        
        logger.info(f"🔄 Aggregating {len(device_updates)} model updates using {self.aggregation_strategy}")
        
        if self.aggregation_strategy == "fedavg":
            return self._federated_averaging(device_updates)
        elif self.aggregation_strategy == "fedprox":
            return self._federated_proximal(device_updates, global_model_weights)
        elif self.aggregation_strategy == "fednova":
            return self._federated_nova(device_updates)
        elif self.aggregation_strategy == "scaffold":
            return self._scaffold(device_updates, global_model_weights)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation_strategy}")
    
    def _federated_averaging(self, device_updates: List[Dict]) -> Dict[str, np.ndarray]:
        """Federated Averaging (FedAvg) aggregation"""
        logger.info("📊 Using Federated Averaging (FedAvg)")
        
        # Calculate total samples
        total_samples = sum(update.get("sample_count", 1) for update in device_updates)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        
        # Get layer names from first update
        first_update = device_updates[0]
        for layer_name in first_update["model_weights"].keys():
            aggregated_weights[layer_name] = np.zeros_like(first_update["model_weights"][layer_name])
        
        # Weighted average of model weights
        for update in device_updates:
            weight_factor = update.get("sample_count", 1) / total_samples
            model_weights = update["model_weights"]
            
            for layer_name, weights in model_weights.items():
                if layer_name in aggregated_weights:
                    aggregated_weights[layer_name] += weight_factor * weights
        
        # Log aggregation details
        logger.info(f"📊 Total samples: {total_samples}")
        for update in device_updates:
            device_id = update.get("device_id", "unknown")
            sample_count = update.get("sample_count", 0)
            weight_factor = sample_count / total_samples
            logger.info(f"📱 Device {device_id}: {sample_count} samples ({weight_factor:.2%})")
        
        return aggregated_weights
    
    def _federated_proximal(self, device_updates: List[Dict], global_model_weights: Optional[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Federated Proximal (FedProx) aggregation with regularization"""
        logger.info("📊 Using Federated Proximal (FedProx)")
        
        if global_model_weights is None:
            logger.warning("No global model weights provided, falling back to FedAvg")
            return self._federated_averaging(device_updates)
        
        # Similar to FedAvg but with proximal term
        aggregated_weights = self._federated_averaging(device_updates)
        
        # Add proximal regularization (simplified)
        mu = 0.01  # Proximal parameter
        for layer_name in aggregated_weights:
            if layer_name in global_model_weights:
                aggregated_weights[layer_name] = (
                    aggregated_weights[layer_name] + 
                    mu * global_model_weights[layer_name]
                ) / (1 + mu)
        
        return aggregated_weights
    
    def _federated_nova(self, device_updates: List[Dict]) -> Dict[str, np.ndarray]:
        """Federated Nova aggregation (simplified version)"""
        logger.info("📊 Using Federated Nova (FedNova)")
        
        # Calculate total samples
        total_samples = sum(update.get("sample_count", 1) for update in device_updates)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        
        # Get layer names from first update
        first_update = device_updates[0]
        for layer_name in first_update["model_weights"].keys():
            aggregated_weights[layer_name] = np.zeros_like(first_update["model_weights"][layer_name])
        
        # Normalized averaging (simplified FedNova)
        for update in device_updates:
            weight_factor = update.get("sample_count", 1) / total_samples
            model_weights = update["model_weights"]
            
            # Normalize by local epochs (simplified)
            local_epochs = update.get("local_epochs", 1)
            normalized_factor = weight_factor / local_epochs
            
            for layer_name, weights in model_weights.items():
                if layer_name in aggregated_weights:
                    aggregated_weights[layer_name] += normalized_factor * weights
        
        return aggregated_weights
    
    def _scaffold(self, device_updates: List[Dict], global_model_weights: Optional[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """SCAFFOLD aggregation (simplified version)"""
        logger.info("📊 Using SCAFFOLD")
        
        if global_model_weights is None:
            logger.warning("No global model weights provided, falling back to FedAvg")
            return self._federated_averaging(device_updates)
        
        # Simplified SCAFFOLD implementation
        aggregated_weights = self._federated_averaging(device_updates)
        
        # Add control variates (simplified)
        for layer_name in aggregated_weights:
            if layer_name in global_model_weights:
                # Simple control variate update
                aggregated_weights[layer_name] = (
                    0.9 * aggregated_weights[layer_name] + 
                    0.1 * global_model_weights[layer_name]
                )
        
        return aggregated_weights
    
    def calculate_model_drift(self, old_weights: Dict[str, np.ndarray], new_weights: Dict[str, np.ndarray]) -> float:
        """Calculate the drift between old and new model weights"""
        total_drift = 0.0
        total_params = 0
        
        for layer_name in old_weights:
            if layer_name in new_weights:
                old_layer = old_weights[layer_name]
                new_layer = new_weights[layer_name]
                
                # Calculate L2 norm of the difference
                drift = np.linalg.norm(new_layer - old_layer)
                total_drift += drift
                total_params += old_layer.size
        
        return total_drift / total_params if total_params > 0 else 0.0
    
    def check_convergence(self, weights_history: List[Dict[str, np.ndarray]], threshold: float = 0.01) -> bool:
        """Check if the model has converged based on weight changes"""
        if len(weights_history) < 2:
            return False
        
        recent_weights = weights_history[-1]
        previous_weights = weights_history[-2]
        
        drift = self.calculate_model_drift(previous_weights, recent_weights)
        
        logger.info(f"📊 Model drift: {drift:.6f} (threshold: {threshold})")
        
        return drift < threshold
    
    def save_aggregated_model(
        self, 
        weights: Dict[str, np.ndarray], 
        round_id: str, 
        metadata: Dict
    ) -> str:
        """Save aggregated model weights"""
        
        model_data = {
            "weights": {k: v.tolist() for k, v in weights.items()},  # Convert to lists for JSON
            "round_id": round_id,
            "aggregation_strategy": self.aggregation_strategy,
            "metadata": metadata,
            "timestamp": str(pd.Timestamp.now())
        }
        
        # Save as JSON
        model_path = Path("federated_models") / f"aggregated_model_round_{round_id}.json"
        model_path.parent.mkdir(exist_ok=True)
        
        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        # Also save as PyTorch model
        torch_path = Path("federated_models") / f"aggregated_model_round_{round_id}.pth"
        torch.save({
            "weights": weights,
            "round_id": round_id,
            "aggregation_strategy": self.aggregation_strategy,
            "metadata": metadata
        }, torch_path)
        
        logger.info(f"💾 Aggregated model saved: {model_path}")
        logger.info(f"💾 PyTorch model saved: {torch_path}")
        
        return str(model_path)
    
    def load_aggregated_model(self, model_path: str) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Load aggregated model weights"""
        
        if model_path.endswith('.json'):
            with open(model_path, 'r') as f:
                model_data = json.load(f)
            
            weights = {k: np.array(v) for k, v in model_data["weights"].items()}
            metadata = model_data.get("metadata", {})
            
        elif model_path.endswith('.pth'):
            model_data = torch.load(model_path, map_location='cpu')
            weights = model_data["weights"]
            metadata = model_data.get("metadata", {})
        
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
        
        logger.info(f"📂 Loaded aggregated model: {model_path}")
        
        return weights, metadata

# Example usage
def main():
    # Create sample device updates
    device_updates = [
        {
            "device_id": "device_001",
            "model_weights": {
                "fc.weight": np.random.randn(4, 256),
                "fc.bias": np.random.randn(4)
            },
            "sample_count": 1000,
            "local_epochs": 3
        },
        {
            "device_id": "device_002", 
            "model_weights": {
                "fc.weight": np.random.randn(4, 256),
                "fc.bias": np.random.randn(4)
            },
            "sample_count": 1500,
            "local_epochs": 3
        }
    ]
    
    # Test different aggregation strategies
    strategies = ["fedavg", "fedprox", "fednova", "scaffold"]
    
    for strategy in strategies:
        print(f"\n🔄 Testing {strategy.upper()}")
        
        aggregator = ModelAggregator(strategy)
        aggregated_weights = aggregator.aggregate_models(device_updates)
        
        print(f"📊 Aggregated weights shape: {aggregated_weights['fc.weight'].shape}")
        print(f"📊 Aggregated bias shape: {aggregated_weights['fc.bias'].shape}")

if __name__ == "__main__":
    main()
