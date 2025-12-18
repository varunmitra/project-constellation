#!/usr/bin/env python3
"""
Combine multiple aggregated models into one super model
Uses weighted averaging based on model performance metrics
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import json

def load_model_weights(model_path: Path) -> Dict:
    """Load model weights and metadata from checkpoint"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Convert to numpy for aggregation
    weights = {}
    for key, value in state_dict.items():
        try:
            if torch.is_tensor(value):
                # Only process numeric tensors
                if value.dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
                    weights[key] = value.cpu().numpy().astype(np.float32)
            elif isinstance(value, np.ndarray):
                # Only process numeric arrays
                if np.issubdtype(value.dtype, np.number):
                    weights[key] = value.astype(np.float32)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                # Try to convert to numpy array
                try:
                    arr = np.array(value, dtype=np.float32)
                    if arr.size > 0:
                        weights[key] = arr
                except (ValueError, TypeError):
                    # Skip if can't convert to numeric
                    continue
        except Exception as e:
            # Skip problematic values
            print(f"   ⚠️  Skipping {key}: {type(value)} - {str(e)[:50]}")
            continue
    
    # Extract metadata
    metadata = {
        'accuracy': checkpoint.get('avg_accuracy', checkpoint.get('accuracy', 0)),
        'loss': checkpoint.get('avg_loss', checkpoint.get('loss', 0)),
        'samples': checkpoint.get('total_samples', 0),
        'devices': len(checkpoint.get('participating_devices', [])),
        'job_id': checkpoint.get('job_id', 'unknown'),
        'job_name': checkpoint.get('job_name', 'unknown'),
        'config': checkpoint.get('config', {})
    }
    
    return weights, metadata

def calculate_model_weights(metadatas: List[Dict], strategy: str = 'accuracy') -> List[float]:
    """
    Calculate weights for each model based on strategy
    
    Strategies:
    - 'accuracy': Weight by accuracy (higher accuracy = higher weight)
    - 'loss': Weight by inverse loss (lower loss = higher weight)
    - 'samples': Weight by number of samples
    - 'equal': Equal weights for all models
    - 'combined': Weight by accuracy * samples / loss
    """
    if strategy == 'equal':
        return [1.0 / len(metadatas)] * len(metadatas)
    
    elif strategy == 'accuracy':
        accuracies = [m['accuracy'] for m in metadatas]
        total = sum(accuracies)
        return [acc / total if total > 0 else 1.0 / len(metadatas) for acc in accuracies]
    
    elif strategy == 'loss':
        losses = [m['loss'] for m in metadatas]
        # Use inverse loss (lower loss = higher weight)
        # Add small epsilon to avoid division by zero
        inv_losses = [1.0 / (loss + 1e-8) for loss in losses]
        total = sum(inv_losses)
        return [inv / total if total > 0 else 1.0 / len(metadatas) for inv in inv_losses]
    
    elif strategy == 'samples':
        samples = [m['samples'] for m in metadatas]
        total = sum(samples)
        return [s / total if total > 0 else 1.0 / len(metadatas) for s in samples]
    
    elif strategy == 'combined':
        # Combined: accuracy * samples / (loss + epsilon)
        scores = []
        for m in metadatas:
            score = (m['accuracy'] * m['samples']) / (m['loss'] + 1e-8)
            scores.append(score)
        total = sum(scores)
        return [s / total if total > 0 else 1.0 / len(metadatas) for s in scores]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def combine_models(
    model_paths: List[Path],
    strategy: str = 'combined',
    output_path: Path = None
) -> Path:
    """
    Combine multiple models into one super model
    
    Args:
        model_paths: List of paths to aggregated model files
        strategy: Weighting strategy ('equal', 'accuracy', 'loss', 'samples', 'combined')
        output_path: Path to save combined model (default: federated_models/super_model.pth)
    
    Returns:
        Path to saved combined model
    """
    print("🚀 Combining Multiple Models into Super Model")
    print("=" * 60)
    
    if not model_paths:
        raise ValueError("No models provided")
    
    # Load all models
    print(f"\n📥 Loading {len(model_paths)} models...")
    model_weights_list = []
    metadatas = []
    
    for i, model_path in enumerate(model_paths, 1):
        print(f"\n[{i}/{len(model_paths)}] Loading: {model_path.name}")
        weights, metadata = load_model_weights(model_path)
        model_weights_list.append(weights)
        metadatas.append(metadata)
        
        print(f"   Accuracy: {metadata['accuracy']:.2f}%")
        print(f"   Loss: {metadata['loss']:.4f}")
        print(f"   Samples: {metadata['samples']}")
        print(f"   Job: {metadata['job_name']}")
    
    # Calculate weight factors for each model
    print(f"\n⚖️  Calculating weight factors using strategy: '{strategy}'")
    weight_factors = calculate_model_weights(metadatas, strategy)
    
    print("\n📊 Model Weight Factors:")
    for i, (path, weight_factor, meta) in enumerate(zip(model_paths, weight_factors, metadatas)):
        print(f"   {path.name}: {weight_factor:.2%} (Acc: {meta['accuracy']:.2f}%, Loss: {meta['loss']:.4f})")
    
    # Verify all models have same architecture
    first_weights = model_weights_list[0]
    for i, model_weights in enumerate(model_weights_list[1:], 1):
        if set(model_weights.keys()) != set(first_weights.keys()):
            raise ValueError(f"Model {i+1} has different architecture than model 1")
        
        for key in first_weights.keys():
            if model_weights[key].shape != first_weights[key].shape:
                raise ValueError(f"Layer {key} has different shape in model {i+1}")
    
    # Aggregate weights
    print(f"\n🔄 Aggregating model weights...")
    aggregated_weights = {}
    
    # Initialize aggregated weights with correct shape and dtype
    for layer_name in first_weights.keys():
        first_weight = first_weights[layer_name]
        if isinstance(first_weight, np.ndarray):
            aggregated_weights[layer_name] = np.zeros_like(first_weight, dtype=np.float32)
        else:
            aggregated_weights[layer_name] = np.zeros_like(np.array(first_weight), dtype=np.float32)
    
    for model_idx, (model_weights, weight_factor) in enumerate(zip(model_weights_list, weight_factors), 1):
        print(f"   Processing model {model_idx}/{len(model_weights_list)} ({weight_factor:.2%})...")
        for layer_name, layer_weights in model_weights.items():
            if layer_name not in aggregated_weights:
                print(f"   ⚠️  Skipping {layer_name} (not in first model)")
                continue
            
            try:
                # Ensure layer_weights is numpy array and float32
                if isinstance(layer_weights, np.ndarray):
                    if not np.issubdtype(layer_weights.dtype, np.number):
                        print(f"   ⚠️  Skipping {layer_name} (non-numeric dtype: {layer_weights.dtype})")
                        continue
                    layer_weights_float = layer_weights.astype(np.float32)
                else:
                    layer_weights_float = np.array(layer_weights, dtype=np.float32)
                
                # Verify shapes match
                if aggregated_weights[layer_name].shape != layer_weights_float.shape:
                    print(f"   ⚠️  Shape mismatch for {layer_name}: {aggregated_weights[layer_name].shape} vs {layer_weights_float.shape}")
                    continue
                
                aggregated_weights[layer_name] += weight_factor * layer_weights_float
            except Exception as e:
                print(f"   ⚠️  Error processing {layer_name}: {e}")
                continue
    
    # Convert back to torch tensors
    aggregated_state_dict = {}
    for key, value in aggregated_weights.items():
        aggregated_state_dict[key] = torch.from_numpy(value)
    
    # Calculate combined metadata
    combined_accuracy = sum(w * m['accuracy'] for w, m in zip(weight_factors, metadatas))
    combined_loss = sum(w * m['loss'] for w, m in zip(weight_factors, metadatas))
    total_samples = sum(m['samples'] for m in metadatas)
    all_devices = []
    for m in metadatas:
        all_devices.extend(m.get('participating_devices', []))
    unique_devices = list(set(all_devices))
    
    # Use config from first model (they should be similar)
    combined_config = metadatas[0].get('config', {})
    
    # Save combined model
    if output_path is None:
        output_path = Path("federated_models") / "super_model_combined.pth"
    
    output_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        "model_state_dict": aggregated_state_dict,
        "job_id": "combined",
        "job_name": "Super Model (Combined)",
        "participating_devices": unique_devices,
        "total_samples": total_samples,
        "avg_accuracy": float(combined_accuracy),
        "avg_loss": float(combined_loss),
        "accuracy": float(combined_accuracy),  # Also save as 'accuracy' for compatibility
        "loss": float(combined_loss),  # Also save as 'loss' for compatibility
        "aggregation_strategy": strategy,
        "source_models": [str(p.name) for p in model_paths],
        "model_weight_factors": weight_factors,
        "config": combined_config,
        "timestamp": datetime.utcnow().isoformat()
    }, output_path)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n✅ Super model saved: {output_path}")
    print(f"   📦 File size: {file_size_mb:.2f} MB")
    print(f"   📊 Combined Accuracy: {combined_accuracy:.2f}%")
    print(f"   📊 Combined Loss: {combined_loss:.4f}")
    print(f"   📊 Total Samples: {total_samples}")
    print(f"   📱 Unique Devices: {len(unique_devices)}")
    
    return output_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine multiple aggregated models into one super model")
    parser.add_argument("--models", nargs="+", help="Paths to model files (default: all in federated_models/)")
    parser.add_argument("--strategy", default="combined", 
                       choices=['equal', 'accuracy', 'loss', 'samples', 'combined'],
                       help="Weighting strategy (default: combined)")
    parser.add_argument("--output", type=str, help="Output path (default: federated_models/super_model_combined.pth)")
    
    args = parser.parse_args()
    
    # Get model paths
    federated_models_dir = Path("federated_models")
    
    if args.models:
        model_paths = [Path(p) for p in args.models]
    else:
        # Use all aggregated models
        model_paths = sorted(federated_models_dir.glob("aggregated_model_*.pth"))
    
    if not model_paths:
        print("❌ No models found!")
        return
    
    # Convert to absolute paths
    model_paths = [p.resolve() if not p.is_absolute() else p for p in model_paths]
    
    # Verify all paths exist
    for path in model_paths:
        if not path.exists():
            print(f"❌ Model not found: {path}")
            return
    
    # Combine models
    output_path = Path(args.output) if args.output else None
    combine_models(model_paths, strategy=args.strategy, output_path=output_path)
    
    print("\n" + "=" * 60)
    print("🎉 Model combination complete!")
    print(f"\n💡 Test the super model with:")
    print(f"   python3 invoke_model.py --model {output_path or 'federated_models/super_model_combined.pth'} --text \"Your text here\"")

if __name__ == "__main__":
    main()

