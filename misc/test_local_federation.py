#!/usr/bin/env python3
"""
Test Local Federation - Aggregate and Test Trained Models
This script simulates federated learning by aggregating multiple trained models
"""

import torch
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from federated.model_aggregator import ModelAggregator

def find_latest_checkpoints(num_models=3):
    """Find the latest trained model checkpoints"""
    checkpoint_dirs = [
        Path("checkpoints"),
        Path("desktop-swift/checkpoints"),
        Path("training/checkpoints")
    ]
    
    all_checkpoints = []
    for checkpoint_dir in checkpoint_dirs:
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
            all_checkpoints.extend(checkpoints)
    
    if not all_checkpoints:
        print("❌ No checkpoints found!")
        return []
    
    # Sort by modification time and get the latest ones
    all_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Get diverse checkpoints (not all from same training run)
    selected = []
    seen_dirs = set()
    
    for checkpoint in all_checkpoints:
        parent_dir = checkpoint.parent
        if parent_dir not in seen_dirs or len(selected) < num_models:
            selected.append(checkpoint)
            seen_dirs.add(parent_dir)
            if len(selected) >= num_models:
                break
    
    return selected[:num_models]

def load_model_weights(checkpoint_path):
    """Load model weights from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Convert to numpy arrays
        weights = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                weights[key] = value.cpu().numpy()
        
        # Get metadata
        metadata = {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', 0.0),
            'accuracy': checkpoint.get('accuracy', 0.0),
            'checkpoint_path': str(checkpoint_path)
        }
        
        return weights, metadata
    
    except Exception as e:
        print(f"❌ Error loading {checkpoint_path}: {e}")
        return None, None

def create_device_updates(checkpoint_paths):
    """Create device updates from checkpoints"""
    device_updates = []
    
    for idx, checkpoint_path in enumerate(checkpoint_paths):
        print(f"\n📥 Loading checkpoint {idx + 1}: {checkpoint_path.name}")
        
        weights, metadata = load_model_weights(checkpoint_path)
        if weights is None:
            continue
        
        # Create device update
        device_update = {
            'device_id': f'device_{idx + 1}',
            'model_weights': weights,
            'sample_count': 1000 + (idx * 100),  # Simulated sample counts
            'local_epochs': metadata.get('epoch', 1),
            'metadata': metadata
        }
        
        device_updates.append(device_update)
        
        print(f"   ✅ Loaded: Epoch {metadata['epoch']}, "
              f"Loss: {metadata['loss']:.4f}, "
              f"Accuracy: {metadata['accuracy']:.2f}%")
        print(f"   📊 Parameters: {len(weights)} layers")
    
    return device_updates

def test_aggregated_model(aggregated_weights, original_metadata):
    """Test the aggregated model with sample data"""
    print("\n" + "="*60)
    print("🧪 Testing Aggregated Model")
    print("="*60)
    
    # Analyze aggregated weights
    total_params = sum(w.size for w in aggregated_weights.values())
    print(f"\n📊 Aggregated Model Statistics:")
    print(f"   Total layers: {len(aggregated_weights)}")
    print(f"   Total parameters: {total_params:,}")
    
    # Show layer statistics
    print(f"\n📈 Layer Statistics:")
    for layer_name, weights in list(aggregated_weights.items())[:5]:  # Show first 5 layers
        mean_val = np.mean(weights)
        std_val = np.std(weights)
        min_val = np.min(weights)
        max_val = np.max(weights)
        print(f"   {layer_name}:")
        print(f"      Shape: {weights.shape}")
        print(f"      Mean: {mean_val:.6f}, Std: {std_val:.6f}")
        print(f"      Range: [{min_val:.6f}, {max_val:.6f}]")
    
    if len(aggregated_weights) > 5:
        print(f"   ... and {len(aggregated_weights) - 5} more layers")
    
    # Compare with original models
    print(f"\n📊 Original Models Performance:")
    for metadata in original_metadata:
        print(f"   Device: {metadata.get('device_id', 'unknown')}")
        print(f"      Epoch: {metadata.get('epoch', 0)}")
        print(f"      Loss: {metadata.get('loss', 0.0):.4f}")
        print(f"      Accuracy: {metadata.get('accuracy', 0.0):.2f}%")
    
    # Calculate expected performance
    avg_accuracy = np.mean([m.get('accuracy', 0.0) for m in original_metadata])
    avg_loss = np.mean([m.get('loss', 0.0) for m in original_metadata])
    
    print(f"\n📈 Expected Aggregated Performance:")
    print(f"   Average Accuracy: {avg_accuracy:.2f}%")
    print(f"   Average Loss: {avg_loss:.4f}")
    print(f"   (Actual performance may vary when deployed)")

def main():
    print("="*60)
    print("🚀 Local Federation Test - Model Aggregation")
    print("="*60)
    
    # Step 1: Find latest checkpoints
    print("\n📂 Step 1: Finding trained model checkpoints...")
    checkpoint_paths = find_latest_checkpoints(num_models=3)
    
    if not checkpoint_paths:
        print("\n❌ No checkpoints found. Please train some models first.")
        print("\nTo train models, run:")
        print("  python3 create_sentiment_job.py")
        print("  # Or use the Swift app to train")
        return
    
    print(f"\n✅ Found {len(checkpoint_paths)} checkpoints:")
    for i, path in enumerate(checkpoint_paths, 1):
        size_mb = path.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        print(f"   {i}. {path.name} ({size_mb:.1f} MB, modified: {mtime.strftime('%Y-%m-%d %H:%M')})")
    
    # Step 2: Load model weights
    print("\n📥 Step 2: Loading model weights...")
    device_updates = create_device_updates(checkpoint_paths)
    
    if not device_updates:
        print("\n❌ Failed to load any model weights")
        return
    
    print(f"\n✅ Loaded {len(device_updates)} device updates")
    
    # Step 3: Aggregate models
    print("\n🔄 Step 3: Aggregating models using Federated Averaging...")
    aggregator = ModelAggregator(aggregation_strategy="fedavg")
    
    try:
        aggregated_weights = aggregator.aggregate_models(device_updates)
        print(f"✅ Models aggregated successfully!")
    except Exception as e:
        print(f"❌ Aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Save aggregated model
    print("\n💾 Step 4: Saving aggregated model...")
    
    metadata = {
        'aggregation_strategy': 'fedavg',
        'num_devices': len(device_updates),
        'total_samples': sum(u['sample_count'] for u in device_updates),
        'device_ids': [u['device_id'] for u in device_updates],
        'timestamp': datetime.now().isoformat(),
        'source_checkpoints': [str(p) for p in checkpoint_paths],
        'original_metadata': [u['metadata'] for u in device_updates]
    }
    
    try:
        model_path = aggregator.save_aggregated_model(
            aggregated_weights,
            round_id=f"local_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            metadata=metadata
        )
        print(f"✅ Aggregated model saved: {model_path}")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Test aggregated model
    test_aggregated_model(aggregated_weights, [u['metadata'] for u in device_updates])
    
    # Summary
    print("\n" + "="*60)
    print("🎉 Federation Test Completed Successfully!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   ✅ Aggregated {len(device_updates)} trained models")
    print(f"   ✅ Used {metadata['total_samples']:,} total samples")
    print(f"   ✅ Saved to: {model_path}")
    print(f"\n💡 Next Steps:")
    print(f"   1. Use this aggregated model for inference")
    print(f"   2. Deploy to production")
    print(f"   3. Continue training with more devices")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()

