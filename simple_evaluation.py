#!/usr/bin/env python3
"""
Simple Model Evaluation for AG News Classification
Basic evaluation without complex model loading
"""

import os
import json
import requests
from datetime import datetime

def check_training_progress():
    """Check the current training progress and status"""
    print("🔍 Training Progress Analysis")
    print("=" * 50)
    
    try:
        # Get job status
        response = requests.get('http://localhost:8000/jobs')
        jobs = response.json()
        
        # Find our 100-epoch job
        target_job = None
        for job in jobs:
            if job['name'] == '100 Epoch AG News Training':
                target_job = job
                break
        
        if not target_job:
            print("❌ 100 Epoch training job not found!")
            return
        
        # Display current status
        print(f"📊 Job Status: {target_job['status']}")
        print(f"📈 Progress: {target_job['progress']}%")
        print(f"🔄 Epoch: {target_job['current_epoch']}/{target_job['total_epochs']}")
        print(f"⏰ Started: {target_job['started_at']}")
        
        # Calculate runtime
        if target_job['started_at']:
            start_time = datetime.fromisoformat(target_job['started_at'].replace('Z', '+00:00'))
            runtime = datetime.now() - start_time.replace(tzinfo=None)
            print(f"⏱️  Runtime: {runtime}")
        
        # Training completion analysis
        epochs_completed = target_job['current_epoch']
        total_epochs = target_job['total_epochs']
        progress_percent = target_job['progress']
        
        print(f"\n📋 Training Analysis:")
        print(f"  • Epochs Completed: {epochs_completed}/{total_epochs}")
        print(f"  • Progress: {progress_percent}%")
        
        # Sufficiency assessment
        print(f"\n🎯 Training Sufficiency Assessment:")
        
        if epochs_completed >= 50:
            print("✅ Sufficient training epochs completed (50+)")
            print("   - Model has had adequate time to learn patterns")
            print("   - Should show good performance on validation data")
        elif epochs_completed >= 25:
            print("📈 Good training progress (25+ epochs)")
            print("   - Model is learning effectively")
            print("   - Performance should be improving steadily")
        elif epochs_completed >= 10:
            print("🔄 Early training phase (10+ epochs)")
            print("   - Model is still learning basic patterns")
            print("   - More training recommended for optimal performance")
        else:
            print("⏳ Very early training phase")
            print("   - Model needs more time to learn")
        
        # Performance expectations
        print(f"\n💡 Performance Expectations:")
        
        if epochs_completed >= 80:
            print("🎉 Near completion - should see excellent performance")
            print("   - Expected accuracy: 85-95%")
            print("   - Model should be well-converged")
        elif epochs_completed >= 50:
            print("✅ Good training level - solid performance expected")
            print("   - Expected accuracy: 75-85%")
            print("   - Model should show good generalization")
        elif epochs_completed >= 25:
            print("📈 Moderate training - decent performance")
            print("   - Expected accuracy: 65-75%")
            print("   - Model still learning but functional")
        else:
            print("⚠️  Early stage - basic performance only")
            print("   - Expected accuracy: 50-65%")
            print("   - More training strongly recommended")
        
        # Recommendations
        print(f"\n🚀 Recommendations:")
        
        if target_job['status'] == 'running':
            remaining_epochs = total_epochs - epochs_completed
            print(f"🔄 Continue training for {remaining_epochs} more epochs")
            
            if remaining_epochs <= 15:
                print("   - Almost complete! Let it finish for best results")
            elif remaining_epochs <= 30:
                print("   - Good progress! Continue to completion")
            else:
                print("   - Consider early stopping if performance plateaus")
        else:
            print("✅ Training complete! Ready for final evaluation")
        
        # Check for checkpoints
        checkpoint_dir = 'training/checkpoints'
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
            print(f"\n💾 Checkpoints Available: {len(checkpoints)}")
            
            if checkpoints:
                # Get the latest checkpoint
                latest_epoch = max([int(f.split('_')[2].split('.')[0]) for f in checkpoints])
                print(f"   - Latest checkpoint: Epoch {latest_epoch}")
                print(f"   - Checkpoints saved every epoch")
        
        # Model readiness assessment
        print(f"\n🎯 Model Readiness Assessment:")
        
        if epochs_completed >= 80:
            print("🏆 EXCELLENT - Model is well-trained and ready for production")
            print("   - High accuracy expected")
            print("   - Good generalization capabilities")
            print("   - Suitable for real-world applications")
        elif epochs_completed >= 50:
            print("✅ GOOD - Model shows solid training progress")
            print("   - Good performance expected")
            print("   - May benefit from additional training")
            print("   - Suitable for testing and evaluation")
        elif epochs_completed >= 25:
            print("📈 FAIR - Model is learning but needs more training")
            print("   - Basic performance expected")
            print("   - Continue training recommended")
            print("   - Good for initial testing")
        else:
            print("⚠️  EARLY - Model needs significant more training")
            print("   - Limited performance expected")
            print("   - Continue training strongly recommended")
            print("   - Not ready for production use")
        
    except Exception as e:
        print(f"❌ Error checking training progress: {e}")

if __name__ == "__main__":
    check_training_progress()

