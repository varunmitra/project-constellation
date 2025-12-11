#!/usr/bin/env python3
"""
Create a sentiment analysis training job
This demonstrates a more descriptive model with clear, interpretable results
"""

import requests
import json

SERVER_URL = "http://localhost:8000"
AUTH_HEADER = {"Authorization": "Bearer constellation-token"}

def create_sentiment_job():
    """Create a sentiment analysis training job"""
    
    job_data = {
        "name": "Sentiment Analysis Demo - Customer Feedback",
        "model_type": "text_classification",  # Reuse text classification architecture
        "dataset": "synthetic",
        "total_epochs": 30,  # Fewer epochs for faster training
        "config": {
            "vocab_size": 10000,
            "seq_length": 100,
            "num_samples": 1500,
            "num_classes": 3,  # Positive, Negative, Neutral
            "batch_size": 32,
            "learning_rate": 0.001,
            "description": "Sentiment Analysis - Classifies text as Positive, Negative, or Neutral"
        }
    }
    
    try:
        response = requests.post(
            f"{SERVER_URL}/jobs",
            json=job_data,
            headers=AUTH_HEADER
        )
        response.raise_for_status()
        job = response.json()
        
        print("✅ Sentiment Analysis Job Created!")
        print(f"   Job ID: {job['id']}")
        print(f"   Name: {job['name']}")
        print(f"   Epochs: {job['total_epochs']}")
        print(f"   Classes: {job['config']['num_classes']} (Positive, Negative, Neutral)")
        print(f"   Status: {job['status']}")
        print()
        print("📊 Why This Model is Better:")
        print("   ✅ More descriptive results (sentiment scores)")
        print("   ✅ Practical use case (customer feedback analysis)")
        print("   ✅ Easy to understand (positive/negative/neutral)")
        print("   ✅ Shows confidence levels clearly")
        print()
        print("🚀 Next Steps:")
        print("   1. Swift app will automatically pick up the job")
        print("   2. Training will start automatically")
        print("   3. After training, test with:")
        print("      python3 test_sentiment.py")
        
        return job['id']
    except Exception as e:
        print(f"❌ Failed to create job: {e}")
        return None

if __name__ == "__main__":
    create_sentiment_job()

