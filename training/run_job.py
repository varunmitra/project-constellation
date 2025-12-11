#!/usr/bin/env python3
"""
Wrapper script to run a specific training job
Called from Swift app with job configuration
"""

import sys
import json
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.engine import ConstellationTrainer
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_job.py <config_json_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    # Read job configuration
    with open(config_file, 'r') as f:
        job_config = json.load(f)
    
    job_id = job_config['id']
    server_url = job_config.get('server_url', 'http://localhost:8000')
    device_id = job_config.get('device_id', 'swift-app')
    assignment_id = job_config.get('assignment_id', '')
    
    # Create trainer
    trainer = ConstellationTrainer(device_id=device_id, server_url=server_url)
    
    # Create job dict in expected format
    job = {
        'id': job_id,
        'name': job_config['name'],
        'model_type': job_config['model_type'],
        'dataset': job_config.get('dataset', 'synthetic'),
        'total_epochs': job_config['total_epochs'],
        'config': job_config.get('config', {})
    }
    
    logger.info(f"🚀 Starting training for job: {job['name']}")
    logger.info(f"📊 Model type: {job['model_type']}, Dataset: {job['dataset']}")
    logger.info(f"🔄 Total epochs: {job['total_epochs']}")
    
    # Execute training
    try:
        success = trainer.train(job, assignment_id)
        if success:
            logger.info("✅ Training completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ Training failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

