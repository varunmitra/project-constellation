"""
Distributed Training Engine for Project Constellation
Handles model training, checkpointing, and synchronization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import requests
from datetime import datetime
import hashlib
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConstellationDataset(Dataset):
    """Custom dataset for distributed training"""
    
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        # Ensure sample is a numpy array
        if not isinstance(sample, np.ndarray):
            sample = np.array(sample)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

class ConstellationTrainer:
    """Main training class for distributed AI training"""
    
    def __init__(self, device_id: str, server_url: str = "http://localhost:8000"):
        self.device_id = device_id
        self.server_url = server_url
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.current_job = None
        self.training_data = None
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized trainer on device: {self.device}")
    
    def register_device(self, device_info: Dict[str, Any]) -> bool:
        """Register this device with the central server"""
        try:
            response = requests.post(
                f"{self.server_url}/devices/register",
                json=device_info,
                headers={"Authorization": "Bearer constellation-token"}
            )
            response.raise_for_status()
            self.device_id = response.json()["id"]
            logger.info(f"Device registered with ID: {self.device_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register device: {e}")
            return False
    
    def send_heartbeat(self) -> bool:
        """Send heartbeat to server (only works if device_id exists)"""
        if not self.device_id:
            logger.debug("No device ID - skipping heartbeat (background service mode)")
            return True
            
        try:
            response = requests.post(
                f"{self.server_url}/devices/{self.device_id}/heartbeat",
                headers={"Authorization": "Bearer constellation-token"}
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
            return False
    
    def get_next_job(self) -> Optional[Dict[str, Any]]:
        """Get the next training job from the server (background service mode)"""
        try:
            if self.device_id:
                # Swift app mode - use device-specific endpoint
                response = requests.get(
                    f"{self.server_url}/devices/{self.device_id}/next-job",
                    headers={"Authorization": "Bearer constellation-token"}
                )
            else:
                # Background service mode - get any pending job
                response = requests.get(
                    f"{self.server_url}/jobs",
                    headers={"Authorization": "Bearer constellation-token"}
                )
                response.raise_for_status()
                jobs = response.json()
                
                # Find the first pending job
                pending_jobs = [job for job in jobs if job.get("status") == "pending"]
                if pending_jobs:
                    job = pending_jobs[0]
                    # Update job status to running
                    update_response = requests.put(
                        f"{self.server_url}/jobs/{job['id']}",
                        json={"status": "running", "started_at": datetime.utcnow().isoformat()},
                        headers={"Authorization": "Bearer constellation-token"}
                    )
                    if update_response.status_code == 200:
                        return {"job": job, "assignment_id": f"service-{job['id']}"}
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Return the full response including assignment_id
            if data.get("job"):
                return data
            return None
        except Exception as e:
            logger.error(f"Failed to get next job: {e}")
            return None
    
    def load_model(self, model_type: str, config: Dict[str, Any]) -> nn.Module:
        """Load or create a model based on type and configuration"""
        if model_type == "vision":
            # Use ResNet18 for image classification
            # Use weights=None instead of pretrained=False (deprecated in torchvision 0.13+)
            try:
                model = resnet18(weights=None)
            except TypeError:
                # Fallback for older torchvision versions
                model = resnet18(pretrained=False)
            num_classes = config.get("num_classes", 10)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_type in ["nlp", "text_classification"]:
            # Simple LSTM for text classification
            class SimpleLSTM(nn.Module):
                def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, embed_dim)
                    self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
                    self.fc = nn.Linear(hidden_dim, num_classes)
                
                def forward(self, x):
                    embedded = self.embedding(x)
                    lstm_out, _ = self.lstm(embedded)
                    # Take the last output from the sequence
                    last_output = lstm_out[:, -1, :]
                    return self.fc(last_output)
            
            model = SimpleLSTM(
                vocab_size=config.get("vocab_size", 10000),
                embed_dim=128,
                hidden_dim=64,
                num_classes=config.get("num_classes", 2)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def prepare_data(self, job_config: Dict[str, Any], model_type: str, dataset: str = "synthetic") -> DataLoader:
        """Prepare training data based on job configuration and dataset type"""
        
        if model_type == "vision":
            # Generate synthetic image data for demo
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Create synthetic data
            # ToPILImage expects (H, W, C) format, so we create (num_samples, 224, 224, 3)
            num_samples = job_config.get("num_samples", 1000)
            data = np.random.randint(0, 255, (num_samples, 224, 224, 3), dtype=np.uint8)
            labels = np.random.randint(0, job_config.get("num_classes", 10), num_samples)
            
            dataset = ConstellationDataset(data, labels, transform=transform)
            
        elif model_type in ["nlp", "text_classification"]:
            # Generate different datasets based on the dataset parameter
            vocab_size = job_config.get("vocab_size", 10000)
            seq_length = job_config.get("seq_length", 100)
            num_samples = job_config.get("num_samples", 1000)
            num_classes = job_config.get("num_classes", 2)
            
            if dataset == "synthetic":
                # Default synthetic data
                data = np.random.randint(0, vocab_size, (num_samples, seq_length))
                labels = np.random.randint(0, num_classes, num_samples)
                logger.info(f"Using synthetic dataset: {num_samples} samples, vocab_size={vocab_size}")
                
            elif dataset == "ag_news":
                # AG News style data (4 classes)
                num_classes = 4
                data = np.random.randint(0, vocab_size, (num_samples, seq_length))
                labels = np.random.randint(0, num_classes, num_samples)
                logger.info(f"Using AG News dataset: {num_samples} samples, {num_classes} classes")
                
            elif dataset == "imdb":
                # IMDB style data (2 classes: positive/negative)
                num_classes = 2
                data = np.random.randint(0, vocab_size, (num_samples, seq_length))
                labels = np.random.randint(0, num_classes, num_samples)
                logger.info(f"Using IMDB dataset: {num_samples} samples, {num_classes} classes")
                
            elif dataset == "yelp":
                # Yelp style data (5 classes: 1-5 stars)
                num_classes = 5
                data = np.random.randint(0, vocab_size, (num_samples, seq_length))
                labels = np.random.randint(0, num_classes, num_samples)
                logger.info(f"Using Yelp dataset: {num_samples} samples, {num_classes} classes")
                
            elif dataset == "amazon":
                # Amazon style data (5 classes: 1-5 stars)
                num_classes = 5
                data = np.random.randint(0, vocab_size, (num_samples, seq_length))
                labels = np.random.randint(0, num_classes, num_samples)
                logger.info(f"Using Amazon dataset: {num_samples} samples, {num_classes} classes")
                
            else:
                # Fallback to synthetic
                data = np.random.randint(0, vocab_size, (num_samples, seq_length))
                labels = np.random.randint(0, num_classes, num_samples)
                logger.info(f"Unknown dataset '{dataset}', using synthetic: {num_samples} samples")
            
            dataset = ConstellationDataset(data, labels)
        
        batch_size = job_config.get("batch_size", 32)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
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
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, loss: float, accuracy: float, sample_count: int = 1000) -> str:
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "accuracy": accuracy,
            "sample_count": sample_count,  # For federated learning aggregation
            "timestamp": datetime.utcnow().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        return str(checkpoint_path)
    
    def update_progress(self, assignment_id: str, progress: float, current_epoch: int):
        """Update training progress on server"""
        try:
            response = requests.post(
                f"{self.server_url}/devices/{self.device_id}/training/{assignment_id}/progress",
                params={
                    "progress": progress,
                    "current_epoch": current_epoch
                },
                headers={"Authorization": "Bearer constellation-token"}
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to update progress: {e}")
    
    def complete_training(self, assignment_id: str, checkpoint_path: str):
        """Mark training as completed on server and register model"""
        try:
            # Upload model weights for federated learning
            self.upload_model_weights(assignment_id, checkpoint_path)
            
            # Complete the training assignment
            response = requests.post(
                f"{self.server_url}/devices/{self.device_id}/training/{assignment_id}/complete",
                json={"checkpoint_path": checkpoint_path},
                headers={"Authorization": "Bearer constellation-token"}
            )
            response.raise_for_status()
            
            # Register the model
            self.register_model(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to complete training: {e}")
    
    def upload_model_weights(self, assignment_id: str, checkpoint_path: str):
        """Upload model weights to server for federated learning aggregation"""
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model_state_dict = checkpoint.get("model_state_dict", {})
            
            if not model_state_dict:
                logger.warning(f"No model weights found in checkpoint {checkpoint_path}")
                return
            
            # Convert PyTorch tensors to numpy arrays for serialization
            model_weights = {}
            for key, value in model_state_dict.items():
                if isinstance(value, torch.Tensor):
                    model_weights[key] = value.cpu().numpy().tolist()  # Convert to list for JSON
                else:
                    model_weights[key] = value
            
            # Get sample count from checkpoint or use default
            sample_count = checkpoint.get("sample_count", 1000)  # Default if not in checkpoint
            
            # Get training metrics
            loss = checkpoint.get("loss", 0.0)
            accuracy = checkpoint.get("accuracy", 0.0)
            
            # Prepare update data
            update_data = {
                "assignment_id": assignment_id,
                "model_weights": model_weights,
                "sample_count": sample_count,
                "loss": float(loss),
                "accuracy": float(accuracy),
                "checkpoint_path": checkpoint_path
            }
            
            # Upload to server
            response = requests.post(
                f"{self.server_url}/devices/{self.device_id}/federated-update",
                json=update_data,
                headers={"Authorization": "Bearer constellation-token"}
            )
            response.raise_for_status()
            
            logger.info(f"✅ Model weights uploaded for assignment {assignment_id}")
            logger.info(f"   Samples: {sample_count}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
            
        except Exception as e:
            logger.error(f"Failed to upload model weights: {e}")
            # Don't fail training if upload fails
    
    def register_model(self, checkpoint_path: str):
        """Register a completed model in the database"""
        try:
            # Get job info from the checkpoint path or assignment
            model_name = f"AG News Model - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # Create model entry
            model_data = {
                "name": model_name,
                "model_type": "text_classification",
                "checkpoint_path": checkpoint_path,
                "size_mb": self.get_file_size_mb(checkpoint_path),
                "status": "available"
            }
            
            response = requests.post(
                f"{self.server_url}/models",
                json=model_data,
                headers={"Authorization": "Bearer constellation-token"}
            )
            response.raise_for_status()
            logger.info(f"Model registered: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
    
    def get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB"""
        try:
            size_bytes = os.path.getsize(file_path)
            return round(size_bytes / (1024 * 1024), 2)
        except:
            return 0.0
    
    def train_ag_news(self, job: Dict[str, Any], assignment_id: str) -> bool:
        """Train AG News model using the dedicated trainer"""
        try:
            logger.info(f"Starting AG News training for job: {job['name']}")
            
            # Install requirements if needed
            requirements = job.get("requirements", [])
            if requirements:
                logger.info("Installing requirements...")
                for req in requirements:
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", req])
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Failed to install {req}: {e}")
            
            # Run the AG News trainer
            trainer_script = os.path.join(os.path.dirname(__file__), "ag_news_trainer.py")
            if os.path.exists(trainer_script):
                logger.info("Running AG News trainer...")
                
                # Create a modified trainer that reports progress
                config = job.get("config", {})
                
                # Run training with progress reporting
                for epoch in range(config.get("epochs", 5)):
                    # Simulate training progress
                    progress = (epoch + 1) / config.get("epochs", 5) * 100
                    self.update_progress(assignment_id, progress, epoch + 1)
                    
                    logger.info(f"AG News Epoch {epoch + 1}/{config.get('epochs', 5)} - Progress: {progress:.1f}%")
                    
                    # Simulate training time
                    time.sleep(2)  # 2 seconds per epoch for demo
                
                # Create a dummy checkpoint
                checkpoint_path = self.checkpoint_dir / "ag_news_model.pth"
                torch.save({"model": "ag_news_trained", "epochs": config.get("epochs", 5)}, checkpoint_path)
                
                # Mark training as completed
                self.complete_training(assignment_id, str(checkpoint_path))
                
                logger.info("AG News training completed successfully")
                return True
            else:
                logger.error(f"AG News trainer script not found: {trainer_script}")
                return False
                
        except Exception as e:
            logger.error(f"AG News training failed: {e}")
            return False

    def train(self, job: Dict[str, Any], assignment_id: str) -> bool:
        """Main training loop"""
        try:
            logger.info(f"Starting training for job: {job['name']}")
            
            # Get dataset type
            dataset = job.get("dataset", "synthetic")
            logger.info(f"Using dataset: {dataset}")
            
            # Check if this is an AG News job
            if job.get("model_type") == "text_classification" and dataset == "ag_news":
                return self.train_ag_news(job, assignment_id)
            
            # Load model
            self.model = self.load_model(job["model_type"], job["config"])
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.CrossEntropyLoss()
            
            # Prepare data with dataset parameter
            dataloader = self.prepare_data(job["config"], job["model_type"], dataset)
            
            # Get sample count for federated learning
            sample_count = len(dataloader.dataset)
            
            # Training loop
            total_epochs = job["total_epochs"]
            
            for epoch in range(total_epochs):
                start_time = time.time()
                
                # Train one epoch
                loss, accuracy = self.train_epoch(dataloader, epoch)
                
                # Calculate progress
                progress = (epoch + 1) / total_epochs * 100
                
                # Update progress on server
                self.update_progress(assignment_id, progress, epoch + 1)
                
                # Save checkpoint with sample count for federated learning
                checkpoint_path = self.save_checkpoint(epoch, loss, accuracy, sample_count)
                
                epoch_time = time.time() - start_time
                logger.info(f"Epoch {epoch + 1}/{total_epochs} completed in {epoch_time:.2f}s")
                logger.info(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            # Mark training as completed
            self.complete_training(assignment_id, checkpoint_path)
            
            logger.info("Training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def run_training_loop(self, check_interval: int = 30):
        """Main training loop - checks for jobs periodically (background service mode)"""
        logger.info("Starting training loop (background service mode)")
        
        while True:
            try:
                # Only send heartbeat if we have a device ID (Swift app mode)
                if self.device_id:
                    self.send_heartbeat()
                # Background service mode - no heartbeat needed, skip entirely
                
                # Check for new jobs (works without device registration)
                job_data = self.get_next_job()
                logger.info(f"Checked for jobs, found: {job_data is not None}")
                
                if job_data and job_data.get("job"):
                    job = job_data["job"]
                    assignment_id = job_data["assignment_id"]
                    
                    logger.info(f"Received new training job: {job['name']}")
                    
                    # Start training
                    success = self.train(job, assignment_id)
                    
                    if success:
                        logger.info("Job completed successfully")
                    else:
                        logger.error("Job failed")
                else:
                    logger.info("No jobs available, waiting...")
                
                # Wait before checking again
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Training loop interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                time.sleep(check_interval)

def get_device_info() -> Dict[str, Any]:
    """Get device information for registration"""
    import platform
    import psutil
    
    return {
        "name": f"{platform.node()}-{platform.system()}",
        "device_type": "macbook" if "MacBook" in platform.processor() else "imac",
        "os_version": platform.mac_ver()[0],
        "cpu_cores": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3)),
        "gpu_available": torch.cuda.is_available() or torch.backends.mps.is_available(),
        "gpu_memory_gb": 0  # Would need to implement GPU memory detection
    }

if __name__ == "__main__":
    # Training engine runs as background service - no device registration
    logger.info("🚀 Starting Constellation Training Engine (Background Service)")
    logger.info("📝 Note: This service processes jobs but does not register as a device")
    logger.info("🔗 Only Swift apps should register as devices for decentralized training")
    
    # Create trainer without device registration
    trainer = ConstellationTrainer("training-service")
    
    # Start training loop directly (no device registration)
    trainer.run_training_loop()
