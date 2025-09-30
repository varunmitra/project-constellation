"""
Central Server for Project Constellation
Handles model distribution, device coordination, and training job management
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import uuid
import json
import os
from pathlib import Path

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./constellation.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class Device(Base):
    __tablename__ = "devices"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    device_type = Column(String, nullable=False)  # macbook, imac, mac_studio
    os_version = Column(String, nullable=False)
    cpu_cores = Column(Integer, nullable=False)
    memory_gb = Column(Integer, nullable=False)
    gpu_available = Column(Boolean, default=False)
    gpu_memory_gb = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    last_seen = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

class TrainingJob(Base):
    __tablename__ = "training_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    model_name = Column(String, nullable=True)  # Name of the specific model
    model_type = Column(String, nullable=False)  # vision, nlp, etc.
    dataset = Column(String, default="synthetic")  # synthetic, ag_news, imdb, yelp, amazon
    status = Column(String, default="pending")  # pending, running, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    total_epochs = Column(Integer, default=10)
    current_epoch = Column(Integer, default=0)
    progress = Column(Float, default=0.0)
    config = Column(Text)  # JSON config for the training job

class DeviceTraining(Base):
    __tablename__ = "device_training"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    device_id = Column(String, nullable=False)
    job_id = Column(String, nullable=False)
    status = Column(String, default="assigned")  # assigned, running, completed, failed
    assigned_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    progress = Column(Float, default=0.0)
    checkpoint_path = Column(String, nullable=True)

class Model(Base):
    __tablename__ = "models"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # text_classification, vision, etc.
    checkpoint_path = Column(String, nullable=False)
    size_mb = Column(Float, default=0.0)
    status = Column(String, default="available")  # available, training, archived
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class DeviceCreate(BaseModel):
    name: str
    device_type: str
    os_version: str
    cpu_cores: int
    memory_gb: int
    gpu_available: bool = False
    gpu_memory_gb: int = 0

class DeviceResponse(BaseModel):
    id: str
    name: str
    device_type: str
    os_version: str
    cpu_cores: int
    memory_gb: int
    gpu_available: bool
    gpu_memory_gb: int
    is_active: bool
    last_seen: datetime
    created_at: datetime

class TrainingJobCreate(BaseModel):
    name: str
    model_name: str = ""
    model_type: str
    dataset: str = "synthetic"  # synthetic, ag_news, imdb, yelp, amazon
    total_epochs: int = 10
    config: dict

class TrainingJobResponse(BaseModel):
    id: str
    name: str
    model_name: Optional[str] = ""
    model_type: str
    dataset: str = "synthetic"
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    total_epochs: int
    current_epoch: int
    progress: float
    config: dict

# FastAPI app
app = FastAPI(title="Project Constellation Server", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # In production, implement proper JWT verification
    if credentials.credentials != "constellation-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# API Endpoints

@app.get("/")
async def root():
    return {"message": "Project Constellation Server", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Device Management
@app.post("/devices/register", response_model=DeviceResponse)
async def register_device(device: DeviceCreate, db: Session = Depends(get_db)):
    # Check if device already exists by name and device_type
    existing_device = db.query(Device).filter(
        Device.name == device.name,
        Device.device_type == device.device_type
    ).first()
    
    if existing_device:
        # Update existing device instead of creating new one
        existing_device.is_active = True
        existing_device.last_seen = datetime.utcnow()
        existing_device.os_version = device.os_version
        existing_device.cpu_cores = device.cpu_cores
        existing_device.memory_gb = device.memory_gb
        existing_device.gpu_available = device.gpu_available
        existing_device.gpu_memory_gb = device.gpu_memory_gb
        db.commit()
        db.refresh(existing_device)
        return existing_device
    
    # Create new device if it doesn't exist
    db_device = Device(**device.dict())
    db.add(db_device)
    db.commit()
    db.refresh(db_device)
    return db_device

@app.get("/devices", response_model=List[DeviceResponse])
async def list_devices(db: Session = Depends(get_db)):
    devices = db.query(Device).filter(Device.is_active == True).all()
    return devices

@app.post("/devices/cleanup")
async def cleanup_duplicate_devices(db: Session = Depends(get_db)):
    """Remove duplicate devices, keeping the most recent one for each name+type combination"""
    # Get all devices grouped by name and device_type
    devices = db.query(Device).all()
    
    # Group by name and device_type
    device_groups = {}
    for device in devices:
        key = (device.name, device.device_type)
        if key not in device_groups:
            device_groups[key] = []
        device_groups[key].append(device)
    
    removed_count = 0
    for key, device_list in device_groups.items():
        if len(device_list) > 1:
            # Keep the most recent device, remove others
            device_list.sort(key=lambda x: x.last_seen, reverse=True)
            keep_device = device_list[0]
            
            for device in device_list[1:]:
                db.delete(device)
                removed_count += 1
    
    db.commit()
    return {"status": "cleanup completed", "removed_devices": removed_count}

@app.get("/devices/{device_id}", response_model=DeviceResponse)
async def get_device(device_id: str, db: Session = Depends(get_db)):
    device = db.query(Device).filter(Device.id == device_id).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return device

@app.post("/devices/{device_id}/heartbeat")
async def device_heartbeat(device_id: str, db: Session = Depends(get_db)):
    device = db.query(Device).filter(Device.id == device_id).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    device.last_seen = datetime.utcnow()
    device.is_active = True
    db.commit()
    return {"status": "heartbeat received"}

# Training Job Management
@app.post("/jobs", response_model=TrainingJobResponse)
async def create_training_job(job: TrainingJobCreate, db: Session = Depends(get_db)):
    db_job = TrainingJob(
        name=job.name,
        model_name=job.model_name,
        model_type=job.model_type,
        dataset=job.dataset,
        total_epochs=job.total_epochs,
        config=json.dumps(job.config)
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    
    # Parse config back to dict for response
    db_job.config = json.loads(db_job.config)
    return db_job

@app.get("/jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(db: Session = Depends(get_db)):
    jobs = db.query(TrainingJob).all()
    for job in jobs:
        job.config = json.loads(job.config)
    return jobs

@app.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    job.config = json.loads(job.config)
    return job

@app.post("/jobs/{job_id}/start")
async def start_training_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if job.status != "pending":
        raise HTTPException(status_code=400, detail="Job is not in pending status")
    
    job.status = "running"
    job.started_at = datetime.utcnow()
    db.commit()
    
    # TODO: Implement job distribution logic
    return {"status": "job started", "job_id": job_id}

@app.post("/jobs/{job_id}/complete")
async def complete_training_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job.status = "completed"
    job.completed_at = datetime.utcnow()
    job.progress = 100.0
    db.commit()
    
    return {"status": "job completed", "job_id": job_id}

@app.post("/jobs/fix-completed")
async def fix_completed_jobs(db: Session = Depends(get_db)):
    """Fix jobs that are at 100% but still show as running"""
    jobs = db.query(TrainingJob).filter(
        TrainingJob.status == "running",
        TrainingJob.progress >= 100.0
    ).all()
    
    fixed_count = 0
    for job in jobs:
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        fixed_count += 1
    
    db.commit()
    return {"status": f"Fixed {fixed_count} completed jobs"}

# Device Training Assignment
@app.get("/devices/{device_id}/next-job")
async def get_next_training_job(device_id: str, db: Session = Depends(get_db)):
    # Find available training jobs (pending or running, exclude completed jobs)
    available_jobs = db.query(TrainingJob).filter(
        TrainingJob.status.in_(["pending", "running"])
    ).all()
    
    # Filter out jobs that are at 100% progress (should be completed)
    available_jobs = [job for job in available_jobs if job.progress < 100.0]
    
    if not available_jobs:
        return {"job": None, "message": "No available training jobs"}
    
    # Prioritize jobs that haven't been started (progress = 0) or are not completed
    # Sort by progress ascending, then by creation time
    available_jobs.sort(key=lambda x: (x.progress, x.created_at))
    
    # Find a job that's not already assigned to this device
    for job in available_jobs:
        existing_assignment = db.query(DeviceTraining).filter(
            DeviceTraining.device_id == device_id,
            DeviceTraining.job_id == job.id,
            DeviceTraining.status.in_(["assigned", "running"])
        ).first()
        
        if not existing_assignment:
            break
    else:
        # If all jobs are assigned to this device, use the first one
        job = available_jobs[0]
    
    # Create device training assignment
    device_training = DeviceTraining(
        device_id=device_id,
        job_id=job.id,
        status="assigned"
    )
    db.add(device_training)
    
    # Update job status to running if it was pending
    if job.status == "pending":
        job.status = "running"
        job.started_at = datetime.utcnow()
    
    db.commit()
    
    job.config = json.loads(job.config)
    return {"job": job, "assignment_id": device_training.id}

@app.post("/devices/{device_id}/training/{assignment_id}/progress")
async def update_training_progress(
    device_id: str,
    assignment_id: str,
    progress: float,
    current_epoch: int,
    db: Session = Depends(get_db)
):
    device_training = db.query(DeviceTraining).filter(
        DeviceTraining.id == assignment_id,
        DeviceTraining.device_id == device_id
    ).first()
    
    if not device_training:
        raise HTTPException(status_code=404, detail="Training assignment not found")
    
    device_training.progress = progress
    device_training.status = "running"
    if not device_training.started_at:
        device_training.started_at = datetime.utcnow()
    
    # Update job progress
    job = db.query(TrainingJob).filter(TrainingJob.id == device_training.job_id).first()
    if job:
        job.current_epoch = current_epoch
        job.progress = progress
        
        # Auto-complete job if progress reaches 100%
        if progress >= 100.0:
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            device_training.status = "completed"
            device_training.completed_at = datetime.utcnow()
            device_training.progress = 100.0
            print(f"✅ Job {job.name} completed automatically at {progress}%")
    
    db.commit()
    return {"status": "progress updated"}

@app.post("/devices/{device_id}/training/{assignment_id}/complete")
async def complete_training(
    device_id: str,
    assignment_id: str,
    request_data: dict,
    db: Session = Depends(get_db)
):
    checkpoint_path = request_data.get("checkpoint_path", "")
    device_training = db.query(DeviceTraining).filter(
        DeviceTraining.id == assignment_id,
        DeviceTraining.device_id == device_id
    ).first()
    
    if not device_training:
        raise HTTPException(status_code=404, detail="Training assignment not found")
    
    device_training.status = "completed"
    device_training.completed_at = datetime.utcnow()
    device_training.checkpoint_path = checkpoint_path
    device_training.progress = 100.0
    
    db.commit()
    return {"status": "training completed"}

# Model Management
@app.get("/models/{model_name}/download")
async def download_model(model_name: str):
    model_path = Path(f"models/{model_name}.pth")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {"download_url": f"/models/{model_name}.pth"}

@app.get("/models")
async def list_models(db: Session = Depends(get_db)):
    """List all available models from database and filesystem"""
    models = []
    
    # Get models from database
    db_models = db.query(Model).all()
    for model in db_models:
        models.append({
            "id": model.id,
            "name": model.name,
            "model_type": model.model_type,
            "checkpoint_path": model.checkpoint_path,
            "size_mb": model.size_mb,
            "status": model.status,
            "created_at": model.created_at,
            "source": "database"
        })
    
    # Get models from filesystem (legacy support)
    models_dir = Path("models")
    if models_dir.exists():
        for model_file in models_dir.glob("*.pth"):
            models.append({
                "id": f"file_{model_file.stem}",
                "name": model_file.stem,
                "model_type": "unknown",
                "checkpoint_path": str(model_file),
                "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                "status": "available",
                "created_at": datetime.fromtimestamp(model_file.stat().st_mtime),
                "source": "filesystem"
            })
    
    return {"models": models}

@app.post("/models")
async def create_model(model_data: dict, db: Session = Depends(get_db)):
    """Create a new model entry in the database"""
    model = Model(
        name=model_data["name"],
        model_type=model_data["model_type"],
        checkpoint_path=model_data["checkpoint_path"],
        size_mb=model_data["size_mb"],
        status=model_data["status"]
    )
    
    db.add(model)
    db.commit()
    db.refresh(model)
    
    return {"status": "model created", "model_id": model.id}

# Federated Learning Endpoints
@app.post("/devices/{device_id}/federated-job")
async def assign_federated_job(device_id: str, job_data: dict):
    """Assign a federated learning job to a device"""
    # In a real implementation, this would queue the job for the device
    # For now, we'll just acknowledge receipt
    return {"status": "federated job assigned", "device_id": device_id}

@app.post("/devices/{device_id}/federated-update")
async def receive_federated_update(device_id: str, update_data: dict):
    """Receive model update from a device"""
    # Store the update in the database or memory
    # In a real implementation, this would be stored persistently
    return {"status": "update received", "device_id": device_id}

@app.get("/devices/{device_id}/federated-update/{round_id}")
async def get_federated_update(device_id: str, round_id: str):
    """Get federated update status for a device"""
    # In a real implementation, this would check the actual update status
    return {"status": "ready", "round_id": round_id}

@app.get("/devices/{device_id}/model-weights/{round_id}")
async def get_model_weights(device_id: str, round_id: str):
    """Get model weights from a device"""
    # In a real implementation, this would return actual model weights
    # For now, return dummy weights
    dummy_weights = {
        "embedding.weight": [[0.1] * 128] * 10000,
        "lstm.weight_ih_l0": [[0.1] * 128] * 1024,
        "lstm.weight_hh_l0": [[0.1] * 256] * 1024,
        "fc.weight": [[0.1] * 256] * 4,
        "fc.bias": [0.1] * 4
    }
    return dummy_weights

@app.post("/devices/{device_id}/global-model")
async def send_global_model(device_id: str, model_data: dict):
    """Send global model to a device"""
    # In a real implementation, this would send the actual global model
    return {"status": "global model sent", "device_id": device_id}

@app.post("/federated/start")
async def start_federated_learning(fed_config: dict):
    """Start a new federated learning process"""
    # This would integrate with the FederatedLearningCoordinator
    return {"status": "federated learning started", "round_id": "fed_round_001"}

@app.get("/federated/status/{round_id}")
async def get_federated_status(round_id: str):
    """Get status of a federated learning round"""
    return {
        "round_id": round_id,
        "status": "active",
        "participating_devices": ["device1", "device2"],
        "completed_devices": 0,
        "total_devices": 2
    }

def add_model_name_column():
    """Add model_name column to training_jobs table if it doesn't exist"""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine("sqlite:///./constellation.db")
    
    try:
        # Check if model_name column exists
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(training_jobs)"))
            columns = [row[1] for row in result.fetchall()]
            
            if 'model_name' not in columns:
                # Add the column
                conn.execute(text("ALTER TABLE training_jobs ADD COLUMN model_name VARCHAR"))
                conn.commit()
                print("✅ Added model_name column to training_jobs table")
            else:
                print("✅ model_name column already exists")
                
    except Exception as e:
        print(f"❌ Error adding model_name column: {e}")

def add_dataset_column():
    """Add dataset column to training_jobs table if it doesn't exist"""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine("sqlite:///./constellation.db")
    
    try:
        # Check if dataset column exists
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(training_jobs)"))
            columns = [row[1] for row in result.fetchall()]
            
            if 'dataset' not in columns:
                # Add the column
                conn.execute(text("ALTER TABLE training_jobs ADD COLUMN dataset VARCHAR DEFAULT 'synthetic'"))
                conn.commit()
                print("✅ Added dataset column to training_jobs table")
            else:
                print("✅ dataset column already exists")
                
    except Exception as e:
        print(f"❌ Error adding dataset column: {e}")

def fix_completed_jobs_on_startup():
    """Fix jobs that are at 100% but still marked as running on startup"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    engine = create_engine("sqlite:///./constellation.db")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Find jobs that are at 100% but still running
        jobs_to_fix = db.query(TrainingJob).filter(
            TrainingJob.status == "running",
            TrainingJob.progress >= 100.0
        ).all()
        
        if jobs_to_fix:
            print(f"🔧 Fixing {len(jobs_to_fix)} completed jobs on startup...")
            for job in jobs_to_fix:
                job.status = "completed"
                job.completed_at = datetime.utcnow()
                print(f"✅ Fixed job: {job.name} ({job.progress}%)")
            
            db.commit()
            print(f"✅ Fixed {len(jobs_to_fix)} completed jobs")
        else:
            print("✅ No completed jobs to fix")
            
    except Exception as e:
        print(f"❌ Error fixing completed jobs: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    # Run database migrations
    add_model_name_column()
    add_dataset_column()
    
    # Fix completed jobs on startup
    fix_completed_jobs_on_startup()
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
