"""
Central Server for Project Constellation
Handles model distribution, device coordination, and training job management
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, WebSocket, WebSocketDisconnect
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

# Database setup - supports both SQLite (dev) and PostgreSQL (production)
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./constellation.db")

# Handle PostgreSQL connection string format (Render uses postgres://, SQLAlchemy needs postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine with appropriate connection args
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # PostgreSQL connection
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

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

# CORS middleware - configurable for production
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
if allowed_origins != "*":
    allowed_origins = [origin.strip() for origin in allowed_origins.split(",")]
else:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✅ WebSocket client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"🔌 WebSocket client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"❌ Error sending WebSocket message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"❌ Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# API Endpoints

@app.get("/")
async def root():
    return {"message": "Project Constellation Server", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow(),
        "websocket_support": True,
        "websocket_endpoint": "/ws"
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Accept connection - FastAPI handles CORS for WebSocket automatically
    origin = websocket.headers.get("origin", "unknown")
    print(f"🔌 WebSocket connection attempt from origin: {origin}")
    
    try:
        await manager.connect(websocket)
        print(f"✅ WebSocket client connected successfully")
        
        # Send initial connection confirmation
        await manager.send_personal_message({
            "type": "connected",
            "message": "WebSocket connection established"
        }, websocket)
        
        # Keep connection alive with ping-pong and handle messages
        import asyncio
        while True:
            try:
                # Wait for messages with timeout to allow ping/pong
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                try:
                    # Try to parse as JSON
                    message = json.loads(data)
                    if message.get("type") == "pong":
                        # Client responded to ping, connection is alive
                        continue
                except:
                    # Not JSON, treat as plain text
                    pass
                # Echo back or handle client messages if needed
                await manager.send_personal_message({"type": "pong", "message": "Connection active"}, websocket)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive (Render may timeout idle connections)
                try:
                    await websocket.send_json({"type": "ping", "timestamp": datetime.utcnow().isoformat()})
                except Exception as ping_error:
                    # Connection might be closed
                    print(f"⚠️ Ping failed, connection may be closed: {ping_error}")
                    break
    except WebSocketDisconnect:
        print(f"🔌 WebSocket client disconnected normally")
        manager.disconnect(websocket)
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        manager.disconnect(websocket)

# Device Management
@app.post("/devices/register", response_model=DeviceResponse)
async def register_device(device: DeviceCreate, request: Request, db: Session = Depends(get_db)):
    # Validate that only Swift apps can register as devices
    user_agent = request.headers.get("user-agent", "").lower()
    constellation_header = request.headers.get("x-constellation-client", "").lower()
    
    # Check if this is a legitimate Swift app registration
    is_swift_app = (
        "constellation" in user_agent or 
        "swift" in user_agent or
        constellation_header == "swift-app" or
        constellation_header == "constellation-swift"
    )
    
    if not is_swift_app:
        raise HTTPException(
            status_code=403, 
            detail="Only Swift Constellation apps can register as devices. Training engines should not register as devices."
        )
    
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
    
    # Broadcast device update via WebSocket
    await manager.broadcast({
        "type": "device_registered",
        "device": {
            "id": db_device.id,
            "name": db_device.name,
            "device_type": db_device.device_type,
            "is_active": db_device.is_active
        }
    })
    
    return db_device

@app.get("/devices", response_model=List[DeviceResponse])
async def list_devices(db: Session = Depends(get_db)):
    # Cleanup inactive devices first
    cleanup_inactive_devices(db)
    devices = db.query(Device).filter(Device.is_active == True).all()
    return devices

@app.post("/devices/cleanup")
async def cleanup_devices(db: Session = Depends(get_db)):
    """Clean up inactive and duplicate devices"""
    # First, mark inactive devices
    inactive_count = cleanup_inactive_devices(db)
    
    # Then remove duplicates
    devices = db.query(Device).all()
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
    return {
        "message": f"Cleaned up {inactive_count} inactive devices and {removed_count} duplicate devices",
        "inactive_cleaned": inactive_count,
        "duplicates_removed": removed_count
    }

@app.get("/devices/{device_id}", response_model=DeviceResponse)
async def get_device(device_id: str, db: Session = Depends(get_db)):
    device = db.query(Device).filter(Device.id == device_id).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    return device

def cleanup_inactive_devices(db: Session):
    """Mark devices as inactive if they haven't been seen for more than 5 minutes"""
    cutoff_time = datetime.utcnow() - timedelta(minutes=5)
    inactive_devices = db.query(Device).filter(
        Device.last_seen < cutoff_time,
        Device.is_active == True
    ).all()
    
    for device in inactive_devices:
        device.is_active = False
        print(f"🔌 Marked device as inactive: {device.name} (last seen: {device.last_seen})")
    
    db.commit()
    return len(inactive_devices)

@app.post("/devices/{device_id}/heartbeat")
async def device_heartbeat(device_id: str, db: Session = Depends(get_db)):
    device = db.query(Device).filter(Device.id == device_id).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    device.last_seen = datetime.utcnow()
    device.is_active = True
    db.commit()
    
    # Cleanup inactive devices
    cleanup_inactive_devices(db)
    
    # Broadcast device heartbeat via WebSocket
    await manager.broadcast({
        "type": "device_heartbeat",
        "device_id": device_id,
        "last_seen": device.last_seen.isoformat()
    })
    
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
    
    # Broadcast new job via WebSocket
    await manager.broadcast({
        "type": "job_created",
        "job": {
            "id": db_job.id,
            "name": db_job.name,
            "status": db_job.status,
            "model_type": db_job.model_type,
            "dataset": db_job.dataset
        }
    })
    
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

def calculate_device_score(device: Device, job: TrainingJob) -> float:
    """Calculate a score for how well a device matches a job's requirements"""
    score = 0.0
    
    # Base score from CPU cores (more cores = better)
    score += device.cpu_cores * 10
    
    # GPU bonus for vision/image tasks
    if job.model_type == "vision" and device.gpu_available:
        score += 1000  # Strong preference for GPU on vision tasks
        score += device.gpu_memory_gb * 50  # More GPU memory = better
    
    # GPU bonus for NLP tasks (can still benefit from GPU)
    elif job.model_type in ["nlp", "text_classification"] and device.gpu_available:
        score += 500  # Moderate preference for GPU
    
    # Memory bonus
    score += device.memory_gb * 5
    
    # Penalty for inactive devices
    if not device.is_active:
        score = 0
    
    # Check if device is already busy (has active training assignments)
    # This will be checked separately, but we can factor it in
    return score

def get_best_device_for_job(job: TrainingJob, db: Session) -> Optional[Device]:
    """Find the best available device for a job using intelligent scoring"""
    # Get all active devices
    active_devices = db.query(Device).filter(Device.is_active == True).all()
    
    if not active_devices:
        return None
    
    # Calculate scores for each device
    device_scores = []
    for device in active_devices:
        # Check if device already has too many active assignments
        active_assignments = db.query(DeviceTraining).filter(
            DeviceTraining.device_id == device.id,
            DeviceTraining.status.in_(["assigned", "running"])
        ).count()
        
        # Skip devices with too many active assignments (load balancing)
        max_concurrent_jobs = 2  # Allow up to 2 concurrent jobs per device
        if active_assignments >= max_concurrent_jobs:
            continue
        
        score = calculate_device_score(device, job)
        device_scores.append((device, score, active_assignments))
    
    if not device_scores:
        return None
    
    # Sort by score (descending), then by active assignments (ascending) for load balancing
    device_scores.sort(key=lambda x: (-x[1], x[2]))
    
    return device_scores[0][0]  # Return best device

@app.post("/jobs/{job_id}/start")
async def start_training_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    # Check if job is already completed
    if job.status == "completed":
        return {
            "status": "job already completed",
            "job_id": job_id,
            "message": "Job has already been completed"
        }
    
    # Check if job is already running and has active assignments
    if job.status == "running":
        active_assignments = db.query(DeviceTraining).filter(
            DeviceTraining.job_id == job_id,
            DeviceTraining.status.in_(["assigned", "running"])
        ).count()
        
        if active_assignments > 0:
            # Job is already running with active assignments
            assigned_device_name = None
            assignment = db.query(DeviceTraining).filter(
                DeviceTraining.job_id == job_id,
                DeviceTraining.status.in_(["assigned", "running"])
            ).first()
            if assignment:
                device = db.query(Device).filter(Device.id == assignment.device_id).first()
                assigned_device_name = device.name if device else None
            
            return {
                "status": "job already running",
                "job_id": job_id,
                "assigned_device": assigned_device_name,
                "message": "Job is already running with active assignments"
            }
        # Job is in "running" status but has no active assignments - allow restart
        print(f"⚠️ Job '{job.name}' is in running status but has no active assignments - restarting")
    
    # Allow starting if job is pending, or if it's running but has no active assignments
    if job.status not in ["pending", "running"]:
        raise HTTPException(status_code=400, detail=f"Job is in '{job.status}' status and cannot be started")
    
    # Intelligent job distribution: find best device for this job
    best_device = get_best_device_for_job(job, db)
    
    if best_device:
        # Check if assignment already exists (avoid duplicates)
        existing_assignment = db.query(DeviceTraining).filter(
            DeviceTraining.device_id == best_device.id,
            DeviceTraining.job_id == job.id,
            DeviceTraining.status.in_(["assigned", "running"])
        ).first()
        
        if not existing_assignment:
            # Create assignment for the best device
            device_training = DeviceTraining(
                device_id=best_device.id,
                job_id=job.id,
                status="assigned"
            )
            db.add(device_training)
            print(f"✅ Assigned job '{job.name}' to device '{best_device.name}' (GPU: {best_device.gpu_available}, Cores: {best_device.cpu_cores})")
        else:
            print(f"ℹ️ Job '{job.name}' already assigned to device '{best_device.name}'")
    else:
        print(f"⚠️ No available devices for job '{job.name}', will be assigned when device requests it")
    
    # Update job status to running if it was pending
    if job.status == "pending":
        job.status = "running"
        if not job.started_at:
            job.started_at = datetime.utcnow()
    
    db.commit()
    
    return {
        "status": "job started", 
        "job_id": job_id,
        "assigned_device": best_device.name if best_device else None
    }

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
    # Get the device
    device = db.query(Device).filter(Device.id == device_id).first()
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    
    if not device.is_active:
        return {"job": None, "message": "Device is not active"}
    
    # Find available training jobs (pending or running, exclude completed jobs)
    available_jobs = db.query(TrainingJob).filter(
        TrainingJob.status.in_(["pending", "running"])
    ).all()
    
    # Filter out jobs that are at 100% progress (should be completed)
    available_jobs = [job for job in available_jobs if job.progress < 100.0]
    
    if not available_jobs:
        return {"job": None, "message": "No available training jobs"}
    
    # Check how many active assignments this device already has
    active_assignments = db.query(DeviceTraining).filter(
        DeviceTraining.device_id == device_id,
        DeviceTraining.status.in_(["assigned", "running"])
    ).count()
    
    max_concurrent_jobs = 2  # Limit concurrent jobs per device
    if active_assignments >= max_concurrent_jobs:
        return {"job": None, "message": "Device is already at maximum concurrent jobs"}
    
    # Score and rank jobs for this device
    job_scores = []
    for job in available_jobs:
        # Check if already assigned to this device
        existing_assignment = db.query(DeviceTraining).filter(
            DeviceTraining.device_id == device_id,
            DeviceTraining.job_id == job.id,
            DeviceTraining.status.in_(["assigned", "running"])
        ).first()
        
        if existing_assignment:
            # If already assigned, prioritize continuing this job
            job_scores.append((job, 10000, existing_assignment.id))
            continue
        
        # Calculate how well this device matches the job
        score = calculate_device_score(device, job)
        
        # Bonus for jobs that haven't started (progress = 0)
        if job.progress == 0:
            score += 100
        
        # Penalty for jobs that are already running elsewhere
        other_assignments = db.query(DeviceTraining).filter(
            DeviceTraining.job_id == job.id,
            DeviceTraining.status.in_(["assigned", "running"])
        ).count()
        if other_assignments > 0:
            score -= 50 * other_assignments  # Prefer jobs not yet started elsewhere
        
        job_scores.append((job, score, None))
    
    # Sort by score (descending) and creation time (ascending for tie-breaking)
    job_scores.sort(key=lambda x: (-x[1], x[0].created_at))
    
    if not job_scores:
        return {"job": None, "message": "No suitable jobs found"}
    
    # Get the best job
    best_job, best_score, existing_assignment_id = job_scores[0]
    
    # If job is already assigned to this device, return existing assignment
    if existing_assignment_id:
        device_training = db.query(DeviceTraining).filter(
            DeviceTraining.id == existing_assignment_id
        ).first()
        best_job.config = json.loads(best_job.config)
        return {"job": best_job, "assignment_id": device_training.id}
    
    # Create new device training assignment
    device_training = DeviceTraining(
        device_id=device_id,
        job_id=best_job.id,
        status="assigned"
    )
    db.add(device_training)
    
    # Update job status to running if it was pending
    if best_job.status == "pending":
        best_job.status = "running"
        best_job.started_at = datetime.utcnow()
    
    db.commit()
    
    print(f"✅ Assigned job '{best_job.name}' to device '{device.name}' (score: {best_score:.1f})")
    
    best_job.config = json.loads(best_job.config)
    return {"job": best_job, "assignment_id": device_training.id}

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
    
    # Broadcast progress update via WebSocket
    await manager.broadcast({
        "type": "job_progress",
        "job_id": job.id if job else device_training.job_id,
        "progress": progress,
        "current_epoch": current_epoch,
        "device_id": device_id
    })
    
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
    
    # Update job status
    job = db.query(TrainingJob).filter(TrainingJob.id == device_training.job_id).first()
    if job:
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.progress = 100.0
        print(f"✅ Job {job.name} marked as completed")
    
    db.commit()
    
    # Broadcast completion via WebSocket
    await manager.broadcast({
        "type": "job_completed",
        "job_id": device_training.job_id,
        "device_id": device_id,
        "assignment_id": assignment_id
    })
    
    # Check if all devices have completed - if so, trigger aggregation
    all_assignments = db.query(DeviceTraining).filter(
        DeviceTraining.job_id == device_training.job_id
    ).all()
    
    completed_count = sum(1 for a in all_assignments if a.status == "completed")
    total_count = len(all_assignments)
    
    if completed_count == total_count and total_count > 1:
        print(f"🔄 All {total_count} devices completed. Triggering aggregation...")
        # Trigger aggregation in background
        try:
            from fastapi import BackgroundTasks
            # Note: This would need BackgroundTasks parameter in function signature
            # For now, we'll just log it - aggregation can be triggered manually via API
            print(f"   Call POST /federated/aggregate/{device_training.job_id} to aggregate models")
        except:
            pass
    
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
async def receive_federated_update(device_id: str, update_data: dict, db: Session = Depends(get_db)):
    """Receive model update from a device for federated learning"""
    import json
    import numpy as np
    
    assignment_id = update_data.get("assignment_id")
    model_weights = update_data.get("model_weights", {})
    sample_count = update_data.get("sample_count", 1000)
    loss = update_data.get("loss", 0.0)
    accuracy = update_data.get("accuracy", 0.0)
    checkpoint_path = update_data.get("checkpoint_path", "")
    
    # Get the training assignment
    device_training = db.query(DeviceTraining).filter(
        DeviceTraining.id == assignment_id,
        DeviceTraining.device_id == device_id
    ).first()
    
    if not device_training:
        raise HTTPException(status_code=404, detail="Training assignment not found")
    
    # Store model weights in a temporary storage (in production, use proper storage)
    # For now, we'll store metadata and the server will aggregate when ready
    # Use absolute path to ensure directory is created in project root
    import os
    project_root = Path(__file__).parent.parent  # Go up from server/ to project root
    federated_updates_dir = project_root / "federated_updates"
    federated_updates_dir.mkdir(exist_ok=True)
    print(f"📁 Federated updates directory: {federated_updates_dir.absolute()}")
    
    # Save update to file (in production, use database or object storage)
    update_file = federated_updates_dir / f"{assignment_id}_{device_id}.json"
    update_data_to_save = {
        "assignment_id": assignment_id,
        "device_id": device_id,
        "job_id": device_training.job_id,
        "model_weights": model_weights,
        "sample_count": sample_count,
        "loss": loss,
        "accuracy": accuracy,
        "checkpoint_path": checkpoint_path,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    with open(update_file, 'w') as f:
        json.dump(update_data_to_save, f)
    
    print(f"✅ Received model weights from device {device_id} for assignment {assignment_id}")
    print(f"   Samples: {sample_count}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Broadcast update via WebSocket
    await manager.broadcast({
        "type": "federated_update_received",
        "device_id": device_id,
        "assignment_id": assignment_id,
        "job_id": device_training.job_id,
        "sample_count": sample_count
    })
    
    return {"status": "update received", "device_id": device_id, "assignment_id": assignment_id}

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

@app.post("/federated/aggregate/{job_id}")
async def aggregate_models(job_id: str, db: Session = Depends(get_db)):
    """Aggregate model weights from all devices that completed training for a job"""
    import json
    import numpy as np
    import torch
    from pathlib import Path
    import traceback
    
    try:
        # Get all completed assignments for this job
        completed_assignments = db.query(DeviceTraining).filter(
            DeviceTraining.job_id == job_id,
            DeviceTraining.status == "completed"
        ).all()
        
        if len(completed_assignments) < 1:
            raise HTTPException(status_code=400, detail="Need at least 1 completed assignment to aggregate")
        
        print(f"🔄 Aggregating models from {len(completed_assignments)} devices for job {job_id}")
        
        # Load model updates from files
        import os
        project_root = Path(__file__).parent.parent  # Go up from server/ to project root
        federated_updates_dir = project_root / "federated_updates"
        federated_updates_dir.mkdir(exist_ok=True)
        device_updates = []
        
        # Also try to find updates by job_id in the JSON files (fallback)
        update_files_by_job = []
        if federated_updates_dir.exists():
            for update_file in federated_updates_dir.glob("*.json"):
                try:
                    with open(update_file, 'r') as f:
                        update_data = json.load(f)
                        if update_data.get("job_id") == job_id:
                            update_files_by_job.append((update_file, update_data))
                except Exception as e:
                    print(f"⚠️  Error reading {update_file}: {e}")
                    continue
        
        # Try to match assignments with update files
        for assignment in completed_assignments:
            update_file = federated_updates_dir / f"{assignment.id}_{assignment.device_id}.json"
            update_data = None
            
            if update_file.exists():
                try:
                    with open(update_file, 'r') as f:
                        update_data = json.load(f)
                except Exception as e:
                    print(f"⚠️  Error reading {update_file}: {e}")
                    continue
            else:
                # Fallback: find by job_id and device_id
                print(f"⚠️  Update file not found: {update_file.name}, trying fallback...")
                for file_path, data in update_files_by_job:
                    if data.get("device_id") == assignment.device_id:
                        update_file = file_path
                        update_data = data
                        print(f"✅ Found update file: {update_file.name}")
                        break
            
            if update_data and "model_weights" in update_data:
                try:
                    # Convert list back to numpy arrays
                    model_weights = {}
                    for key, value in update_data["model_weights"].items():
                        if isinstance(value, list):
                            # Check if it's a nested list (multi-dimensional array)
                            if len(value) > 0 and isinstance(value[0], list):
                                arr = np.array(value, dtype=np.float32)
                            else:
                                # 1D array - check if it contains integers (like num_batches_tracked)
                                arr = np.array(value)
                                if arr.dtype == np.int64 or arr.dtype == np.int32:
                                    # Keep as int for tracking values, but ensure it's compatible
                                    model_weights[key] = arr.astype(np.int64)
                                    continue
                                else:
                                    arr = arr.astype(np.float32)
                            model_weights[key] = arr
                        elif isinstance(value, (int, float)):
                            # Handle scalar values - keep integers as int, floats as float32
                            if isinstance(value, int):
                                # Store as scalar numpy int64, not array
                                model_weights[key] = np.int64(value)
                            else:
                                model_weights[key] = np.float32(value)
                        else:
                            model_weights[key] = np.array(value, dtype=np.float32)
                    
                    device_updates.append({
                        "device_id": assignment.device_id,
                        "model_weights": model_weights,
                        "sample_count": update_data.get("sample_count", 1000),
                        "loss": update_data.get("loss", 0.0),
                        "accuracy": update_data.get("accuracy", 0.0)
                    })
                    print(f"✅ Loaded update from device {assignment.device_id}")
                except Exception as e:
                    print(f"❌ Error processing model weights for {assignment.device_id}: {e}")
                    traceback.print_exc()
                    continue
        
        if not device_updates:
            # If still no updates, try using all updates for this job_id
            if update_files_by_job:
                print(f"⚠️  No updates matched assignments, using all updates for job {job_id}")
                for file_path, update_data in update_files_by_job:
                    try:
                        model_weights = {}
                        for key, value in update_data["model_weights"].items():
                            if isinstance(value, list):
                                model_weights[key] = np.array(value, dtype=np.float32)
                            else:
                                model_weights[key] = np.array(value, dtype=np.float32)
                        
                        device_updates.append({
                            "device_id": update_data.get("device_id", "unknown"),
                            "model_weights": model_weights,
                            "sample_count": update_data.get("sample_count", 1000),
                            "loss": update_data.get("loss", 0.0),
                            "accuracy": update_data.get("accuracy", 0.0)
                        })
                    except Exception as e:
                        print(f"❌ Error processing update from {file_path}: {e}")
                        continue
            
            if not device_updates:
                raise HTTPException(status_code=404, detail=f"No model updates found for job {job_id}. Checked {len(completed_assignments)} assignments and {len(update_files_by_job)} update files.")
        
        # Aggregate using Federated Averaging
        total_samples = sum(update["sample_count"] for update in device_updates)
        if total_samples == 0:
            total_samples = len(device_updates) * 1000  # Fallback
        
        aggregated_weights = {}
        
        # Get layer names from first update, filtering out integer tracking values
        first_update = device_updates[0]
        integer_keys = set()  # Track integer keys to skip during aggregation
        
        for layer_name in first_update["model_weights"].keys():
            try:
                first_weight = first_update["model_weights"][layer_name]
                # Check if this is an integer tracking value (like num_batches_tracked)
                is_integer = False
                if isinstance(first_weight, (np.integer, int)):
                    is_integer = True
                elif isinstance(first_weight, np.ndarray):
                    # Check if it's an integer array (any size) or contains num_batches_tracked
                    if np.issubdtype(first_weight.dtype, np.integer) or "num_batches_tracked" in layer_name:
                        is_integer = True
                
                if is_integer:
                    # Store integer tracking values as-is (use first value)
                    integer_keys.add(layer_name)
                    if isinstance(first_weight, np.ndarray):
                        aggregated_weights[layer_name] = first_weight.copy()
                    else:
                        aggregated_weights[layer_name] = np.int64(first_weight)
                    print(f"  📌 Skipping integer key: {layer_name}")
                else:
                    # Ensure we use float32 for aggregation to avoid dtype casting errors
                    # Convert to float32 first, then create zeros
                    if isinstance(first_weight, np.ndarray):
                        first_weight_float32 = first_weight.astype(np.float32)
                    else:
                        first_weight_float32 = np.array(first_weight, dtype=np.float32)
                    aggregated_weights[layer_name] = np.zeros(first_weight_float32.shape, dtype=np.float32)
                    print(f"  ✅ Initialized float32 key: {layer_name} (shape: {aggregated_weights[layer_name].shape}, dtype: {aggregated_weights[layer_name].dtype})")
            except Exception as e:
                print(f"⚠️  Error initializing layer {layer_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Weighted average
        for update in device_updates:
            weight_factor = update["sample_count"] / total_samples
            for layer_name, weights in update["model_weights"].items():
                if layer_name in aggregated_weights:
                    # Skip integer tracking values - use first value as-is
                    if layer_name in integer_keys:
                        continue  # Skip aggregation for integer tracking values
                    
                    try:
                        # Ensure shapes match and convert to float32
                        if isinstance(weights, np.ndarray):
                            weights_float32 = weights.astype(np.float32)
                        else:
                            weights_float32 = np.array(weights, dtype=np.float32)
                        
                        # Defensive check: ensure aggregated_weights is float32
                        if aggregated_weights[layer_name].dtype != np.float32:
                            print(f"⚠️  Converting {layer_name} from {aggregated_weights[layer_name].dtype} to float32")
                            aggregated_weights[layer_name] = aggregated_weights[layer_name].astype(np.float32)
                        
                        if aggregated_weights[layer_name].shape == weights_float32.shape:
                            # Use explicit assignment to avoid dtype casting issues
                            aggregated_weights[layer_name] = (aggregated_weights[layer_name] + (weight_factor * weights_float32)).astype(np.float32)
                        else:
                            print(f"⚠️  Shape mismatch for {layer_name}: {aggregated_weights[layer_name].shape} vs {weights_float32.shape}")
                    except Exception as e:
                        print(f"❌ Error aggregating layer {layer_name}: {e}")
                        print(f"   Aggregated dtype: {aggregated_weights[layer_name].dtype}, shape: {aggregated_weights[layer_name].shape}")
                        print(f"   Weights dtype: {weights.dtype if isinstance(weights, np.ndarray) else type(weights)}, shape: {weights.shape if isinstance(weights, np.ndarray) else 'N/A'}")
                        import traceback
                        traceback.print_exc()
                        continue
        
        # Convert numpy arrays back to PyTorch tensors and save
        aggregated_state_dict = {}
        for key, value in aggregated_weights.items():
            try:
                aggregated_state_dict[key] = torch.from_numpy(value)
            except Exception as e:
                print(f"⚠️  Error converting {key} to tensor: {e}")
                continue
        
        # Save aggregated model
        import os
        project_root = Path(__file__).parent.parent  # Go up from server/ to project root
        models_dir = project_root / "federated_models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / f"aggregated_model_{job_id}.pth"
        
        # Get job config for num_classes if available
        job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
        job_config = {}
        if job:
            try:
                job_config = json.loads(job.config) if isinstance(job.config, str) else job.config
            except:
                job_config = {}
        
        torch.save({
            "model_state_dict": aggregated_state_dict,
            "job_id": job_id,
            "participating_devices": [u["device_id"] for u in device_updates],
            "total_samples": total_samples,
            "avg_loss": float(np.mean([u["loss"] for u in device_updates])),
            "avg_accuracy": float(np.mean([u["accuracy"] for u in device_updates])),
            "config": job_config,  # Include config for model loading
            "timestamp": datetime.utcnow().isoformat()
        }, model_path)
        
        # Update job with aggregated model path
        if job:
            job_config["aggregated_model_path"] = str(model_path)
            job.config = json.dumps(job_config)
            db.commit()
        
        print(f"✅ Model aggregated and saved to {model_path}")
        print(f"   Total samples: {total_samples}")
        print(f"   Participating devices: {len(device_updates)}")
        print(f"   Average accuracy: {np.mean([u['accuracy'] for u in device_updates]):.2f}%")
        
        # Broadcast aggregation completion
        try:
            await manager.broadcast({
                "type": "model_aggregated",
                "job_id": job_id,
                "model_path": str(model_path),
                "participating_devices": len(device_updates)
            })
        except Exception as e:
            print(f"⚠️  Error broadcasting aggregation: {e}")
        
        return {
            "status": "aggregated",
            "job_id": job_id,
            "model_path": str(model_path),
            "participating_devices": len(device_updates),
            "total_samples": total_samples
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in aggregation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Aggregation failed: {str(e)}")

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
    """Add model_name column to training_jobs table if it doesn't exist (legacy migration)"""
    from sqlalchemy import text, inspect
    from sqlalchemy.exc import OperationalError, ProgrammingError
    
    try:
        # Test database connection first
        with engine.connect() as test_conn:
            test_conn.execute(text("SELECT 1"))
        
        # Check if table exists first
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        if 'training_jobs' not in table_names:
            print("ℹ️ training_jobs table doesn't exist yet, skipping migration (will be created by Base.metadata.create_all)")
            return
        
        # Check if model_name column exists
        try:
            columns = [col['name'] for col in inspector.get_columns('training_jobs')]
        except (OperationalError, ProgrammingError) as e:
            print(f"ℹ️ Could not inspect columns (table may not exist yet): {e}")
            return
        except Exception as e:
            print(f"ℹ️ Could not inspect columns: {e}")
            return
        
        if 'model_name' not in columns:
            # Add the column - use transaction properly
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE training_jobs ADD COLUMN model_name VARCHAR"))
            print("✅ Added model_name column to training_jobs table")
        else:
            print("ℹ️ model_name column already exists")
                
    except (OperationalError, ProgrammingError) as e:
        # These are expected if table doesn't exist - non-critical
        print(f"ℹ️ Migration check for model_name column skipped: {e}")
    except Exception as e:
        # Don't fail startup if migration fails
        print(f"ℹ️ Migration check for model_name column: {e} (non-critical)")

def add_dataset_column():
    """Add dataset column to training_jobs table if it doesn't exist (legacy migration)"""
    from sqlalchemy import text, inspect
    from sqlalchemy.exc import OperationalError, ProgrammingError
    
    try:
        # Test database connection first
        with engine.connect() as test_conn:
            test_conn.execute(text("SELECT 1"))
        
        # Check if table exists first
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        if 'training_jobs' not in table_names:
            print("ℹ️ training_jobs table doesn't exist yet, skipping migration (will be created by Base.metadata.create_all)")
            return
        
        # Check if dataset column exists
        try:
            columns = [col['name'] for col in inspector.get_columns('training_jobs')]
        except (OperationalError, ProgrammingError) as e:
            print(f"ℹ️ Could not inspect columns (table may not exist yet): {e}")
            return
        except Exception as e:
            print(f"ℹ️ Could not inspect columns: {e}")
            return
        
        if 'dataset' not in columns:
            # Add the column - use transaction properly
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE training_jobs ADD COLUMN dataset VARCHAR DEFAULT 'synthetic'"))
            print("✅ Added dataset column to training_jobs table")
        else:
            print("ℹ️ dataset column already exists")
                
    except (OperationalError, ProgrammingError) as e:
        # These are expected if table doesn't exist - non-critical
        print(f"ℹ️ Migration check for dataset column skipped: {e}")
    except Exception as e:
        # Don't fail startup if migration fails
        print(f"ℹ️ Migration check for dataset column: {e} (non-critical)")

def fix_completed_jobs_on_startup():
    """Fix jobs that are at 100% but still marked as running on startup"""
    from sqlalchemy import inspect, text
    from sqlalchemy.exc import OperationalError, ProgrammingError
    
    try:
        # Test database connection first
        with engine.connect() as test_conn:
            test_conn.execute(text("SELECT 1"))
        
        # Check if table exists first
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        
        if 'training_jobs' not in table_names:
            print("ℹ️ training_jobs table doesn't exist yet, skipping job fix")
            return
        
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
                print("ℹ️ No completed jobs to fix")
                
        except (OperationalError, ProgrammingError) as e:
            print(f"ℹ️ Error fixing completed jobs (table may not exist yet): {e}")
            db.rollback()
        except Exception as e:
            print(f"ℹ️ Error fixing completed jobs: {e} (non-critical)")
            db.rollback()
        finally:
            db.close()
            
    except (OperationalError, ProgrammingError) as e:
        # These are expected if table doesn't exist - non-critical
        print(f"ℹ️ Error checking table existence for job fix: {e} (non-critical)")
    except Exception as e:
        # Don't fail startup if this check fails
        print(f"ℹ️ Error checking table existence for job fix: {e} (non-critical)")

if __name__ == "__main__":
    # Debug: Log database connection info
    db_type = "SQLite" if DATABASE_URL.startswith("sqlite") else "PostgreSQL"
    print(f"🔍 Database connection: {db_type}")
    print(f"🔍 DATABASE_URL starts with: {DATABASE_URL[:20]}...")
    
    # Run database migrations (these are legacy migrations for backward compatibility)
    # Note: These columns are already in the TrainingJob model, so new databases
    # will have them created automatically by Base.metadata.create_all()
    add_model_name_column()
    add_dataset_column()
    
    # Fix completed jobs on startup
    fix_completed_jobs_on_startup()
    
    import uvicorn
    # Get port from environment (for cloud deployment)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)