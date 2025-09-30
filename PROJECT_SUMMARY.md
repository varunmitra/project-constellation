# Project Constellation - Comprehensive Summary

## Project Overview
**Project Constellation** is a decentralized AI and model training infrastructure that leverages idle employee devices for distributed machine learning. The system consists of a central server, desktop applications, and a web dashboard for monitoring and management.

## Architecture Components

### 1. Central Server (Python FastAPI)
- **Location**: `server/app.py`
- **Port**: 8000
- **Database**: SQLite (`constellation.db`)
- **Key Features**:
  - Device registration and management
  - Training job orchestration
  - Model management
  - Federated learning coordination
  - Real-time progress monitoring

### 2. Training Engine (Python PyTorch)
- **Location**: `training/engine.py`
- **Purpose**: Distributed training execution
- **Features**:
  - Supports vision and NLP models
  - AG News text classification
  - Dataset selection (synthetic, ag_news, imdb, yelp, amazon)
  - Checkpointing and fault tolerance
  - Progress reporting

### 3. Desktop Applications
- **Swift App**: `desktop-swift/ConstellationApp.swift` (macOS native)
- **Python App**: Alternative cross-platform option
- **Features**:
  - Device registration
  - Training controls
  - Progress monitoring
  - Idle detection
  - Resource management

### 4. Web Dashboard (React)
- **Location**: `dashboard/`
- **Port**: 3000
- **Features**:
  - Real-time monitoring
  - Device status
  - Training progress
  - Job management
  - Model repository

### 5. Federated Learning System
- **Location**: `federated/`
- **Components**:
  - `coordinator.py`: Manages distributed training
  - `client.py`: Runs on each device
  - `data_distributor.py`: Splits datasets across devices
  - `model_aggregator.py`: Merges model updates
  - `test_with_swift_app.py`: Integration testing

## Current Status (as of last session)

### ✅ Completed Features
1. **Core Infrastructure**: Server, training engine, dashboard, Swift app
2. **Device Management**: Registration, heartbeat, cleanup of duplicates
3. **Training System**: Job creation, progress tracking, completion
4. **Dataset Selection**: Multiple datasets (synthetic, ag_news, imdb, yelp, amazon)
5. **Model Management**: Training, storage, retrieval
6. **Federated Learning**: Basic implementation with coordinator and clients
7. **Codebase Cleanup**: Removed duplicates, consolidated dependencies

### 🔧 Recent Fixes Applied
1. **Device Deduplication**: Fixed multiple device registrations (reduced from 18 to 2)
2. **Server Validation**: Fixed `model_name` field validation errors
3. **Database Migration**: Added `model_name` and `dataset` columns
4. **Training Engine**: Fixed AG News training and progress reporting
5. **API Endpoints**: Fixed 422 errors in training completion

### 🚨 Current Issues
1. **New Device Registration**: Swift app on other Mac not appearing in dashboard
2. **Network Configuration**: Need to configure Swift app with server IP
3. **Training Scripts**: Missing `training/ag_news_trainer.py` causing training failures

## Network Configuration

### Server Details
- **IP Address**: `192.168.1.113`
- **Port**: `8000`
- **URL**: `http://192.168.1.113:8000`
- **Health Check**: `http://192.168.1.113:8000/health`

### Device Registration
- **Endpoint**: `POST /devices/register`
- **Heartbeat**: `POST /devices/{device_id}/heartbeat`
- **Job Assignment**: `GET /devices/{device_id}/next-job`

## Database Schema

### Devices Table
```sql
CREATE TABLE devices (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    device_type VARCHAR NOT NULL,
    cores INTEGER,
    ram_gb FLOAT,
    gpu_available BOOLEAN,
    gpu_name VARCHAR,
    is_active BOOLEAN DEFAULT TRUE,
    last_seen DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Training Jobs Table
```sql
CREATE TABLE training_jobs (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    model_name VARCHAR,
    model_type VARCHAR NOT NULL,
    dataset VARCHAR DEFAULT 'synthetic',
    status VARCHAR DEFAULT 'pending',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at DATETIME,
    completed_at DATETIME,
    total_epochs INTEGER DEFAULT 10,
    current_epoch INTEGER DEFAULT 0,
    progress FLOAT DEFAULT 0.0,
    config TEXT
);
```

### Models Table
```sql
CREATE TABLE models (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    model_type VARCHAR NOT NULL,
    checkpoint_path VARCHAR NOT NULL,
    size_mb FLOAT DEFAULT 0.0,
    status VARCHAR DEFAULT 'available',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Key API Endpoints

### Device Management
- `GET /devices` - List all devices
- `POST /devices/register` - Register new device
- `POST /devices/cleanup` - Remove duplicate devices
- `POST /devices/{device_id}/heartbeat` - Update device status

### Training Jobs
- `GET /jobs` - List all jobs
- `POST /jobs` - Create new job
- `GET /jobs/{job_id}` - Get job details
- `POST /jobs/{job_id}/start` - Start job
- `POST /jobs/{job_id}/complete` - Complete job

### Models
- `GET /models` - List all models
- `POST /models` - Create model entry
- `GET /models/{model_name}/download` - Download model

## File Structure
```
project-constellation/
├── README.md
├── requirements.txt
├── setup.sh
├── PROJECT_STRUCTURE.md
├── server/
│   └── app.py
├── training/
│   └── engine.py
├── federated/
│   ├── coordinator.py
│   ├── client.py
│   ├── data_distributor.py
│   ├── model_aggregator.py
│   └── test_with_swift_app.py
├── dashboard/
│   ├── src/
│   ├── package.json
│   └── ...
├── desktop-swift/
│   └── ConstellationApp.swift
└── scripts/
    ├── start-server.sh
    ├── start-dashboard.sh
    └── start-training.sh
```

## Setup Instructions

### 1. Server Setup
```bash
cd /Users/vmitra/Documents/GitHub/project-constellation
python3 server/app.py
```

### 2. Dashboard Setup
```bash
cd dashboard
npm install
npm start
```

### 3. Training Engine
```bash
cd training
python3 engine.py
```

### 4. Swift App
- Build and run `desktop-swift/ConstellationApp.swift`
- Configure server URL: `http://192.168.1.113:8000`

## Troubleshooting Guide

### Device Not Appearing
1. Check network connectivity: `curl http://192.168.1.113:8000/health`
2. Verify Swift app configuration
3. Check server logs for registration attempts
4. Run device cleanup: `curl -X POST http://localhost:8000/devices/cleanup`

### Training Failures
1. Check for missing training scripts
2. Verify model configuration
3. Check dataset availability
4. Review training engine logs

### Server Errors
1. Check database schema migrations
2. Verify API endpoint configurations
3. Review validation errors
4. Check CORS settings

## Recent Session Notes

### Issues Resolved
- **Duplicate Devices**: Cleaned up 18 devices down to 2
- **Server Validation**: Fixed `model_name` field validation
- **Training Progress**: Fixed 422 errors in completion endpoint
- **Database Schema**: Added missing columns with migrations

### Current State
- **Server**: Running on `http://192.168.1.113:8000`
- **Dashboard**: Accessible at `http://localhost:3000`
- **Devices**: 2 registered (both from current machine)
- **Training Engine**: Running and waiting for jobs

### Next Steps
1. Configure Swift app on other Mac with server IP
2. Test device registration from other Mac
3. Create training jobs to test distributed learning
4. Monitor federated learning progress

## Development Notes

### Code Quality
- ESLint configured for dashboard
- Type hints in Python code
- Error handling implemented
- Logging throughout system

### Performance Considerations
- Database indexing on device IDs
- Efficient device cleanup
- Async API endpoints
- Real-time progress updates

### Security Notes
- CORS configured for localhost
- API token authentication
- Device ID validation
- Input sanitization

## Contact Information
- **Project Location**: `/Users/vmitra/Documents/GitHub/project-constellation`
- **Last Updated**: September 24, 2025
- **Status**: Active Development

---

*This document serves as a comprehensive reference for continuing work on Project Constellation. All major components, configurations, and recent changes are documented for seamless project continuation.*
