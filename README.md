# Project Constellation

**A Distributed Federated Learning Platform for Decentralized AI Training**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18.2-blue.svg)](https://reactjs.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Components](#components)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Deployment](#deployment)
- [Technology Stack](#technology-stack)

---

## 🎯 Overview

**Project Constellation** is a sophisticated federated learning platform that enables distributed AI model training across multiple devices without centralizing sensitive data. The system intelligently coordinates training jobs across a network of devices (MacBooks, iMacs, Mac Studios), aggregates model updates using federated averaging, and provides real-time monitoring through a web dashboard.

### Problem Solved

Traditional machine learning requires centralizing data, which poses privacy concerns and scalability limitations. Constellation solves this by:
- **Privacy-Preserving**: Data never leaves the device
- **Scalable**: Leverages distributed compute resources
- **Efficient**: Intelligent job distribution based on device capabilities
- **Real-Time**: WebSocket-based monitoring and updates

---

## ✨ Key Features

### 🤖 Distributed Training
- **Intelligent Job Distribution**: Automatically assigns training jobs to devices based on GPU availability, CPU cores, and memory
- **Load Balancing**: Prevents device overload by limiting concurrent jobs per device
- **Checkpoint Management**: Automatic checkpointing and model versioning

### 🔄 Federated Learning
- **Federated Averaging (FedAvg)**: Aggregates model updates from multiple devices
- **Privacy-Preserving**: Training data never leaves the device
- **Multi-Round Training**: Supports iterative federated learning rounds
- **Model Aggregation**: Combines updates using weighted averaging based on sample counts

### 📊 Real-Time Monitoring
- **Web Dashboard**: React-based UI for monitoring devices, jobs, and models
- **WebSocket Updates**: Real-time progress updates and status changes
- **Device Management**: Track device availability, capabilities, and health
- **Job Tracking**: Monitor training progress, epochs, and completion status

### 🖥️ Multi-Platform Support
- **macOS Native App**: Swift-based desktop application for macOS devices
- **Python Training Engine**: Cross-platform training engine
- **RESTful API**: Standard HTTP API for integration
- **Web Dashboard**: Browser-based management interface

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Dashboard (React)                     │
│              Real-time monitoring & job management            │
└───────────────────────┬─────────────────────────────────────┘
                        │ WebSocket + REST API
┌───────────────────────┴─────────────────────────────────────┐
│              Central Server (FastAPI)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Device     │  │   Training   │  │   Model      │      │
│  │  Management  │  │   Job Queue  │  │  Aggregation │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│              Database (SQLite/PostgreSQL)                     │
└───────┬──────────────────┬──────────────────┬────────────────┘
        │                  │                  │
┌───────▼──────┐  ┌────────▼──────┐  ┌───────▼────────┐
│  Device 1    │  │   Device 2     │  │   Device N      │
│ (MacBook)    │  │   (iMac)       │  │  (Mac Studio)   │
│              │  │                │  │                 │
│ ┌──────────┐ │  │ ┌──────────┐  │  │ ┌──────────┐   │
│ │ Training │ │  │ │ Training │  │  │ │ Training │   │
│ │ Engine   │ │  │ │ Engine   │  │  │ │ Engine   │   │
│ └──────────┘ │  │ └──────────┘  │  │ └──────────┘   │
│              │  │                │  │                 │
│ Local Data   │  │ Local Data     │  │ Local Data      │
└──────────────┘  └────────────────┘  └─────────────────┘
```

### Core Components

1. **Central Server** (`server/app.py`)
   - FastAPI-based REST API
   - WebSocket server for real-time updates
   - Device registration and heartbeat management
   - Training job orchestration
   - Model aggregation engine

2. **Training Engine** (`training/engine.py`)
   - PyTorch-based model training
   - Supports vision and NLP models
   - Checkpoint management
   - Progress reporting to server

3. **Federated Learning Coordinator** (`federated/coordinator.py`)
   - Manages federated learning rounds
   - Coordinates data distribution
   - Handles model aggregation

4. **macOS Desktop App** (`desktop-swift/`)
   - Native Swift application
   - Device registration
   - Job execution
   - Training progress display

5. **Web Dashboard** (`dashboard/`)
   - React-based UI
   - Real-time device and job monitoring
   - Model management
   - Statistics and analytics

---

## 🔄 How It Works

### 1. Device Registration
- Devices (MacBooks, iMacs, etc.) register with the central server
- Server records device capabilities (CPU cores, GPU, memory)
- Devices send periodic heartbeats to maintain active status

### 2. Job Creation
- Admin creates training jobs via dashboard or API
- Job specifies: model type, dataset, epochs, configuration
- Server queues job and assigns to best available device

### 3. Intelligent Job Assignment
- Server scores devices based on:
  - GPU availability (critical for vision tasks)
  - CPU cores and memory
  - Current workload
  - Device type and capabilities
- Best matching device receives the job

### 4. Distributed Training
- Device downloads job configuration
- Loads local data (or generates synthetic data)
- Trains model locally using PyTorch
- Sends progress updates to server via WebSocket
- Uploads model weights upon completion

### 5. Federated Aggregation
- When multiple devices complete training:
  - Server collects model weights from all devices
  - Applies Federated Averaging algorithm
  - Creates aggregated global model
  - Stores model for distribution

### 6. Model Distribution
- Aggregated model available for download
- Can be used for inference or further training
- Supports multiple model types (vision, NLP, etc.)

---

## 📦 Components

### Server Components

| Component | Description | Location |
|-----------|-------------|----------|
| **API Server** | FastAPI REST API + WebSocket | `server/app.py` |
| **Training Engine** | PyTorch training logic | `training/engine.py` |
| **Job Runner** | Job execution wrapper | `training/run_job.py` |
| **Federated Coordinator** | Federated learning orchestration | `federated/coordinator.py` |
| **Model Aggregator** | FedAvg and other aggregation strategies | `federated/model_aggregator.py` |
| **Data Distributor** | Data splitting for federated learning | `federated/data_distributor.py` |

### Client Components

| Component | Description | Location |
|-----------|-------------|----------|
| **macOS App** | Swift desktop application | `desktop-swift/ConstellationApp.swift` |
| **Python Client** | Federated learning client | `federated/client.py` |
| **Web Dashboard** | React monitoring interface | `dashboard/src/` |

### Configuration Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `docker-compose.yml` | Docker orchestration |
| `Dockerfile.server` | Server container image |
| `Dockerfile.training` | Training engine container |
| `render.yaml` | Render.com deployment config |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+ (for dashboard)
- PostgreSQL (optional, SQLite works for development)
- macOS (for Swift app, optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Constellation
   ```

2. **Install Python dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install dashboard dependencies**
   ```bash
   cd dashboard
   npm install
   cd ..
   ```

4. **Initialize database**
   ```bash
   python server/app.py
   # This will create the SQLite database automatically
   ```

### Running the System

1. **Start the server**
   ```bash
   python server/app.py
   # Server runs on http://localhost:8000
   ```

2. **Start the dashboard** (in a new terminal)
   ```bash
   cd dashboard
   npm start
   # Dashboard runs on http://localhost:3000
   ```

3. **Register a device** (macOS app or Python script)
   ```bash
   # Using Python
   python -c "
   import requests
   requests.post('http://localhost:8000/devices/register', json={
       'name': 'My MacBook',
       'device_type': 'macbook',
       'os_version': '14.0',
       'cpu_cores': 8,
       'memory_gb': 16,
       'gpu_available': True,
       'gpu_memory_gb': 8
   })
   "
   ```

4. **Create a training job** (via dashboard or API)
   ```bash
   curl -X POST http://localhost:8000/jobs \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Sentiment Analysis",
       "model_type": "nlp",
       "dataset": "ag_news",
       "total_epochs": 10,
       "config": {"learning_rate": 0.001, "batch_size": 32}
     }'
   ```

---

## 💡 Usage Examples

### Example 1: Create and Run a Vision Model Training Job

```python
import requests

# Create job
job = requests.post('http://localhost:8000/jobs', json={
    'name': 'Image Classification',
    'model_type': 'vision',
    'dataset': 'synthetic',
    'total_epochs': 5,
    'config': {
        'learning_rate': 0.001,
        'batch_size': 64,
        'num_classes': 10
    }
}).json()

# Start job
requests.post(f'http://localhost:8000/jobs/{job["id"]}/start')
```

### Example 2: Federated Learning Workflow

```python
from federated.coordinator import FederatedLearningCoordinator

coordinator = FederatedLearningCoordinator(server_url="http://localhost:8000")

# Start federated training
round_id = await coordinator.start_federated_training(
    job_id="job-123",
    dataset_name="ag_news",
    model_type="nlp",
    config={
        "max_rounds": 5,
        "local_epochs": 3,
        "learning_rate": 0.001
    },
    min_devices=2,
    max_devices=5
)

# Monitor progress
status = await coordinator.get_round_status(round_id)
```

### Example 3: Model Aggregation

```python
import requests

# After devices complete training, aggregate models
result = requests.post(
    f'http://localhost:8000/federated/aggregate/{job_id}'
).json()

print(f"Aggregated model saved to: {result['model_path']}")
print(f"Participating devices: {result['participating_devices']}")
```

---

## 🌐 Deployment

### Docker Deployment

```bash
# Start all services
docker-compose up -d

# Services:
# - Server: http://localhost:8000
# - Dashboard: http://localhost:3000
# - Redis: localhost:6379
```

### Cloud Deployment (Render.com)

See [DEPLOYMENT_QUICK_START.md](./DEPLOYMENT_QUICK_START.md) for detailed instructions.

**Quick Summary:**
1. Deploy PostgreSQL database on Render
2. Deploy web service using `Dockerfile.server`
3. Deploy static site (dashboard) using `dashboard/build`
4. Configure environment variables
5. Done! 🎉

---

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **PyTorch**: Deep learning framework
- **SQLAlchemy**: Database ORM
- **Uvicorn**: ASGI server
- **WebSockets**: Real-time communication

### Frontend
- **React**: UI library
- **Tailwind CSS**: Utility-first CSS framework
- **Recharts**: Charting library
- **Axios**: HTTP client

### Infrastructure
- **Docker**: Containerization
- **PostgreSQL**: Production database
- **SQLite**: Development database
- **Redis**: Caching and job queues (optional)

### Client
- **Swift**: macOS native application
- **Python**: Training engine and client libraries

---

## 📊 Supported Model Types

- **Vision Models**: ResNet18, custom CNNs
- **NLP Models**: LSTM-based text classification
- **Custom Models**: Extensible architecture for new model types

## 📈 Supported Datasets

- **Synthetic**: Generated data for testing
- **AG News**: News article classification
- **IMDB**: Movie review sentiment
- **Custom**: Add your own datasets

---

## 🔐 Security Features

- **Device Authentication**: Token-based device registration
- **CORS Protection**: Configurable allowed origins
- **Data Privacy**: Data never leaves devices
- **Secure Aggregation**: Federated averaging preserves privacy

---

## 📝 API Documentation

When the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🤝 Contributing

This is a private project. For questions or issues, contact the project maintainer.

---

## 📄 License

See [LICENSE](./LICENSE) file for details.

---

## 🎯 Future Enhancements

- [ ] Support for more model architectures
- [ ] Advanced aggregation strategies (FedProx, SCAFFOLD)
- [ ] Differential privacy integration
- [ ] Mobile device support (iOS/iPadOS)
- [ ] Model compression and quantization
- [ ] Automated hyperparameter tuning
- [ ] Multi-tenant support
- [ ] Advanced monitoring and alerting

---

## 📞 Support

For deployment assistance, see [DEPLOYMENT_QUICK_START.md](./DEPLOYMENT_QUICK_START.md).

---

**Built with ❤️ for distributed AI training**


