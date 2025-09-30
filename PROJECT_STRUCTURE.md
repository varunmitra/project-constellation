# Project Constellation - Clean Structure

## 📁 Directory Structure

```
project-constellation/
├── README.md                    # Main project documentation
├── requirements.txt            # Python dependencies
├── setup.sh                    # Setup script
├── PROJECT_STRUCTURE.md        # This file
│
├── server/                     # Central server
│   ├── app.py                  # FastAPI server
│   └── constellation.db        # SQLite database
│
├── training/                   # Training engine
│   ├── engine.py               # PyTorch training engine
│   ├── ag_news_trainer.py      # AG News trainer
│   ├── checkpoints/            # Model checkpoints
│   └── data/                   # Training data
│
├── federated/                  # Federated learning
│   ├── coordinator.py          # FL coordinator
│   ├── client.py               # FL client
│   ├── data_distributor.py    # Data distribution
│   ├── model_aggregator.py    # Model aggregation
│   ├── demo_federated_learning.py
│   └── test_with_swift_app.py
│
├── dashboard/                  # React dashboard
│   ├── src/
│   │   ├── pages/              # Dashboard pages
│   │   ├── components/         # Reusable components
│   │   └── context/            # React context
│   ├── package.json
│   └── public/
│
├── desktop-swift/              # macOS desktop app
│   ├── ConstellationApp.swift
│   ├── ConstellationApp_Network.swift
│   ├── build.sh
│   └── install.sh
│
├── scripts/                    # Utility scripts
│   ├── start-server.sh
│   ├── start-training.sh
│   └── start-dashboard.sh
│
└── checkpoints/                # Global model checkpoints
    ├── ag_news_model.pth
    └── ag_news_real_model.pth
```

## 🧹 Cleanup Summary

### Removed Files:
- ❌ Duplicate distribution packages
- ❌ Temporary demo files
- ❌ Redundant documentation
- ❌ Test data directories
- ❌ Python cache files
- ❌ Duplicate requirements files
- ❌ Temporary training results

### Kept Essential Files:
- ✅ Core server and training engine
- ✅ Federated learning components
- ✅ React dashboard
- ✅ Swift desktop app
- ✅ Utility scripts
- ✅ Model checkpoints
- ✅ Main documentation

## 🚀 Quick Start

1. **Setup**: `./setup.sh`
2. **Start Server**: `./scripts/start-server.sh`
3. **Start Training**: `./scripts/start-training.sh`
4. **Start Dashboard**: `./scripts/start-dashboard.sh`

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Swift App     │    │   Web Dashboard│    │   Training      │
│   (Desktop)     │◄──►│   (React)      │◄──►│   Engine        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   FastAPI       │
                    │   Server        │
                    │   (Central)    │
                    └─────────────────┘
```

## 🎯 Key Features

- **Distributed Training**: Multi-device model training
- **Federated Learning**: Privacy-preserving aggregation
- **Dataset Diversity**: Multiple dataset support
- **Real-time Monitoring**: Web dashboard
- **Cross-platform**: macOS + Web interface
