# Project Constellation

**Constellation: vmitra's Garage Week 2025 project** - A decentralized AI training infrastructure that leverages idle employee workstations for distributed ML training through federated learning, reducing cloud costs and democratizing compute access.

## 🚀 Features

- **Distributed Training**: Train AI models across multiple devices
- **Federated Learning**: Privacy-preserving model aggregation
- **Dataset Diversity**: Support for multiple datasets (IMDB, Yelp, Amazon, AG News)
- **Real-time Monitoring**: Web dashboard for job and device management
- **Cross-platform**: macOS desktop app and web interface

## 📁 Project Structure

```
project-constellation/
├── server/                 # Central coordination server
│   └── app.py             # FastAPI server with REST endpoints
├── training/              # Training engine
│   ├── engine.py          # PyTorch training engine
│   ├── ag_news_trainer.py # AG News dataset trainer
│   └── checkpoints/       # Model checkpoints
├── federated/             # Federated learning components
│   ├── coordinator.py     # Federated learning coordinator
│   ├── client.py          # Federated learning client
│   ├── data_distributor.py # Data distribution utility
│   └── model_aggregator.py # Model aggregation utility
├── dashboard/             # React web dashboard
│   ├── src/
│   │   ├── pages/         # Dashboard pages
│   │   ├── components/    # Reusable components
│   │   └── context/       # React context
│   └── package.json
├── desktop-swift/         # macOS desktop application
│   ├── ConstellationApp.swift
│   └── ConstellationApp_Network.swift
└── scripts/               # Utility scripts
    ├── start-server.sh
    ├── start-training.sh
    └── start-dashboard.sh
```

## 🛠️ Quick Start

### 1. Start the Server
```bash
cd server
python3 app.py
```

### 2. Start the Training Engine
```bash
cd training
python3 engine.py
```

### 3. Start the Dashboard
```bash
cd dashboard
npm install
npm start
```

### 4. Build Desktop App
```bash
cd desktop-swift
./build.sh
```

## 🎯 Core Components

### Server (FastAPI)
- Device registration and management
- Training job coordination
- Model repository
- Federated learning coordination

### Training Engine (PyTorch)
- Distributed model training
- Checkpoint management
- Progress reporting
- Dataset handling

### Dashboard (React)
- Real-time monitoring
- Job management
- Device status
- Model repository

### Desktop App (Swift)
- Device registration
- Training controls
- Progress monitoring
- Idle detection

## 📊 Supported Datasets

- **IMDB**: Movie review sentiment analysis (2 classes)
- **Yelp**: Restaurant review classification (5 classes)
- **Amazon**: Product review classification (5 classes)
- **AG News**: News article categorization (4 classes)
- **Synthetic**: Configurable test data

## 🔧 Configuration

### Environment Variables
- `SERVER_URL`: Central server URL (default: http://localhost:8000)
- `DEVICE_ID`: Unique device identifier
- `TRAINING_INTERVAL`: Job polling interval (default: 30s)

### Database
- SQLite database for job and device management
- Automatic migrations on startup
- Persistent model storage

## 📈 Monitoring

Access the web dashboard at `http://localhost:3000` to:
- View device status
- Monitor training progress
- Manage training jobs
- Browse model repository

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For issues and questions, please open an issue on GitHub.
