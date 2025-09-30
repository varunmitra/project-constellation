# 🧠 Project Constellation - Swift Desktop App

A native macOS application for distributed AI training that runs in your menu bar.

## 🚀 Quick Installation

### Option 1: Build and Run (Recommended)
```bash
cd desktop-swift
./build.sh
./build/Constellation
```

### Option 2: Install System-Wide
```bash
cd desktop-swift
./install.sh
# Then copy Constellation.app to /Applications/
```

## 📱 Features

- **Menu Bar Integration**: Runs in your menu bar with a brain icon
- **Automatic Registration**: Registers your device with the central server
- **Background Training**: Automatically starts training when jobs are available
- **Real-time Status**: Shows training progress and job information
- **Heartbeat Monitoring**: Maintains connection with the central server

## 🎯 Usage

1. **Start the app**: Run `./build/Constellation` or double-click `Constellation.app`
2. **Look for the brain icon** in your menu bar (top-right of your screen)
3. **Right-click the icon** to see the menu with:
   - Current status
   - Training progress
   - Device information
   - Start/Stop controls
   - Quit option

## 🔧 Requirements

- macOS 10.15+ (Catalina or later)
- Swift 5.0+ (included with Xcode Command Line Tools)
- Central server running on http://localhost:8000

## 🛠️ Development

### Build from Source
```bash
# Compile the Swift application
swiftc -o build/Constellation \
    -framework AppKit \
    -framework Foundation \
    ConstellationApp.swift
```

### Run in Development
```bash
# Start the server first
cd ../server && python3 app.py

# Then run the app
./build/Constellation
```

## 📊 What It Does

1. **Registers your device** with the central server
2. **Sends heartbeats** every 30 seconds to stay connected
3. **Checks for training jobs** every 10 seconds
4. **Automatically starts training** when jobs are available
5. **Updates progress** in real-time to the server
6. **Shows status** in the menu bar

## 🎉 Success Indicators

- ✅ Brain icon appears in menu bar
- ✅ Status shows "Connected" or "Training"
- ✅ Progress updates in real-time
- ✅ Training jobs complete automatically

## 🛑 Troubleshooting

### App Won't Start
- Check if Swift is installed: `swift --version`
- Install Xcode Command Line Tools: `xcode-select --install`

### Can't Connect to Server
- Make sure the central server is running: `cd ../server && python3 app.py`
- Check server URL in the code (default: http://localhost:8000)

### No Training Jobs
- Create a training job via the web dashboard: http://localhost:3000
- Or use the script: `python3 ../create_training_job.py`

## 🔄 Updates

To update the app:
1. Stop the current app (right-click menu bar icon → Quit)
2. Rebuild: `./build.sh`
3. Restart: `./build/Constellation`

## 📱 Menu Bar Controls

- **Status**: Shows current state (Idle, Training, Connected, etc.)
- **Job**: Shows current training job name
- **Progress**: Shows training progress percentage
- **Start/Stop Training**: Manual control over training
- **Device Info**: Shows your device name and specs
- **Quit**: Exit the application

The app is designed to run silently in the background and only show up in your menu bar. It will automatically handle training jobs when they're available and keep your device connected to the distributed training network.
