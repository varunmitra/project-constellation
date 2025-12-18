# Demo Flow Commands

This document contains all commands needed to run the complete demo flow.

## Prerequisites
- Server is running on Render: `https://project-constellation.onrender.com`
- Dashboard options:
  - **Render Dashboard (Recommended):** https://constellation-dashboard.onrender.com/ (no setup needed!)
  - **Local Dashboard:** `http://localhost:3000` (requires local setup)
- Python virtual environment should be activated (for local commands)
- Desktop app will connect to Render server automatically

## Demo Flow Steps

### Step 1: Remove Federated Models from Local
```bash
# Navigate to project root
cd /Users/vmitra/Documents/GitHub/project-constellation

# Remove all federated models
rm -rf federated_models/*.pth
echo "✅ Removed all federated models"
```

### Step 2: Run Inference Script (Show No Results)
```bash
# Try to run inference - should fail or show no model found
python3 invoke_model.py --interactive
# Or with a test text:
python3 invoke_model.py --text "This is a test article about technology"
```

**Expected Result:** Should show "❌ No model found!" or similar error

### Step 3: Verify Server is Running on Render
```bash
# Check server health (no need to start locally - it's on Render)
curl https://project-constellation.onrender.com/health

# Should return: {"status":"healthy",...}
```

**Note:** Server is already running on Render at `https://project-constellation.onrender.com`

### Step 4: Access the Dashboard

**Option A: Use Render Dashboard (Recommended - No Setup!)**
- Open: **https://constellation-dashboard.onrender.com/**
- No terminal needed - works directly in browser!

**Option B: Start Local Dashboard (Terminal 2)**
```bash
# In a new terminal, navigate to project root
cd /Users/vmitra/Documents/GitHub/project-constellation

# Set API URL to point to Render (important!)
export REACT_APP_API_URL=https://project-constellation.onrender.com

# Start the dashboard
./scripts/start-dashboard.sh
# Or manually:
cd dashboard && npm start
```

**Keep this terminal open** - Dashboard will run on `http://localhost:3000` but connect to Render server

### Step 5: Build and Run Desktop App with Terminal Logs (Terminal 3)
```bash
# In a new terminal, navigate to desktop-swift directory
cd /Users/vmitra/Documents/GitHub/project-constellation/desktop-swift

# Build the app (if not already built)
./build.sh

# Option A: Run executable directly (RECOMMENDED - shows all logs)
./build/Constellation

# Option B: Run from app bundle (also shows logs)
./Constellation.app/Contents/MacOS/Constellation

# Option C: Open app bundle AND stream logs separately
# Terminal 3a: open Constellation.app
# Terminal 3b: log stream --predicate 'process == "Constellation"' --style compact
```

**Keep this terminal open** - You'll see all the app logs here including:
- Connection status
- Device registration
- Job polling
- Training progress
- Training output

### Step 6: Create a New Training Job

**Option A: Via Dashboard (Recommended)**
1. Open browser to:
   - **Render Dashboard:** https://constellation-dashboard.onrender.com/ (recommended)
   - **OR Local Dashboard:** http://localhost:3000
2. Navigate to "Jobs" page
3. Click "Create New Job" button
4. Fill in:
   - Name: "Demo Training Job"
   - Model Type: "text_classification"
   - Dataset: "synthetic" (or "ag_news" for real data)
   - Total Epochs: 10 (or more for longer demo)
   - Number of Classes: 4 (for AG News) or leave empty for default
   - Click "Create"

**Option B: Via Script (Terminal 4)**
```bash
# In a new terminal
cd /Users/vmitra/Documents/GitHub/project-constellation

# Activate virtual environment
source venv/bin/activate

# Create a training job (using Render server)
python3 -c "
import requests
import json

SERVER_URL = 'https://project-constellation.onrender.com'
AUTH_HEADER = {'Authorization': 'Bearer constellation-token'}

job_data = {
    'name': 'Demo Training Job',
    'model_type': 'text_classification',
    'dataset': 'synthetic',
    'total_epochs': 10,
    'config': {
        'vocab_size': 10000,
        'seq_length': 100,
        'num_samples': 1000,
        'num_classes': 4,
        'batch_size': 32,
        'learning_rate': 0.001
    }
}

response = requests.post(
    f'{SERVER_URL}/jobs',
    json=job_data,
    headers=AUTH_HEADER,
    timeout=30
)
response.raise_for_status()
print(json.dumps(response.json(), indent=2))
"
```

### Step 7: Start Training via App

The desktop app should automatically:
1. Detect the new job (polls every 10 seconds)
2. Register for the job
3. Start training automatically

**Watch Terminal 3** for logs showing:
- `📋 Received training job: Demo Training Job`
- `🚀 Starting training: Demo Training Job`
- Training progress updates
- `✅ Training completed successfully`

### Step 8: Monitor Activity

**Activity Monitor:**
```bash
# Open Activity Monitor to see CPU/GPU usage
open -a "Activity Monitor"
# Or use command line:
top -pid $(pgrep -f Constellation)
```

**Terminal Logs:**
- Watch Terminal 3 (Desktop App) for training logs
- Watch Terminal 1 (Server) for server-side logs
- Watch Terminal 2 (Dashboard) for dashboard logs

### Step 9: Download and Aggregate Model Locally

**Wait for training to complete**, then:

```bash
# In Terminal 4 (or any terminal)
cd /Users/vmitra/Documents/GitHub/project-constellation

# Activate virtual environment
source venv/bin/activate

# Download and aggregate models
python3 download_and_aggregate.py
```

**Expected Output:**
- Downloads federated updates from server
- Aggregates them using FedAvg
- Saves aggregated model to `federated_models/aggregated_model_<job_id>.pth`

### Step 10: Run Inference Script (Show Trained Model)

```bash
# Run inference with the newly trained model
python3 invoke_model.py --interactive

# Or test with specific text:
python3 invoke_model.py --text "Breaking news about technology and AI developments"
```

**Expected Result:**
- Should successfully load the aggregated model
- Show predictions with confidence scores
- Display class probabilities

## Quick Reference: All Commands in Order

```bash
# Terminal 1: Verify Render Server
curl https://project-constellation.onrender.com/health

# Terminal 2: Dashboard (local, connects to Render)
cd /Users/vmitra/Documents/GitHub/project-constellation
export REACT_APP_API_URL=https://project-constellation.onrender.com
./scripts/start-dashboard.sh

# Terminal 3: Desktop App
cd /Users/vmitra/Documents/GitHub/project-constellation/desktop-swift
./build.sh  # Only needed first time
./build/Constellation
# App will auto-connect to Render server

# Terminal 4: Commands
cd /Users/vmitra/Documents/GitHub/project-constellation
source venv/bin/activate

# Step 1: Clean up
rm -rf federated_models/*.pth

# Step 2: Test inference (should fail)
python3 invoke_model.py --text "Test"

# Step 6: Create job (pointing to Render)
python3 -c "
import requests, json
response = requests.post(
    'https://project-constellation.onrender.com/jobs',
    json={
        'name': 'Demo Training Job',
        'model_type': 'text_classification',
        'dataset': 'synthetic',
        'total_epochs': 10,
        'config': {
            'vocab_size': 10000,
            'seq_length': 100,
            'num_samples': 1000,
            'num_classes': 4,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    },
    headers={'Authorization': 'Bearer constellation-token'},
    timeout=30
)
print(json.dumps(response.json(), indent=2))
"

# Step 9: Download and aggregate (after training completes)
python3 download_and_aggregate.py

# Step 10: Test inference (should work)
python3 invoke_model.py --interactive
```

## Troubleshooting

### Desktop App Not Connecting
- Check Render server is running: `curl https://project-constellation.onrender.com/health`
- Check server URL in app settings (should be `https://project-constellation.onrender.com`)
- Check auth token matches: `constellation-token`
- Note: Render free tier may have cold starts (first request takes 30-60 seconds)

### Training Not Starting
- Check device is registered: `curl https://project-constellation.onrender.com/devices -H "Authorization: Bearer constellation-token"`
- Check job status: `curl https://project-constellation.onrender.com/jobs -H "Authorization: Bearer constellation-token"`
- Check app logs in Terminal 3
- Wait for Render cold start if server was idle

### Dashboard Not Connecting to Render
- Set environment variable: `export REACT_APP_API_URL=https://project-constellation.onrender.com`
- Or edit `dashboard/src/context/AppContext.js` to use Render URL
- Restart dashboard after changing

### Model Not Found After Aggregation
- Check `federated_models/` directory: `ls -la federated_models/`
- Verify aggregation completed successfully
- Try specifying model path: `python3 invoke_model.py --model federated_models/aggregated_model_<job_id>.pth --text "Test"`

