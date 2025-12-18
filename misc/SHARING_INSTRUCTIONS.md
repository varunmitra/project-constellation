# Sharing Constellation App for Federated Learning

## Quick Start Guide for Your Friend

### Prerequisites
1. **macOS** (10.15+)
2. **Python 3.9+** installed (`python3 --version`)
3. **Git** (for cloning the repo)

### Setup Steps

#### 1. Clone the Repository
```bash
git clone https://github.com/varunmitra/project-constellation.git
cd project-constellation
```

#### 2. Install Python Dependencies
```bash
pip3 install torch numpy pandas requests
```

#### 3. Build and Install the App
```bash
cd desktop-swift
./quick-install.sh
```

#### 4. Configure the App
1. Open **Constellation** from Applications
2. Click the brain icon in the menu bar
3. Click **"Configure Server"** and enter: `https://project-constellation.onrender.com`
4. (Optional) Click **"Configure Scripts Path"** if training scripts aren't auto-detected
5. Click **"Connect to Server"**
6. Click **"Start Training"**

### What Changed for Sharing

✅ **Server URL**: Configurable via UI (already works)
✅ **Training Scripts Path**: Now configurable via menu
✅ **Project Root Detection**: Checks multiple locations including:
   - App bundle (if scripts are bundled)
   - User-specified path
   - Common GitHub locations
   - Current directory and parent directories

### Distribution Options

#### Option A: Git Clone (Recommended)
- Friend clones repo
- Installs dependencies
- Builds app
- **Pros**: Always up-to-date, full source access
- **Cons**: Requires Git and build tools

#### Option B: Pre-built App + Scripts
1. Build the app: `cd desktop-swift && ./quick-install.sh`
2. Copy `Constellation.app` and `training/` folder
3. Friend places them together
4. Friend installs Python dependencies
5. Friend runs app and configures scripts path if needed

#### Option C: Bundle Everything (Future)
- Include training scripts in app bundle
- Include Python dependencies
- Single download and run

### Troubleshooting

**"Training script not found"**
- Use menu → "Configure Scripts Path" to point to project root
- Make sure `training/run_job.py` exists in that directory

**"Python not found"**
- Install Python: `brew install python3` or download from python.org
- Verify: `python3 --version`

**"Cannot connect to server"**
- Check server URL is correct
- Verify internet connection
- Check if Render server is running

### Testing Federated Learning

1. Both you and your friend run the app
2. Create a training job on the server
3. Both devices will pick up the job
4. Training runs in parallel
5. Models are aggregated on the server

