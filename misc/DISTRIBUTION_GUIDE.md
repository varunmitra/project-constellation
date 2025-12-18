# Constellation App Distribution Guide

## ✅ Bundling Approach Implemented

Training scripts are now **automatically bundled** in the app during build!

### How It Works

1. **Build Process**: `quick-install.sh` copies training scripts into app bundle
2. **Runtime**: App checks `Contents/Resources/training/` first
3. **Distribution**: Single app bundle is self-contained (except Python deps)

### For Sharing with Your Friend

#### Option 1: Send Pre-built App (Easiest!)

1. **You build the app:**
   ```bash
   cd desktop-swift
   ./quick-install.sh
   ```

2. **Zip the app:**
   ```bash
   cd /Applications
   zip -r Constellation.app.zip Constellation.app
   ```

3. **Send to friend:**
   - Friend downloads and unzips
   - Friend installs Python dependencies: `pip3 install torch numpy pandas requests`
   - Friend runs the app - **scripts are already bundled!**

#### Option 2: Friend Builds from Source

```bash
git clone https://github.com/varunmitra/project-constellation.git
cd project-constellation/desktop-swift
./quick-install.sh
pip3 install torch numpy pandas requests
```

### What's Bundled

✅ Training scripts (`run_job.py`, `engine.py`, etc.)
✅ Training data files
✅ Everything needed for training

### What Friend Still Needs

⚠️ Python 3.9+ installed
⚠️ Python packages: `pip3 install torch numpy pandas requests`

### Future: True Self-Contained App

To make it truly "download and run", we could bundle:
- Python runtime
- Python dependencies (via venv)
- Everything in one package

But for now, bundling scripts makes distribution much easier!

