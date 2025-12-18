# Constellation App Distribution Checklist

## For Sharing with Friends (Federated Learning Testing)

### What Your Friend Needs:

1. **macOS** (the app is macOS-only currently)
2. **Python 3.9+** installed
3. **Python packages**: torch, numpy, pandas, requests
4. **Access to the server** (Render URL or local server)

### Current Setup Options:

#### Option A: Git Clone (Recommended for Testing)
1. Friend clones the repo: `git clone https://github.com/varunmitra/project-constellation.git`
2. Installs Python dependencies:
   ```bash
   cd project-constellation
   pip3 install torch numpy pandas requests
   ```
3. Builds/runs the app:
   ```bash
   cd desktop-swift
   ./quick-install.sh
   ```
4. Opens the app and configures server URL

#### Option B: Pre-built App + Scripts
1. You provide:
   - Pre-built Constellation.app
   - Training scripts folder
2. Friend places them in same directory structure
3. Friend installs Python dependencies
4. Runs the app

### What Needs to Be Fixed:

1. **Flexible Project Root Detection** - Currently hardcoded paths
2. **Training Scripts Location** - Should be configurable or bundled
3. **Python Path Detection** - Should handle different Python installations
4. **First-Run Setup** - Guide user through initial configuration

### Recommended Changes:

1. Make `getProjectRoot()` check app bundle first
2. Allow user to specify training scripts location
3. Add setup wizard for first-time users
4. Create installer package (.dmg)
