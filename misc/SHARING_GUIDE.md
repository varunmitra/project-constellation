# Sharing Constellation App for Federated Learning

## Overview
To share the Constellation app with a friend for true federated learning testing, you need to:

1. **Package the app** with all dependencies
2. **Ensure training scripts are accessible**
3. **Configure server URL** (already supported via UI)
4. **Provide setup instructions**

## Current Status

### ✅ Already Works:
- Server URL is configurable via UI (menu bar → Configure Server)
- Device registration is automatic
- Training scripts are referenced relatively

### ⚠️ Needs Changes:
- Project root detection assumes specific paths
- Python dependencies need to be documented
- Training scripts need to be bundled or accessible

## Required Changes

### 1. Make Project Root Detection More Flexible

The app currently looks for training scripts in hardcoded paths. We should:
- Bundle training scripts with the app, OR
- Make project root detection more flexible, OR
- Allow users to specify training script location

### 2. Bundle Training Scripts

Option A: Include training scripts in app bundle
Option B: Download from GitHub on first run
Option C: Allow user to specify location

### 3. Python Dependencies

Friend needs:
- Python 3.9+ installed
- Required packages: torch, numpy, pandas, requests, etc.

## Recommended Approach

### Option 1: Bundle Everything (Easiest for Users)
- Include training scripts in app bundle
- Include Python dependencies (via venv or bundled Python)
- User just needs to download and run

### Option 2: Git Clone + App (Current Approach)
- Friend clones the repo
- Installs Python dependencies
- Runs the app (app finds scripts via getProjectRoot)

### Option 3: Hybrid
- App downloads training scripts on first run
- User specifies Python location
- More flexible but requires internet

## Next Steps

Which approach would you prefer? I can implement:
1. Bundle training scripts in app
2. Make project root detection more flexible
3. Add first-run setup wizard
4. Create distribution package
