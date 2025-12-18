# Bundling Training Scripts in App

## Approach: Bundle Everything

The app now bundles training scripts directly in the app bundle, making it easy to distribute.

### How It Works

1. **Build Process**: `quick-install.sh` automatically copies training scripts into `Constellation.app/Contents/Resources/training/`
2. **Runtime**: App checks app bundle first for training scripts
3. **Distribution**: Single app bundle contains everything needed

### Building for Distribution

```bash
cd desktop-swift
./quick-install.sh
```

This will:
- Build the Swift app
- Copy training scripts into app bundle
- Install to Applications folder

### What Gets Bundled

- `training/run_job.py` - Main training script
- `training/engine.py` - Training engine
- `training/ag_news_trainer.py` - AG News trainer
- `training/data/` - Training data files

### For Your Friend

**Option 1: Pre-built App (Easiest)**
1. You build the app: `cd desktop-swift && ./quick-install.sh`
2. You zip `Constellation.app` and send it
3. Friend installs Python dependencies: `pip3 install torch numpy pandas requests`
4. Friend runs the app - scripts are already bundled!

**Option 2: Git Clone**
- Friend clones repo and builds themselves
- Same process, scripts get bundled automatically

### Python Dependencies

Friend still needs to install Python packages:
```bash
pip3 install torch numpy pandas requests
```

### Future: Bundle Python Too

For true "download and run", we could:
1. Bundle Python runtime in app
2. Create venv with dependencies
3. Point to bundled Python instead of system Python

This would make it truly self-contained!

