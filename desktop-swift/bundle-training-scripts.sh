#!/bin/bash
# Helper script to bundle training scripts into app

APP_BUNDLE="Constellation.app"
TRAINING_SRC="../training"
TRAINING_DEST="$APP_BUNDLE/Contents/Resources/training"

if [ ! -d "$APP_BUNDLE" ]; then
    echo "❌ App bundle not found: $APP_BUNDLE"
    exit 1
fi

if [ ! -d "$TRAINING_SRC" ]; then
    echo "❌ Training scripts not found: $TRAINING_SRC"
    exit 1
fi

echo "📦 Bundling training scripts into app..."
mkdir -p "$TRAINING_DEST"
cp -r "$TRAINING_SRC"/* "$TRAINING_DEST/" 2>/dev/null || true

# Verify
if [ -f "$TRAINING_DEST/run_job.py" ]; then
    echo "✅ Training scripts bundled successfully"
    echo "   Location: $TRAINING_DEST"
else
    echo "⚠️  Warning: run_job.py not found after bundling"
fi
