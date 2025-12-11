#!/bin/bash
# Quick test script for federated learning

echo "🧪 Quick Federated Learning Test"
echo "=================================="
echo ""

# Check if server is running
echo "1️⃣ Checking server..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Server is running"
else
    echo "❌ Server is not running. Start it with: python3 server/app.py"
    exit 1
fi

# Check if we have completed jobs
echo ""
echo "2️⃣ Checking for completed jobs..."
JOBS=$(curl -s http://localhost:8000/jobs -H "Authorization: Bearer constellation-token" | python3 -c "import sys, json; data=json.load(sys.stdin); print(len([j for j in data if j.get('status') == 'completed']))" 2>/dev/null)
echo "   Found $JOBS completed job(s)"

# Check for federated updates
echo ""
echo "3️⃣ Checking for federated updates..."
if [ -d "federated_updates" ]; then
    UPDATE_COUNT=$(ls -1 federated_updates/*.json 2>/dev/null | wc -l)
    echo "   Found $UPDATE_COUNT update file(s)"
    if [ "$UPDATE_COUNT" -gt 0 ]; then
        echo "   ✅ Model weights have been uploaded"
    else
        echo "   ⚠️ No updates found yet"
    fi
else
    echo "   ⚠️ federated_updates directory doesn't exist yet"
fi

# Check for aggregated models
echo ""
echo "4️⃣ Checking for aggregated models..."
if [ -d "federated_models" ]; then
    MODEL_COUNT=$(ls -1 federated_models/*.pth 2>/dev/null | wc -l)
    echo "   Found $MODEL_COUNT aggregated model(s)"
    if [ "$MODEL_COUNT" -gt 0 ]; then
        echo "   ✅ Aggregated models exist"
        ls -lh federated_models/*.pth 2>/dev/null | head -3
    else
        echo "   ⚠️ No aggregated models yet"
    fi
else
    echo "   ⚠️ federated_models directory doesn't exist yet"
fi

echo ""
echo "=================================="
echo "📋 To test federated learning:"
echo "   1. Run: python3 test_federated_learning.py"
echo "   2. Or manually:"
echo "      - Create a job via dashboard"
echo "      - Have multiple devices train on it"
echo "      - Call: POST /federated/aggregate/{job_id}"
echo ""

