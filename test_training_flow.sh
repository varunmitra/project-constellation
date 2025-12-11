#!/bin/bash
# Test script to verify training flow

echo "🧪 Testing Constellation Training Flow"
echo "======================================"
echo ""

# Check if server is running
echo "1️⃣ Checking if server is running..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Server is running"
else
    echo "❌ Server is not running. Start it with: python3 server/app.py"
    exit 1
fi

# Check if there's a job
echo ""
echo "2️⃣ Checking for available jobs..."
JOBS=$(curl -s http://localhost:8000/jobs | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data))" 2>/dev/null)
if [ "$JOBS" -gt 0 ]; then
    echo "✅ Found $JOBS job(s)"
else
    echo "⚠️ No jobs found. Create one via the dashboard at http://localhost:3000"
fi

# Check device status
echo ""
echo "3️⃣ Checking device status..."
DEVICE_ID=$(curl -s http://localhost:8000/devices | python3 -c "import sys, json; data=json.load(sys.stdin); print(data[0]['id'] if data else 'none')" 2>/dev/null)
if [ "$DEVICE_ID" != "none" ] && [ ! -z "$DEVICE_ID" ]; then
    echo "✅ Device found: $DEVICE_ID"
    
    # Check for next job
    echo ""
    echo "4️⃣ Checking for next job assignment..."
    RESPONSE=$(curl -s "http://localhost:8000/devices/$DEVICE_ID/next-job" -H "Authorization: Bearer constellation-token")
    JOB_NAME=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('job', {}).get('name', 'None'))" 2>/dev/null)
    ASSIGNMENT_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('assignment_id', 'None'))" 2>/dev/null)
    
    if [ "$JOB_NAME" != "None" ] && [ ! -z "$JOB_NAME" ]; then
        echo "✅ Job available: $JOB_NAME"
        echo "   Assignment ID: $ASSIGNMENT_ID"
    else
        echo "⚠️ No job available for this device"
    fi
else
    echo "⚠️ No devices registered"
fi

# Check Swift app
echo ""
echo "5️⃣ Checking Swift app..."
if ps aux | grep -E "build/Constellation" | grep -v grep > /dev/null; then
    echo "✅ Swift app is running"
    echo "   Check menu bar for Constellation icon"
    echo "   Make sure to click 'Connect to Server' if not connected"
else
    echo "⚠️ Swift app is not running"
    echo "   Start it with: cd desktop-swift && ./build/Constellation"
fi

# Check Python training script
echo ""
echo "6️⃣ Checking Python training script..."
if [ -f "training/run_job.py" ]; then
    echo "✅ Training script exists"
    if python3 training/run_job.py --help > /dev/null 2>&1; then
        echo "✅ Training script is executable"
    else
        echo "⚠️ Training script may have issues (this is expected - it needs a config file)"
    fi
else
    echo "❌ Training script not found"
fi

echo ""
echo "======================================"
echo "📋 Next Steps:"
echo "1. Make sure Swift app is connected (click 'Connect to Server' in menu)"
echo "2. Create a training job via dashboard at http://localhost:3000"
echo "3. The Swift app should automatically pick up the job within 10 seconds"
echo "4. Check logs: tail -f /tmp/swift_app.log"
echo ""

