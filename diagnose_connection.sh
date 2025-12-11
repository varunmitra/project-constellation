#!/bin/bash
echo "🔍 Constellation Connection Diagnostics"
echo "======================================"
echo ""

echo "1️⃣ Checking if server is accessible..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Server is running and accessible"
else
    echo "❌ Server is not accessible"
    exit 1
fi

echo ""
echo "2️⃣ Checking if Swift app is running..."
if ps aux | grep -E "build/Constellation" | grep -v grep > /dev/null; then
    PID=$(ps aux | grep -E "build/Constellation" | grep -v grep | awk '{print $2}')
    echo "✅ Swift app is running (PID: $PID)"
else
    echo "❌ Swift app is not running"
    exit 1
fi

echo ""
echo "3️⃣ Checking recent Swift app logs..."
echo "   Looking for connection attempts..."
RECENT=$(tail -100 /tmp/swift_app.log | grep -E "(Connect|Connected|Device registered|Testing connection)" | tail -5)
if [ -z "$RECENT" ]; then
    echo "⚠️ No connection attempts found in logs"
    echo "   This means you haven't clicked 'Connect to Server' yet"
else
    echo "$RECENT"
fi

echo ""
echo "4️⃣ Testing server connection manually..."
RESPONSE=$(curl -s http://localhost:8000/health)
if [ "$RESPONSE" == "OK" ]; then
    echo "✅ Server health check passed"
else
    echo "⚠️ Server health check returned: $RESPONSE"
fi

echo ""
echo "5️⃣ Checking for available jobs..."
JOBS=$(curl -s http://localhost:8000/jobs | python3 -c "import sys, json; data=json.load(sys.stdin); print(len([j for j in data if j.get('status') == 'pending']))" 2>/dev/null)
if [ "$JOBS" -gt 0 ]; then
    echo "✅ Found $JOBS pending job(s) waiting"
else
    echo "⚠️ No pending jobs found"
fi

echo ""
echo "======================================"
echo "📋 Action Required:"
echo ""
echo "Please click 'Connect to Server' in the Constellation menu bar app."
echo "Then run this script again to verify connection."
echo ""
echo "To monitor logs in real-time:"
echo "  tail -f /tmp/swift_app.log"
echo ""

