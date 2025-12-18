#!/bin/bash
# Phase 1 Testing Script for Project Constellation
# Run this script to test all Phase 1 features

set -e

echo "=========================================="
echo "Project Constellation - Phase 1 Testing"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SERVER_URL="http://localhost:8000"

# Test 1: Server Health Check
echo -e "${YELLOW}Test 1: Server Health Check${NC}"
if curl -s "$SERVER_URL/health" > /dev/null; then
    echo -e "${GREEN}✅ Server is running${NC}"
    curl -s "$SERVER_URL/health" | python3 -m json.tool
else
    echo -e "${RED}❌ Server is not running. Start it with: python3 server/app.py${NC}"
    exit 1
fi
echo ""

# Test 2: Job Creation
echo -e "${YELLOW}Test 2: Job Creation${NC}"
JOB_RESPONSE=$(curl -s -X POST "$SERVER_URL/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Job - Phase 1",
    "model_type": "text_classification",
    "dataset": "ag_news",
    "total_epochs": 3,
    "config": {
      "vocab_size": 10000,
      "seq_length": 100,
      "num_samples": 1000,
      "num_classes": 4,
      "batch_size": 32,
      "learning_rate": 0.001,
      "epochs": 3
    }
  }')

JOB_ID=$(echo "$JOB_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])" 2>/dev/null || echo "")

if [ -n "$JOB_ID" ]; then
    echo -e "${GREEN}✅ Job created successfully${NC}"
    echo "Job ID: $JOB_ID"
    echo "$JOB_RESPONSE" | python3 -m json.tool
else
    echo -e "${RED}❌ Job creation failed${NC}"
    echo "$JOB_RESPONSE"
fi
echo ""

# Test 3: Device Registration
echo -e "${YELLOW}Test 3: Device Registration${NC}"
DEVICE_RESPONSE=$(curl -s -X POST "$SERVER_URL/devices/register" \
  -H "Content-Type: application/json" \
  -H "User-Agent: Constellation-Swift/1.0" \
  -H "X-Constellation-Client: swift-app" \
  -d '{
    "name": "Test Device - Phase 1",
    "device_type": "macbook",
    "os_version": "15.1.0",
    "cpu_cores": 8,
    "memory_gb": 16,
    "gpu_available": true,
    "gpu_memory_gb": 8
  }')

DEVICE_ID=$(echo "$DEVICE_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])" 2>/dev/null || echo "")

if [ -n "$DEVICE_ID" ]; then
    echo -e "${GREEN}✅ Device registered successfully${NC}"
    echo "Device ID: $DEVICE_ID"
    echo "$DEVICE_RESPONSE" | python3 -m json.tool
else
    echo -e "${RED}❌ Device registration failed${NC}"
    echo "$DEVICE_RESPONSE"
fi
echo ""

# Test 4: Intelligent Job Distribution
if [ -n "$JOB_ID" ] && [ -n "$DEVICE_ID" ]; then
    echo -e "${YELLOW}Test 4: Intelligent Job Distribution${NC}"
    START_RESPONSE=$(curl -s -X POST "$SERVER_URL/jobs/$JOB_ID/start")
    
    if echo "$START_RESPONSE" | grep -q "assigned_device"; then
        echo -e "${GREEN}✅ Job distribution working${NC}"
        echo "$START_RESPONSE" | python3 -m json.tool
    else
        echo -e "${YELLOW}⚠️ Job distribution response:${NC}"
        echo "$START_RESPONSE" | python3 -m json.tool
    fi
    echo ""
fi

# Test 5: List Devices
echo -e "${YELLOW}Test 5: List Devices${NC}"
DEVICES=$(curl -s "$SERVER_URL/devices")
DEVICE_COUNT=$(echo "$DEVICES" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
echo -e "${GREEN}✅ Found $DEVICE_COUNT device(s)${NC}"
echo "$DEVICES" | python3 -m json.tool | head -20
echo ""

# Test 6: List Jobs
echo -e "${YELLOW}Test 6: List Jobs${NC}"
JOBS=$(curl -s "$SERVER_URL/jobs")
JOB_COUNT=$(echo "$JOBS" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
echo -e "${GREEN}✅ Found $JOB_COUNT job(s)${NC}"
echo "$JOBS" | python3 -m json.tool | head -30
echo ""

# Test 7: WebSocket Endpoint (check if it exists)
echo -e "${YELLOW}Test 7: WebSocket Endpoint${NC}"
echo "WebSocket endpoint should be available at: ws://localhost:8000/ws"
echo "Test this manually in browser console:"
echo "  const ws = new WebSocket('ws://localhost:8000/ws');"
echo "  ws.onopen = () => console.log('✅ WebSocket connected');"
echo ""

echo "=========================================="
echo -e "${GREEN}Phase 1 API Tests Complete${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start dashboard: cd dashboard && npm start"
echo "2. Open http://localhost:3000 in browser"
echo "3. Check browser console for WebSocket connection"
echo "4. Test job creation from dashboard UI"
echo "5. Build and run Swift app: cd desktop-swift && ./build.sh"

