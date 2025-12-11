#!/bin/bash

# Project Constellation - Start Training Engine
# This script starts the distributed training engine

echo "🧠 Starting Project Constellation Training Engine..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p checkpoints
mkdir -p logs

# Start the training engine
echo "🚀 Starting training engine..."
echo "🛑 Press Ctrl+C to stop the training engine"
echo ""

python training/engine.py
