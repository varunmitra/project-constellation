#!/bin/bash

# Project Constellation - Start Central Server
# This script starts the central server for the distributed AI training infrastructure

echo "🚀 Starting Project Constellation Central Server..."

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
mkdir -p models
mkdir -p checkpoints
mkdir -p logs

# Start the server
echo "🌐 Starting server on http://localhost:8000..."
echo "📊 Dashboard will be available at http://localhost:3000"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

python server/app.py
