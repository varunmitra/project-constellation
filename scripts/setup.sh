#!/bin/bash

# Project Constellation - Complete Setup Script
# This script sets up the entire distributed AI training infrastructure

echo "🌟 Setting up Project Constellation - Decentralized AI Training Infrastructure"
echo "=================================================================================="

# Check system requirements
echo "🔍 Checking system requirements..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
else
    echo "✅ Python 3 found: $(python3 --version)"
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16 or later."
    exit 1
else
    echo "✅ Node.js found: $(node --version)"
fi

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm."
    exit 1
else
    echo "✅ npm found: $(npm --version)"
fi

# Check Xcode Command Line Tools (for macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v xcodebuild &> /dev/null; then
        echo "⚠️  Xcode Command Line Tools not found. Please install them for the desktop app:"
        echo "   xcode-select --install"
    else
        echo "✅ Xcode Command Line Tools found"
    fi
fi

echo ""
echo "📦 Setting up Python environment..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "🎨 Setting up web dashboard..."

# Setup dashboard
cd dashboard
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi
cd ..

echo ""
echo "📁 Creating necessary directories..."

# Create directories
mkdir -p models
mkdir -p checkpoints
mkdir -p logs
mkdir -p data

echo ""
echo "🔧 Making scripts executable..."

# Make scripts executable
chmod +x scripts/*.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start the system:"
echo "   1. Start the central server:    ./scripts/start-server.sh"
echo "   2. Start the web dashboard:     ./scripts/start-dashboard.sh"
echo "   3. Start training engine:       ./scripts/start-training.sh"
echo "   4. Open desktop app:            Open desktop/Constellation.xcodeproj in Xcode"
echo ""
echo "🌐 Access points:"
echo "   - Web Dashboard: http://localhost:3000"
echo "   - API Server:    http://localhost:8000"
echo "   - API Docs:      http://localhost:8000/docs"
echo ""
echo "📚 For more information, see README.md"
