#!/bin/bash

# Project Constellation Setup Script

echo "🚀 Setting up Project Constellation..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Install Node.js dependencies for dashboard
echo "📦 Installing Node.js dependencies..."
cd dashboard
npm install
cd ..

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p training/checkpoints
mkdir -p federated/federated_models
mkdir -p checkpoints

# Set permissions
echo "🔧 Setting permissions..."
chmod +x scripts/*.sh

echo "✅ Setup complete!"
echo ""
echo "To start the system:"
echo "1. Start server: ./scripts/start-server.sh"
echo "2. Start training: ./scripts/start-training.sh"
echo "3. Start dashboard: ./scripts/start-dashboard.sh"
echo ""
echo "Or run all at once: ./scripts/setup.sh"
