#!/bin/bash

# Project Constellation - Start Web Dashboard
# This script starts the React web dashboard

echo "🎨 Starting Project Constellation Web Dashboard..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16 or later."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm."
    exit 1
fi

# Navigate to dashboard directory
cd dashboard

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the development server
echo "🌐 Starting dashboard on http://localhost:3000..."
echo "🛑 Press Ctrl+C to stop the dashboard"
echo ""

npm start
