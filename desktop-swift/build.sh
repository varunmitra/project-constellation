#!/bin/bash

# Project Constellation - Build Swift Desktop App
# This script compiles the Swift application into a standalone executable

echo "🧠 Building Project Constellation Desktop App..."

# Check if Swift is available
if ! command -v swift &> /dev/null; then
    echo "❌ Swift is not installed. Please install Xcode Command Line Tools:"
    echo "   xcode-select --install"
    exit 1
fi

# Create build directory
mkdir -p build

# Compile the Swift application
echo "📦 Compiling Swift application..."
swiftc -o build/Constellation \
    -import-objc-header /dev/null \
    -framework AppKit \
    -framework Foundation \
    -framework UserNotifications \
    ConstellationApp.swift

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Executable created: build/Constellation"
    echo ""
    echo "To run the app:"
    echo "  ./build/Constellation"
    echo ""
    echo "To install system-wide:"
    echo "  sudo cp build/Constellation /usr/local/bin/"
    echo "  Constellation"
else
    echo "❌ Build failed!"
    exit 1
fi
