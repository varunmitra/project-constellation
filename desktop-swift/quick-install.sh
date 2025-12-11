#!/bin/bash

# Project Constellation - Quick Install Script
# This script builds and installs the Constellation app to your Applications folder

echo "🧠 Project Constellation - Quick Install"
echo "========================================"

# Build the app
echo "📦 Building application..."
./build.sh

if [ $? -ne 0 ]; then
    echo "❌ Build failed, cannot install"
    exit 1
fi

# Create app bundle
echo "📱 Creating application bundle..."
mkdir -p "Constellation.app/Contents/MacOS"
mkdir -p "Constellation.app/Contents/Resources"

# Copy executable
cp build/Constellation "Constellation.app/Contents/MacOS/Constellation"

# Create Info.plist
cat > "Constellation.app/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>Constellation</string>
    <key>CFBundleIdentifier</key>
    <string>com.constellation.app</string>
    <key>CFBundleName</key>
    <string>Constellation</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Make executable
chmod +x "Constellation.app/Contents/MacOS/Constellation"

# Install to Applications
echo "📥 Installing to Applications folder..."
if [ -d "/Applications/Constellation.app" ]; then
    echo "⚠️  Removing existing installation..."
    rm -rf "/Applications/Constellation.app"
fi

cp -r "Constellation.app" "/Applications/"

echo "✅ Installation complete!"
echo ""
echo "🚀 Constellation is now installed in your Applications folder"
echo "📱 Look for the brain icon in your menu bar (top-right)"
echo "🛑 To quit, right-click the menu bar icon and select 'Quit Constellation'"
echo ""
echo "📊 Make sure the central server is running:"
echo "   cd ../server && python3 app.py"
echo ""
echo "🌐 Web dashboard: http://localhost:3000"
