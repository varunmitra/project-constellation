#!/bin/bash

# Project Constellation - Install Swift Desktop App
# This script builds and installs the Constellation app system-wide

echo "🧠 Installing Project Constellation Desktop App..."

# Build the application
echo "📦 Building application..."
./build.sh

if [ $? -ne 0 ]; then
    echo "❌ Build failed, cannot install"
    exit 1
fi

# Create application bundle
echo "📱 Creating application bundle..."
mkdir -p "Constellation.app/Contents/MacOS"
mkdir -p "Constellation.app/Contents/Resources"

# Copy executable
cp build/Constellation "Constellation.app/Contents/MacOS/Constellation"

# Create Info.plist
cat > "Constellation.app/Contents/Info.plist" << EOF
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

echo "✅ Application bundle created: Constellation.app"
echo ""
echo "🚀 Installation options:"
echo "  1. Run directly: ./Constellation.app/Contents/MacOS/Constellation"
echo "  2. Install to Applications: cp -r Constellation.app /Applications/"
echo "  3. Install to user Applications: cp -r Constellation.app ~/Applications/"
echo ""
echo "📱 The app will appear in your menu bar with a brain icon"
echo "🛑 To quit, right-click the menu bar icon and select 'Quit Constellation'"
