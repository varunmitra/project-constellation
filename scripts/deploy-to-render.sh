#!/bin/bash

# Deployment script for Render.com
# This script helps prepare the project for Render deployment

echo "🚀 Preparing Project Constellation for Render Deployment"
echo "========================================================="

# Check if render.yaml exists
if [ ! -f "render.yaml" ]; then
    echo "❌ render.yaml not found!"
    echo "Creating render.yaml..."
    # The file should already exist, but if not, we'll create it
fi

echo ""
echo "📋 Pre-deployment Checklist:"
echo ""

# Check for required files
echo "Checking required files..."
files=("Dockerfile.server" "docker-compose.yml" "requirements.txt" "server/app.py")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing!"
    fi
done

echo ""
echo "🔧 Environment Setup:"
echo ""
echo "1. Create a Render account at https://render.com"
echo "2. Connect your GitHub repository"
echo "3. Render will auto-detect render.yaml"
echo "4. Review and deploy!"
echo ""
echo "📝 Manual Steps:"
echo ""
echo "1. Go to Render Dashboard"
echo "2. Click 'New +' → 'Blueprint'"
echo "3. Connect your GitHub repository"
echo "4. Select 'render.yaml'"
echo "5. Click 'Apply'"
echo ""
echo "✨ Render will automatically:"
echo "   - Create PostgreSQL database"
echo "   - Deploy server service"
echo "   - Deploy dashboard static site"
echo "   - Set up environment variables"
echo ""
echo "💰 Cost: FREE (Render free tier)"
echo ""
echo "🔗 After deployment:"
echo "   - Update Swift app with server URL"
echo "   - Test device registration"
echo "   - Monitor in Render dashboard"
echo ""


