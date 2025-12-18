# Configuring Server URL in Swift App

## Default Behavior

The Swift app **defaults to Render server**: `https://project-constellation.onrender.com`

## How Server URL Works

1. **Default**: If no URL is saved, app uses Render
2. **Saved URL**: App saves your last used URL in UserDefaults
3. **Training Scripts**: Receive server URL from Swift app via job config

## To Use Render Server (Recommended)

### Option 1: Via App UI (Easiest)
1. Run the app: `./build/Constellation`
2. Look for menu item: **"Server"** or **"Configure Server"**
3. Set URL to: `https://project-constellation.onrender.com`
4. Click **"Test Connection"** to verify
5. Save settings

### Option 2: Reset to Default
If app is stuck on localhost, reset it:

```bash
# Clear saved server URL (forces app to use Render default)
defaults delete com.project.constellation ConstellationServerURL 2>/dev/null

# Or check what's currently saved:
defaults read com.project.constellation ConstellationServerURL
```

Then restart the app - it will use Render default.

### Option 3: Set via Command Line
```bash
# Set to Render
defaults write com.project.constellation ConstellationServerURL "https://project-constellation.onrender.com"

# Set to localhost (for local development)
defaults write com.project.constellation ConstellationServerURL "http://localhost:8000"
```

## Verify Current Setting

```bash
defaults read com.project.constellation ConstellationServerURL || echo "Using default: https://project-constellation.onrender.com"
```

## Important Notes

- **Building** (`./build.sh` or `swiftc`) doesn't change the server URL
- Server URL is stored in **UserDefaults** (persists between app restarts)
- Training scripts automatically receive the correct server URL from the Swift app
- The app code defaults to Render (see `ConstellationApp.swift` line 437)

