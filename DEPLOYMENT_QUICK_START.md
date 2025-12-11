# Quick Deployment Guide - Render.com

## 🚀 5-Minute Deployment

### Step 1: Prepare Your Repository
1. Ensure all code is committed and pushed to GitHub
2. Verify `render.yaml` exists in the root directory
3. Check that `Dockerfile.server` is present
4. **Note**: The `render.yaml` file includes region settings - if deploying manually, ensure web service and database use the same region!

### Step 2: Create Render Account & Link GitHub
1. Go to [render.com](https://render.com)
2. Sign up with GitHub OAuth (click **"Sign up with GitHub"**)
3. **Important**: When authorizing Render:
   - Select **"Authorize Render"** to grant access
   - Choose **"All repositories"** OR specifically select `project-constellation`
   - Click **"Authorize"** to complete the connection
4. Verify connection:
   - Go to Render dashboard → **Account Settings** → **Connected Accounts**
   - You should see GitHub listed as connected
5. If linking fails, see **"GitHub Repository Linking Issues"** in Troubleshooting section below

### Step 3: Deploy Database
1. In Render dashboard, click **"New +"** → **"PostgreSQL"**
2. Name: `constellation-db`
3. **Region**: Select a region (e.g., `Oregon (US West)`, `Frankfurt (EU Central)`, etc.)
   - **Important**: Note which region you select - your web service should use the SAME region!
4. Plan: **Free** (for testing) or **Starter** ($7/month for production)
5. Database: `constellation`
6. User: `constellation_user`
7. Click **"Create Database"**
8. **Copy the Internal Database URL** (you'll need this) dpg-d4ticrur433s73cig450-a

### Step 4: Deploy Web Service
1. In Render dashboard, click **"New +"** → **"Web Service"**
2. **Connect your GitHub repository**:
   - Click **"Connect GitHub"** if you see this option
   - Search for `varunmitra/project-constellation` or select it from the list
   - If repository doesn't appear, see Troubleshooting section below
3. **Select Region** (IMPORTANT):
   - Choose the **SAME region** you used for the database in Step 3
   - This ensures low latency between your web service and database
   - Common regions: `Oregon (US West)`, `Frankfurt (EU Central)`, `Singapore (Asia Pacific)`
   - If you don't select a region, Render will scroll back to this field when you click "Create"
4. **If `render.yaml` is NOT auto-detected**, configure manually:

   **Option A: Use Docker (Recommended)**
   - **Name**: `constellation-server`
   - **Language/Environment**: Change from "Python 3" to **"Docker"**
   - **Dockerfile Path**: `./Dockerfile.server`
   - **Build Command**: Leave empty (Docker handles it)
   - **Start Command**: `python server/app.py`
   - **Plan**: Free (testing) or Starter ($7/month)

   **Option B: Use Python 3 (If Docker doesn't work)**
   - **Name**: `constellation-server`
   - **Language**: Keep "Python 3"
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python server/app.py` or `uvicorn server.app:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free (testing) or Starter ($7/month)
6. Add environment variables:
   - `DATABASE_URL`: Use the database connection string from Step 3
   - `PORT`: `8000`
   - `PYTHONPATH`: `/app`
   - `ENVIRONMENT`: `production`
   - `ALLOWED_ORIGINS`: `*` (for testing) or your domain (production)
7. **Before clicking "Create Web Service"**: Scroll up and verify the **Region** is selected (same as database)
8. Click **"Create Web Service"**
9. Wait for build to complete (~5-10 minutes)

### Step 5: Deploy Dashboard (Static Site)

**⚠️ IMPORTANT: Make sure you select "Static Site" NOT "Web Service"!**

1. In Render dashboard, click **"New +"** → **"Static Site"** (NOT "Web Service")
   - If you see "Web Service" selected, cancel and start over
   - The page title should say "Create a new Static Site"

2. Connect your GitHub repository
   - Click **"Connect GitHub"** if needed
   - Select `varunmitra/project-constellation`
   - Select branch: `main`

3. Configure settings:
   - **Name**: `constellation-dashboard`
   - **Branch**: `main`
   - **Root Directory**: `dashboard` ⚠️ **CRITICAL**: Set this to `dashboard` to avoid Python auto-detection!
     - This tells Render to use the `dashboard/` folder as the root
     - Render won't see `requirements.txt` in the parent directory
   - **Build Command**: `npm install && npm run build`
     - ⚠️ Should use `npm`, NOT `pip` or Python commands
     - Note: No `cd dashboard` needed since Root Directory is already `dashboard`
   - **Publish Directory**: `build`
     - ⚠️ Should point to React build output (relative to Root Directory)
     - Since Root Directory is `dashboard`, use `build` not `dashboard/build`

4. Add Environment Variable:
   - Click **"Add Environment Variable"**
   - **Key**: `REACT_APP_API_URL`
   - **Value**: Your web service URL (e.g., `https://project-constellation.onrender.com`)
   - ⚠️ Use `https://` and your actual service name

5. **Verify Before Creating:**
   - ✅ Service type shows "Static Site" (not "Web Service")
   - ✅ Build Command uses `npm` (not `pip` or Python)
   - ✅ Publish Directory is `dashboard/build` (not Python paths)
   - ✅ No Python/requirements.txt references

6. Click **"Create Static Site"**
7. Wait for build (~3-5 minutes)
   - Should see npm installing packages, NOT pip/Python

### Step 6: Verify Deployment
1. **Test Server**:
   - Visit: `https://constellation-server.onrender.com/health`
   - Should return: `{"status": "healthy", "timestamp": "..."}`
   - Visit: `https://constellation-server.onrender.com/` 
   - Should return: `{"message": "Project Constellation Server", "version": "1.0.0"}`

2. **Test Dashboard**:
   - Visit your static site URL
   - Should load the React dashboard

3. **Test WebSocket**:
   - Use a WebSocket client to connect to `wss://constellation-server.onrender.com/ws`
   - Should connect successfully

### Step 7: Update Dashboard API URL
1. Go to your Static Site settings in Render
2. Update `REACT_APP_API_URL` environment variable to your web service URL
3. Redeploy the static site

---

## 🎯 What Happens After Deployment

### Immediate Next Steps

1. **Verify Services Are Running**
   - Check Render dashboard - all services should show "Live" status
   - Web service URL: `https://project-constellation.onrender.com` (or your custom name)
   - Dashboard URL: `https://constellation-dashboard.onrender.com` (or your custom name)

2. **Test Core Functionality**
   ```bash
   # Health check
   curl https://project-constellation.onrender.com/health
   
   # List devices (should be empty initially)
   curl https://project-constellation.onrender.com/devices
   
   # List jobs (should be empty initially)
   curl https://project-constellation.onrender.com/jobs
   
   # List models
   curl https://project-constellation.onrender.com/models
   ```

3. **Access the Dashboard**
   - **If dashboard is deployed**: Open your dashboard URL in a browser
     - URL format: `https://constellation-dashboard.onrender.com` (or your custom name)
     - Find the URL in Render dashboard → Static Sites → constellation-dashboard
   - **If dashboard is NOT deployed yet**: See "Deploying the Dashboard" section below
   - You should see the React interface with:
     - Devices page (empty initially)
     - Jobs page (empty initially)
     - Models page
     - Settings page

---

## 🖥️ Accessing the Dashboard

### Option 1: Dashboard Already Deployed on Render

If you've already deployed the dashboard as a Static Site:

1. **Find Your Dashboard URL**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click on **"Static Sites"** in the left sidebar
   - Find your `constellation-dashboard` service
   - The URL will be displayed (e.g., `https://constellation-dashboard.onrender.com`)

2. **Open in Browser**
   - Click the URL or copy-paste it into your browser
   - The dashboard should load automatically

3. **Verify It's Working**
   - You should see the Constellation dashboard interface
   - Check browser console (F12) for any errors
   - If you see "Cannot connect to API", see troubleshooting below

### Option 2: Deploy Dashboard Now (If Not Deployed)

If you haven't deployed the dashboard yet:

1. **Go to Render Dashboard**
   - Visit [dashboard.render.com](https://dashboard.render.com)
   - Make sure you're logged in

2. **Create Static Site**
   - Click **"New +"** button (top right)
   - Select **"Static Site"**

3. **Connect Repository**
   - Click **"Connect GitHub"** if needed
   - Select `varunmitra/project-constellation` repository
   - Click **"Connect"**

4. **Configure Build Settings**
   - **Name**: `constellation-dashboard`
   - **Branch**: `main`
   - **Root Directory**: Leave empty (or set to `.`)
   - **Build Command**: `cd dashboard && npm install && npm run build`
   - **Publish Directory**: `dashboard/build`

5. **Set Environment Variable**
   - Scroll to **"Environment Variables"** section
   - Click **"Add Environment Variable"**
   - **Key**: `REACT_APP_API_URL`
   - **Value**: Your web service URL (e.g., `https://project-constellation.onrender.com`)
   - **Important**: Use `https://` and include your actual service name

6. **Create Static Site**
   - Click **"Create Static Site"** button
   - Wait for build to complete (~3-5 minutes)

7. **Access Dashboard**
   - Once build completes, Render will show your dashboard URL
   - Click the URL or copy it
   - Open in browser

### Option 3: Run Dashboard Locally (Development)

For local development/testing:

1. **Navigate to Dashboard Directory**
   ```bash
   cd dashboard
   ```

2. **Install Dependencies** (first time only)
   ```bash
   npm install
   ```

3. **Set API URL**
   ```bash
   # For local development (pointing to local server)
   export REACT_APP_API_URL=http://localhost:8000
   
   # OR for testing with Render deployment
   export REACT_APP_API_URL=https://project-constellation.onrender.com
   ```

4. **Start Development Server**
   ```bash
   npm start
   ```

5. **Open Dashboard**
   - Browser should automatically open to `http://localhost:3000`
   - Or manually visit: `http://localhost:3000`

### Dashboard Features

Once the dashboard loads, you'll see:

1. **Devices Page**
   - Shows all registered devices (Swift apps, etc.)
   - Device status, specs, last seen time
   - Currently empty until devices connect

2. **Jobs Page**
   - Lists all training jobs
   - Job status, progress, assigned devices
   - Create new jobs from here

3. **Models Page**
   - Browse available models
   - Download models
   - Upload new models

4. **Settings Page**
   - Configure dashboard settings
   - API connection settings

### Troubleshooting Dashboard Access

**Dashboard Shows "Cannot Connect to API":**
- ✅ Check `REACT_APP_API_URL` environment variable is set correctly
- ✅ Verify web service is running: `curl https://project-constellation.onrender.com/health`
- ✅ Ensure URL uses `https://` not `http://`
- ✅ Check CORS settings in server (`ALLOWED_ORIGINS` should include dashboard URL)
- ✅ Redeploy dashboard after changing environment variables

**Dashboard URL Shows 404 or "Not Found":**
- ✅ Verify static site is deployed and shows "Live" status
- ✅ Check build completed successfully (check Render logs)
- ✅ Ensure `publishPath` is set to `dashboard/build`

**Dashboard Loads But Shows Empty/No Data:**
- ✅ This is normal if no devices/jobs exist yet
- ✅ Connect a Swift app to register a device
- ✅ Create a training job to see it appear
- ✅ Check browser console (F12) for API errors

**Dashboard Shows 404 for JS/CSS Files or "MIME type 'text/plain'" Error:**
- ❌ **Problem**: Static files aren't being served correctly or build didn't complete
- ✅ **Solution 1**: Check Build Logs
  - Go to Render → Static Site → Logs
  - Verify `npm run build` completed successfully
  - Look for "Build successful" or "The build folder is ready to be deployed"
  - If build failed, fix errors and rebuild
  
- ✅ **Solution 2**: Verify Configuration Settings
  - **Root Directory**: Must be `dashboard` (not empty!)
  - **Build Command**: `npm install && npm run build`
  - **Publish Directory**: `build` (relative to Root Directory)
  - Full path Render uses: `dashboard/build/`
  
- ✅ **Solution 3**: Verify Build Output Structure**
  - After build, should have:
    - `dashboard/build/index.html`
    - `dashboard/build/static/js/main.*.js`
    - `dashboard/build/static/css/main.*.css`
    - `dashboard/build/manifest.json`
  - If these don't exist, build didn't complete
  
- ✅ **Solution 4**: Check for Build Errors**
  - Common issues:
    - Missing dependencies (check `package.json`)
    - Node version incompatibility
    - Build script errors
  - Fix errors and rebuild
  
- ✅ **Solution 5**: Rebuild Dashboard**
  - Option A: Manual rebuild
    - Render → Static Site → Manual Deploy → Deploy latest commit
  - Option B: Trigger via commit
    - Push any change to trigger auto-rebuild
    - Wait for build to complete (~3-5 minutes)
  
- ✅ **Solution 6**: Verify `_redirects` File**
  - File should exist: `dashboard/public/_redirects`
  - Content: `/*    /index.html   200`
  - This is automatically copied to `build/` during build
  - Ensures SPA routing works correctly

**Build Fails:**

**Error: "Import in body of module" or ESLint errors**
- ✅ **Solution**: The latest commit (08f1038) fixes these errors
- ✅ If Render is building from an older commit:
  1. Go to Render → Static Site → Manual Deploy → Deploy latest commit
  2. Or wait a few minutes for auto-deploy to pick up the latest commit
  3. Verify the commit hash in build logs matches `08f1038` or later

**Error: "Installing Poetry" or "Could not find torch==2.1.0" (Python packages being installed)**
- ❌ **Problem**: Render is detecting `requirements.txt` in root and treating your Static Site as a Python service
- ✅ **Solution**: 
  1. Delete the incorrectly created service
  2. Make sure you select **"Static Site"** not **"Web Service"**
  3. **CRITICAL**: Set **Root Directory** to `dashboard` (not empty!)
     - This prevents Render from seeing `requirements.txt` in the root
  4. Set **Build Command** to: `npm install && npm run build` (no `cd dashboard` needed)
  5. Set **Publish Directory** to: `build` (relative to Root Directory)
  6. Verify service type shows "Static Site" not "Web Service"

**Other Build Issues:**
- ✅ Check build logs in Render dashboard
- ✅ Verify `package.json` exists in `dashboard/` directory
- ✅ Ensure Node.js version is compatible (check `package.json` for engines)
- ✅ Check for npm dependency errors in logs
- ✅ Verify you're creating a **Static Site**, not a **Web Service**

**Dashboard URL Not Showing:**
- ✅ Go to Render dashboard → Static Sites
- ✅ Find your `constellation-dashboard` service
- ✅ URL is displayed at the top of the service page
- ✅ If not visible, service may still be building

### Quick Access Checklist

- [ ] Dashboard deployed as Static Site on Render
- [ ] `REACT_APP_API_URL` environment variable set to web service URL
- [ ] Web service is running and accessible
- [ ] Dashboard URL obtained from Render dashboard
- [ ] Dashboard opens in browser without errors
- [ ] Browser console shows no API connection errors

### Using Your Deployed System

#### 1. **Register Devices** (via Swift App or API)
   ```bash
   # Register a device via API
   curl -X POST https://project-constellation.onrender.com/devices/register \
     -H "Content-Type: application/json" \
     -H "x-constellation-client: swift-app" \
     -d '{
       "name": "My MacBook",
       "device_type": "macbook",
       "os_version": "14.0",
       "cpu_cores": 8,
       "memory_gb": 16,
       "gpu_available": true,
       "gpu_memory_gb": 8
     }'
   ```

#### 2. **Create Training Jobs**
   ```bash
   # Create a training job
   curl -X POST https://project-constellation.onrender.com/jobs \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Sentiment Analysis Training",
       "model_type": "nlp",
       "model_name": "bert-base",
       "dataset": "imdb",
       "total_epochs": 10,
       "config": "{\"batch_size\": 32, \"learning_rate\": 0.001}"
     }'
   ```

#### 3. **Monitor Progress**
   - Use the dashboard to view:
     - Device status and activity
     - Training job progress
     - Model repository
   - Or use WebSocket for real-time updates: `wss://project-constellation.onrender.com/ws`

### Available API Endpoints

**Health & Info:**
- `GET /` - Server info
- `GET /health` - Health check

**Devices:**
- `POST /devices/register` - Register a new device
- `GET /devices` - List all devices
- `GET /devices/{device_id}` - Get device details
- `POST /devices/{device_id}/heartbeat` - Update device heartbeat
- `GET /devices/{device_id}/next-job` - Get next job for device

**Training Jobs:**
- `POST /jobs` - Create a training job
- `GET /jobs` - List all jobs
- `GET /jobs/{job_id}` - Get job details
- `POST /jobs/{job_id}/start` - Start a job
- `POST /jobs/{job_id}/complete` - Mark job as complete

**Models:**
- `GET /models` - List all models
- `GET /models/{model_name}/download` - Download a model
- `POST /models` - Upload a model

**Federated Learning:**
- `POST /federated/start` - Start federated learning round
- `POST /devices/{device_id}/federated-update` - Submit federated update
- `POST /federated/aggregate/{job_id}` - Aggregate federated updates

**WebSocket:**
- `WS /ws` - Real-time updates connection

### Monitoring & Maintenance

1. **Check Logs**
   - Render dashboard → Your service → Logs
   - Monitor for errors, warnings, or issues

2. **Database Management**
   - Render dashboard → Database → Connect
   - Use PostgreSQL client to inspect tables:
     - `devices`
     - `training_jobs`
     - `device_training`
     - `models`

3. **Performance Monitoring**
   - Free tier: Basic logs in Render dashboard
   - Monitor response times and error rates
   - Watch for database connection issues

### Common Post-Deployment Tasks

1. **Set Up Custom Domain** (Optional)
   - Render dashboard → Your service → Settings → Custom Domain
   - Add your domain and configure DNS

2. **Configure Environment Variables**
   - Add production-specific settings:
     - `ALLOWED_ORIGINS`: Your actual domain (not `*`)
     - `ENVIRONMENT`: `production`
     - Any API keys or secrets needed

3. **Set Up Monitoring** (Optional)
   - Configure alerts for:
     - Service downtime
     - High error rates
     - Database connection failures

4. **Backup Strategy**
   - Render free tier: Manual backups
   - Upgrade to Starter for automatic backups
   - Export database regularly for critical data

### Troubleshooting Post-Deployment

**Service Not Responding:**
- Check Render dashboard for service status
- Free tier services spin down after 15min inactivity (50s cold start)
- Check logs for errors

**Database Connection Issues:**
- Verify `DATABASE_URL` environment variable is set
- Check database is running in Render dashboard
- Ensure database and service are in same region

**Dashboard Can't Connect:**
- Verify `REACT_APP_API_URL` is set correctly
- Check CORS settings (`ALLOWED_ORIGINS`)
- Ensure web service is running

### Next Steps for Production

- [ ] Upgrade to Starter plan ($7/month) for always-on service
- [ ] Set up custom domain
- [ ] Configure proper CORS origins
- [ ] Set up database backups
- [ ] Add authentication/API keys
- [ ] Configure monitoring and alerts
- [ ] Set up CI/CD for automatic deployments
- [ ] Review and optimize performance

---

## 📱 Connecting Swift Desktop App to Render Deployment

### Method 1: Using the Settings UI (Recommended)

1. **Open the Constellation Swift App**
   - Launch the app on your Mac

2. **Open Settings**
   - Click the gear icon (⚙️) in the top-right corner
   - Or use the menu: **Constellation → Settings**

3. **Configure Server URL**
   - In the **Server Configuration** section:
     - **Server URL**: Enter your Render service URL
       - Example: `https://project-constellation.onrender.com`
       - **Important**: Use `https://` not `http://`
     - **Auth Token**: Leave as default (`constellation-token`) or set your custom token

4. **Configure Device Name** (Optional)
   - Set a unique name for this device
   - Example: `MacBook-Pro-John`, `iMac-Studio-1`

5. **Test Connection**
   - Click **"Test Connection"** button
   - Should show success if server is reachable

6. **Save Settings**
   - Click **"Save"** button
   - Configuration is saved to: `~/Library/Application Support/constellation_config.json`

7. **Connect**
   - Click **"Connect"** button in the main window
   - Status should change to "Connected" (green dot)

### Method 2: Manual Configuration File

1. **Create Configuration File**
   ```bash
   # Create the directory if it doesn't exist
   mkdir -p ~/Library/Application\ Support
   
   # Create configuration file
   cat > ~/Library/Application\ Support/constellation_config.json << EOF
   {
     "server_url": "https://project-constellation.onrender.com",
     "auth_token": "constellation-token",
     "device_name": "My-MacBook-Pro",
     "auto_connect": true
   }
   EOF
   ```

2. **Restart the App**
   - The app will automatically load the configuration
   - If `auto_connect` is `true`, it will connect automatically

### Method 3: Update Default URL in Code

If you want to change the default URL for all users:

1. **Edit `desktop-swift/ConstellationApp_Network.swift`**
   ```swift
   // Change line 24 from:
   @Published var serverURL = "http://localhost:8000"
   
   // To:
   @Published var serverURL = "https://project-constellation.onrender.com"
   ```

2. **Rebuild the App**
   ```bash
   cd desktop-swift
   ./build.sh
   ```

### Verification Steps

1. **Check Connection Status**
   - App should show green dot and "Connected" status
   - Server URL should display your Render URL

2. **Verify Device Registration**
   ```bash
   # Check if device appears in the API
   curl https://project-constellation.onrender.com/devices
   ```
   - Should return your device in the list

3. **Check Dashboard**
   - Open your Render dashboard URL
   - Go to Devices page
   - Your Swift app device should appear in the list

### Troubleshooting Swift App Connection

**Connection Fails:**
- ✅ Verify Render service is running (check Render dashboard)
- ✅ Ensure URL uses `https://` not `http://`
- ✅ Check URL doesn't have trailing slash: `https://project-constellation.onrender.com` (not `/`)
- ✅ Free tier: First connection may take 50 seconds (cold start)
- ✅ Check macOS firewall isn't blocking the connection

**"Server error: 404" or "Invalid response":**
- ✅ Verify the URL is correct
- ✅ Check `/health` endpoint works: `curl https://project-constellation.onrender.com/health`
- ✅ Ensure service is deployed and running

**"Connection timeout":**
- ✅ Free tier services spin down after 15min inactivity
- ✅ First request wakes up the service (takes ~50 seconds)
- ✅ Consider upgrading to Starter plan for always-on service

**Device Not Appearing in Dashboard:**
- ✅ Check device registration succeeded (look for "✅ Connected" in app)
- ✅ Verify device appears in API: `curl https://project-constellation.onrender.com/devices`
- ✅ Check dashboard is pointing to correct API URL

**SSL/Certificate Errors:**
- ✅ Render provides SSL certificates automatically
- ✅ Ensure you're using `https://` not `http://`
- ✅ If custom domain, verify SSL certificate is configured

### Example Configuration

**For Production:**
```json
{
  "server_url": "https://project-constellation.onrender.com",
  "auth_token": "your-secure-token-here",
  "device_name": "MacBook-Pro-Office",
  "auto_connect": true
}
```

**For Local Development:**
```json
{
  "server_url": "http://localhost:8000",
  "auth_token": "constellation-token",
  "device_name": "MacBook-Pro-Dev",
  "auto_connect": true
}
```

### Multiple Devices

You can connect multiple Mac devices to the same Render deployment:

1. **Each device needs unique configuration:**
   - Use different `device_name` for each device
   - Same `server_url` (your Render URL)
   - Same or different `auth_token` (depending on your security setup)

2. **All devices will appear in dashboard:**
   - Dashboard → Devices page shows all connected devices
   - Each device can participate in training jobs

### Security Considerations

1. **Auth Token**
   - Default token is `constellation-token` (for testing)
   - For production, use a strong, unique token
   - Consider implementing proper authentication in the server

2. **HTTPS Only**
   - Always use `https://` for Render deployments
   - Never use `http://` for production

3. **Network Security**
   - Devices connect over public internet to Render
   - Ensure your Render service has proper CORS settings
   - Consider adding IP whitelisting if needed

---

## 🔨 Build Commands Explained

### For Web Service (Docker)
Since you're using Docker (`Dockerfile.server`), **no build command is needed**:
- **Build Command**: Leave empty (`""`)
- Docker automatically:
  1. Builds the Docker image using `Dockerfile.server`
  2. Installs system dependencies (gcc, g++)
  3. Installs Python packages from `requirements.txt`
  4. Copies application code
  5. Sets up the environment

### For Dashboard (Static Site)
The dashboard uses npm to build:
- **Build Command**: `cd dashboard && npm install && npm run build`
- This command:
  1. Changes to the `dashboard` directory
  2. Installs npm dependencies (`npm install`)
  3. Builds the React app (`npm run build`)
  4. Outputs to `dashboard/build` directory

### If Not Using Docker (Alternative)
If deploying without Docker, you would use:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python server/app.py`
- But Docker is recommended for consistency and isolation

---

## 🔧 Troubleshooting

### render.yaml Not Auto-Detected

**Problem**: Render shows "Python 3" instead of detecting Docker, and `render.yaml` is not found.

**Solution - Option 1: Switch to Docker (Recommended)**
1. On the "Create Web Service" page, find the **"Language"** dropdown
2. Change it from **"Python 3"** to **"Docker"**
3. A new field **"Dockerfile Path"** will appear
4. Enter: `./Dockerfile.server`
5. **Build Command**: Leave empty (Docker handles it automatically)
6. **Start Command**: `python server/app.py`
7. Continue with the rest of the configuration

**Solution - Option 2: Use Python 3 (If Docker doesn't work)**
1. Keep **"Language"** as **"Python 3"**
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `python server/app.py`
   - Alternative: `uvicorn server.app:app --host 0.0.0.0 --port $PORT`
4. **Root Directory**: Leave empty (or set to `.` if needed)
5. Continue with environment variables

**Why this happens**: Render may not detect `render.yaml` if:
- The file wasn't committed/pushed to GitHub
- The file is in a subdirectory
- Render's auto-detection failed

**To fix auto-detection**:
- Ensure `render.yaml` is in the root directory
- Commit and push the file to GitHub
- Refresh the Render page and try again

### Region Selection Issue (Page Scrolls Back)

**Problem**: When clicking "Create Web Service", the page scrolls back to the Region field.

**Solution**:
1. **Region is a required field** - You must select a region before creating the service
2. Scroll up to the **Region** dropdown (usually near the top of the form)
3. Select the **SAME region** you used for your PostgreSQL database
   - Check your database settings to see which region it's in
   - Or go to your database → Settings → Region
4. After selecting the region, scroll back down and click **"Create Web Service"** again
5. The form should now submit successfully

**Why this happens**: Render requires all services in the same region for optimal performance and to ensure they can communicate properly.

### GitHub Repository Linking Issues

If you're getting errors when linking your GitHub repository to Render, try these steps:

1. **Check GitHub OAuth Permissions**
   - Go to Render dashboard → **Account Settings** → **Connected Accounts**
   - If GitHub is not connected, click **"Connect GitHub"**
   - Authorize Render to access your repositories
   - Make sure to grant access to **all repositories** or specifically `project-constellation`

2. **Verify Repository Access**
   - Ensure you're logged into Render with the same GitHub account that owns the repository
   - Check that the repository exists at: `https://github.com/varunmitra/project-constellation`
   - If the repository is private, Render needs access to it

3. **Reconnect GitHub Account**
   - Go to Render → **Account Settings** → **Connected Accounts**
   - Click **"Disconnect"** next to GitHub
   - Click **"Connect GitHub"** again
   - Re-authorize Render with the correct permissions

4. **Check Repository Visibility**
   - If your repo is private, ensure Render has been granted access
   - You may need to go to GitHub → **Settings** → **Applications** → **Authorized OAuth Apps** → **Render**
   - Verify that `project-constellation` is in the list of accessible repositories

5. **Manual Repository Selection**
   - When creating a new service, instead of auto-detection:
     - Click **"Connect GitHub"** if prompted
     - Search for `varunmitra/project-constellation` manually
     - Select it from the dropdown

6. **Common Error Messages & Solutions**
   - **"Repository not found"**: Check repository name/owner matches exactly
   - **"Access denied"**: Re-authorize Render in GitHub settings
   - **"OAuth error"**: Disconnect and reconnect GitHub account
   - **"Permission denied"**: Grant Render access to private repositories

7. **Alternative: Deploy via Git URL**
   - If OAuth continues to fail, you can deploy using the public Git URL:
     - Repository URL: `https://github.com/varunmitra/project-constellation.git`
     - Note: This only works for public repositories

### Build Fails

**Common Error: "Could not find a version that satisfies the requirement sqlite3"**
- **Cause**: `sqlite3` is a built-in Python module and should NOT be in `requirements.txt`
- **Fix**: Remove `sqlite3` from `requirements.txt` (already fixed in the repo)
- **Note**: `sqlite3` comes with Python by default, no installation needed

**Other Build Issues**:
- Check build logs in Render dashboard for specific errors
- Verify `requirements.txt` has all dependencies (no built-in modules)
- Ensure `Dockerfile.server` is correct
- If Python version errors appear, check that dependencies support Python 3.9 (used in Dockerfile)

### Database Connection Issues
- Verify `DATABASE_URL` is set correctly
- Check database is running and accessible
- Ensure connection string uses `postgresql://` (not `postgres://`)

### WebSocket Not Working
- Verify WebSocket is enabled in Render (should be automatic)
- Check CORS settings in `app.py`
- Ensure `ALLOWED_ORIGINS` includes your dashboard URL

### Dashboard Can't Connect to API
- ✅ Verify `REACT_APP_API_URL` is set correctly in Static Site environment variables
- ✅ Check CORS settings allow your dashboard domain (`ALLOWED_ORIGINS` should include dashboard URL)
- ✅ Ensure API server is running (check Render dashboard)
- ✅ Verify API endpoints work: `curl https://project-constellation.onrender.com/devices`
- ✅ Check browser console for specific error messages
- ✅ Free tier: First request may take 50 seconds (cold start)

### WebSocket Connection Fails
- ⚠️ **IMPORTANT**: Render free tier **does NOT support WebSockets**
- ✅ Dashboard will automatically fall back to polling mode if WebSocket fails
- ✅ This is **expected behavior** on free tier - polling mode works fine, just less real-time
- ✅ **Why WebSocket fails on free tier:**
  - Render free tier uses HTTP/1.1 load balancers that don't support WebSocket upgrades
  - WebSocket connections require HTTP/1.1 Upgrade header, which free tier doesn't handle
  - Error: `WebSocket connection to 'wss://...' failed` is normal on free tier
- ✅ **Solutions:**
  1. **Use polling mode** (already implemented, works perfectly)
  2. **Upgrade to Starter plan** ($7/month) for WebSocket support
- ✅ Verify WebSocket endpoint exists: `wss://project-constellation.onrender.com/ws` (will fail on free tier)
- ✅ Check server logs - you may see connection attempts but they'll fail on free tier
- ✅ **Dashboard will work fine** - it automatically uses polling when WebSocket fails

---

## 📊 Cost Breakdown

### Free Tier (Testing)
- Web Service: $0 (spins down after 15min inactivity)
- Database: $0 (1GB storage)
- Static Site: $0
- **Total: $0/month**

### Starter Tier (Production)
- Web Service: $7/month
- Database: $7/month (10GB storage)
- Static Site: $0
- **Total: $14/month**

---

## 🔄 Updating Your Deployment

### Automatic Updates
- Render automatically deploys on every push to main branch
- Or manually trigger from Render dashboard

### Manual Update
1. Push changes to GitHub
2. Go to Render dashboard
3. Click **"Manual Deploy"** → **"Deploy latest commit"**

---

## 🔐 Production Checklist

- [ ] Upgrade to Starter plan ($7/month each for server + DB)
- [ ] Set `ALLOWED_ORIGINS` to your actual domain
- [ ] Configure custom domain (optional)
- [ ] Set up authentication tokens
- [ ] Enable database backups
- [ ] Set up monitoring/alerting
- [ ] Configure file storage for models (S3/Spaces)
- [ ] Review security settings
- [ ] Test all endpoints
- [ ] Verify WebSocket connections

---

## 📞 Support

- Render Docs: https://render.com/docs
- Render Community: https://community.render.com
- Check logs in Render dashboard for errors

---

**That's it! Your app should be live in ~10 minutes.** 🎉

