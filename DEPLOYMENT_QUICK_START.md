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
1. In Render dashboard, click **"New +"** → **"Static Site"**
2. Connect your GitHub repository
3. Configure:
   - **Name**: `constellation-dashboard`
   - **Build Command**: `cd dashboard && npm install && npm run build`
   - **Publish Directory**: `dashboard/build`
   - **Environment Variable**: 
     - `REACT_APP_API_URL`: Your web service URL (e.g., `https://constellation-server.onrender.com`)
4. Click **"Create Static Site"**
5. Wait for build (~3-5 minutes)

### Step 6: Verify Deployment
1. **Test Server**:
   - Visit: `https://constellation-server.onrender.com/health`
   - Should return: `{"status": "healthy", "timestamp": "..."}`

2. **Test Dashboard**:
   - Visit your static site URL
   - Should load the React dashboard

3. **Test WebSocket**:
   - Use a WebSocket client to connect to `wss://constellation-server.onrender.com/ws`
   - Should connect successfully

### Step 7: Update Dashboard API URL
1. Go to your Static Site settings in Render
2. Update `REACT_APP_API_URL` environment variable
3. Redeploy the static site

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
- Verify `REACT_APP_API_URL` is set correctly
- Check CORS settings allow your dashboard domain
- Ensure API server is running

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

