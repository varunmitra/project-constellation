# Quick Deployment Guide - Render.com

## 🚀 5-Minute Deployment

### Step 1: Prepare Your Repository
1. Ensure all code is committed and pushed to GitHub
2. Verify `render.yaml` exists in the root directory
3. Check that `Dockerfile.server` is present

### Step 2: Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with GitHub OAuth
3. Authorize Render to access your repositories

### Step 3: Deploy Database
1. In Render dashboard, click **"New +"** → **"PostgreSQL"**
2. Name: `constellation-db`
3. Plan: **Free** (for testing) or **Starter** ($7/month for production)
4. Database: `constellation`
5. User: `constellation_user`
6. Click **"Create Database"**
7. **Copy the Internal Database URL** (you'll need this)

### Step 4: Deploy Web Service
1. In Render dashboard, click **"New +"** → **"Web Service"**
2. Connect your GitHub repository
3. Render will detect `render.yaml` automatically
4. Or configure manually:
   - **Name**: `constellation-server`
   - **Environment**: Docker
   - **Dockerfile Path**: `./Dockerfile.server`
   - **Start Command**: `python server/app.py`
   - **Plan**: Free (testing) or Starter ($7/month)
5. Add environment variables:
   - `DATABASE_URL`: Use the database connection string from Step 3
   - `PORT`: `8000`
   - `PYTHONPATH`: `/app`
   - `ENVIRONMENT`: `production`
   - `ALLOWED_ORIGINS`: `*` (for testing) or your domain (production)
6. Click **"Create Web Service"**
7. Wait for build to complete (~5-10 minutes)

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

## 🔧 Troubleshooting

### Build Fails
- Check build logs in Render dashboard
- Verify `requirements.txt` has all dependencies
- Ensure `Dockerfile.server` is correct

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

