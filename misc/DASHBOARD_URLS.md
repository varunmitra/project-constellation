# Dashboard Access URLs

## 🌐 Render Dashboard (Production - Recommended)

**URL:** https://constellation-dashboard.onrender.com/

**Status:** ✅ Deployed and Available

**Advantages:**
- ✅ No local setup required
- ✅ Works from any browser/device
- ✅ Always available (24/7)
- ✅ Already connected to Render server
- ✅ Shareable URL for demos

**Use When:**
- Creating training jobs
- Monitoring training progress
- Viewing devices and models
- Demonstrating to others
- Quick access without setup

---

## 💻 Local Dashboard (Development)

**URL:** http://localhost:3000

**Status:** Requires local setup

**Setup:**
```bash
cd /Users/vmitra/Documents/GitHub/project-constellation
export REACT_APP_API_URL=https://project-constellation.onrender.com
./scripts/start-dashboard.sh
```

**Advantages:**
- ✅ Full control over code
- ✅ Easy to debug
- ✅ See console logs
- ✅ Test dashboard changes

**Use When:**
- Developing dashboard features
- Debugging dashboard issues
- Testing local changes
- Need to see detailed logs

---

## 🔗 Related URLs

- **Server API:** https://project-constellation.onrender.com
- **Server Health:** https://project-constellation.onrender.com/health
- **API Docs:** https://project-constellation.onrender.com/docs

---

## 📝 Quick Reference

**To create a training job:**
1. Open https://constellation-dashboard.onrender.com/
2. Click "Jobs" → "Create Job"
3. Fill form → Click "Create"
4. Desktop app will auto-detect and train!

