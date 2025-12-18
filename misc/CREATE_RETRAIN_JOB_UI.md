# How to Create a Retraining Job Through the Dashboard UI

## 🎯 Two Options: Render Dashboard (Recommended) or Local Dashboard

### Option A: Use Render Dashboard (Easiest - No Setup Required!)

**🌐 Render Dashboard URL:** https://constellation-dashboard.onrender.com/

**Advantages:**
- ✅ No local setup needed
- ✅ Works from any browser/device
- ✅ Always available
- ✅ Already connected to Render server

**Steps:**
1. Open https://constellation-dashboard.onrender.com/ in your browser
2. Skip to Step 3 below (Create Job)

---

### Option B: Use Local Dashboard (For Development)

### 1. Start the Dashboard Locally

```bash
cd /Users/vmitra/Documents/GitHub/project-constellation
export REACT_APP_API_URL=https://project-constellation.onrender.com
./scripts/start-dashboard.sh
```

### 2. Open Dashboard in Browser

Navigate to: **http://localhost:3000**

---

## Step-by-Step Instructions (Same for Both Options)

### 3. Navigate to Jobs Page

### 3. Navigate to Jobs Page

- Click on **"Jobs"** in the sidebar navigation
- You'll see a list of existing jobs (if any)

### 4. Click "Create Job" Button

- Look for the **"Create Job"** button (usually at the top right)
- Click it to open the job creation modal

### 5. Fill Out the Form

Fill in the following fields:

#### **Job Name** (Required)
```
Retrained Model - Fixed Deterministic Tokenization
```
or any descriptive name you prefer

#### **Model Type** (Required)
Select from dropdown:
- ✅ **Text Classification** (select this for retraining)

#### **Model** (Optional)
- Leave as **"None (Create New Model)"** or select an existing model
- For retraining, you can leave this blank to create a new model

#### **Dataset** (Required)
Select from dropdown:
- **Synthetic Data (Default)** - Uses random data (fast, good for testing)
- **AG News** - Real news classification data (4 classes: World, Sports, Business, Sci/Tech)
- **IMDB** - Movie reviews (2 classes: Positive/Negative)
- **Yelp** - Restaurant reviews (5 classes: 1-5 stars)
- **Amazon** - Product reviews (5 classes: 1-5 stars)

**Recommendation for retraining:**
- Use **"AG News"** if you want real text classification
- Use **"Synthetic Data"** for quick testing

#### **Total Epochs** (Required)
Enter a number:
- **10-20** epochs: Quick training (good for testing)
- **20-30** epochs: Better accuracy (recommended)
- **30+** epochs: Best accuracy (takes longer)

**Recommendation:** Start with **20** epochs

#### **Number of Classes** (Optional)
- **Leave empty** to use dataset default:
  - AG News: 4 classes (World, Sports, Business, Sci/Tech)
  - Synthetic: 2 classes
  - IMDB: 2 classes
  - Yelp/Amazon: 5 classes
- **Or specify** if you want a custom number:
  - For AG News: Enter **4**
  - For sentiment: Enter **3** (Positive, Neutral, Negative)

### 6. Click "Create Job"

- Review your settings
- Click the **"Create Job"** button
- The modal will close and you'll see the new job appear in the list

### 7. Verify Job Created

- The new job should appear in the jobs list
- Status will be **"pending"** initially
- You'll see the job details (name, type, dataset, epochs)

### 8. Start Desktop App (If Not Running)

The desktop app will automatically detect and start training:

```bash
cd /Users/vmitra/Documents/GitHub/project-constellation/desktop-swift
./build/Constellation
```

**Watch the terminal** - you'll see:
- Job detection
- Training start
- Progress updates
- Completion status

### 9. Monitor Training

**In Dashboard:**
- Refresh the Jobs page to see status updates
- Status will change: `pending` → `running` → `completed`
- Progress bar will show training progress

**In Desktop App Terminal:**
- Real-time training logs
- Epoch-by-epoch accuracy and loss
- Training completion message

## Example Form Values for Retraining

Here's a recommended configuration:

| Field | Value |
|-------|-------|
| **Job Name** | `Retrained Model - Fixed Tokenization` |
| **Model Type** | `Text Classification` |
| **Model** | `None (Create New Model)` |
| **Dataset** | `AG News` |
| **Total Epochs** | `20` |
| **Number of Classes** | `4` (or leave empty) |

## Quick Reference: Field Descriptions

- **Job Name**: Descriptive name for your training job
- **Model Type**: Type of model to train (text, vision, NLP)
- **Model**: Optional - select existing model or create new
- **Dataset**: Training data source (synthetic or real datasets)
- **Total Epochs**: How many training iterations (more = better but slower)
- **Number of Classes**: Output categories (leave empty for dataset default)

## Troubleshooting

**Job doesn't appear:**
- Check browser console for errors (F12)
- Verify dashboard is connected to Render server
- Check network tab for API errors

**Training doesn't start:**
- Make sure desktop app is running
- Check app is connected to Render server
- Verify device is registered (check Devices page)

**Job status stuck:**
- Refresh the dashboard page
- Check server logs on Render
- Verify desktop app is running and connected

## After Training Completes

1. **Download and Aggregate:**
   ```bash
   python3 download_and_aggregate.py
   ```

2. **Test the New Model:**
   ```bash
   python3 invoke_model.py --text "Breaking news about technology"
   ```

3. **Expected Result:** Should correctly predict "Sci/Tech"! 🎉

