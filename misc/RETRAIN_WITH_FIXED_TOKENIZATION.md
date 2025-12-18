# How to Retrain with Fixed Deterministic Tokenization

The tokenization fixes are already in place! You just need to create a new training job and the app will automatically use the fixed code.

## ✅ What's Already Fixed

- ✅ `training/engine.py` - Uses deterministic hashing
- ✅ `training/ag_news_trainer.py` - Uses deterministic hashing  
- ✅ `federated/client.py` - Uses deterministic hashing
- ✅ `desktop-swift/Constellation.app/Contents/Resources/training/` - Bundled scripts updated
- ✅ `invoke_model.py` - Uses deterministic hashing for inference

## 🚀 Steps to Retrain

### Option 1: Using Dashboard UI (Easiest)

**📖 See detailed UI guide:** `CREATE_RETRAIN_JOB_UI.md`

**🌐 Render Dashboard (Recommended - No Setup!):**
- **URL:** https://constellation-dashboard.onrender.com/
- Just open in browser and create jobs - no local setup needed!

**OR Use Local Dashboard:**

1. **Start Dashboard** (if not running):
   ```bash
   cd /Users/vmitra/Documents/GitHub/project-constellation
   export REACT_APP_API_URL=https://project-constellation.onrender.com
   ./scripts/start-dashboard.sh
   ```

2. **Open Dashboard**: http://localhost:3000

3. **Create New Job**:
   - Go to **"Jobs"** page (click in sidebar)
   - Click **"Create Job"** button (top right)
   - Fill in the form:
     - **Job Name**: "Retrained Model with Fixed Tokenization"
     - **Model Type**: Select **"Text Classification"**
     - **Model**: Leave as "None (Create New Model)"
     - **Dataset**: Select **"AG News"** (or "Synthetic Data" for quick test)
     - **Total Epochs**: Enter **20** (or more for better accuracy)
     - **Number of Classes**: Enter **4** (or leave empty for AG News default)
   - Click **"Create Job"** button

4. **Start Desktop App** (if not running):
   ```bash
   cd desktop-swift
   ./build/Constellation
   ```

5. **Training Starts Automatically**: The app will detect the job and start training within 10 seconds

### Option 2: Using Script (Quick)

```bash
cd /Users/vmitra/Documents/GitHub/project-constellation
source venv/bin/activate

# Create a new training job
python3 -c "
import requests
import json

SERVER_URL = 'https://project-constellation.onrender.com'
AUTH_HEADER = {'Authorization': 'Bearer constellation-token'}

job_data = {
    'name': 'Retrained Model - Fixed Tokenization',
    'model_type': 'text_classification',
    'dataset': 'synthetic',  # or 'ag_news' for real data
    'total_epochs': 20,
    'config': {
        'vocab_size': 10000,
        'seq_length': 100,
        'num_samples': 2000,  # More samples = better training
        'num_classes': 4,
        'batch_size': 32,
        'learning_rate': 0.001
    }
}

response = requests.post(
    f'{SERVER_URL}/jobs',
    json=job_data,
    headers=AUTH_HEADER,
    timeout=30
)
response.raise_for_status()
job = response.json()
print(f'✅ Job created: {job[\"name\"]}')
print(f'   Job ID: {job[\"id\"]}')
print(f'   Status: {job[\"status\"]}')
print()
print('🚀 Desktop app will automatically pick up this job and start training!')
"
```

### Option 3: Using create_sentiment_job.py (Modified)

```bash
cd /Users/vmitra/Documents/GitHub/project-constellation
source venv/bin/activate

# Edit create_sentiment_job.py to change dataset/epochs if needed
python3 create_sentiment_job.py
```

## 📊 Monitor Training

**Watch the Desktop App Terminal** for training logs:
- Connection status
- Job detection
- Training progress
- Epoch-by-epoch accuracy and loss

**Or Check Dashboard**: http://localhost:3000
- See job status
- Monitor progress
- View device activity

## ⏳ After Training Completes

1. **Download and Aggregate**:
   ```bash
   python3 download_and_aggregate.py
   ```

2. **Test the New Model**:
   ```bash
   python3 invoke_model.py --text "Breaking news about technology and AI"
   ```

3. **Expected Result**: Should correctly predict "Sci/Tech" instead of "Sports"!

## 🔍 Verify Fixed Tokenization is Being Used

Check the training logs in the desktop app terminal. You should see:
- Training progress with actual accuracy/loss values
- No errors about tokenization
- Model training successfully

The bundled training scripts in `Constellation.app/Contents/Resources/training/` already have the fixes, so the app will automatically use deterministic hashing.

## 💡 Tips

- **More Epochs**: Use 20-30 epochs for better accuracy
- **More Samples**: Increase `num_samples` to 2000-5000 for better training
- **Real Data**: Use `dataset: 'ag_news'` if you want to train on actual AG News data
- **Multiple Devices**: Run the app on multiple devices for true federated learning

## 🎯 Quick Test After Retraining

```bash
# Test with technology text (should predict Sci/Tech)
python3 invoke_model.py --text "New AI breakthrough in machine learning research"

# Test with sports text (should predict Sports)  
python3 invoke_model.py --text "Team wins championship game in overtime"

# Test with business text (should predict Business)
python3 invoke_model.py --text "Stock market reaches new all-time high"
```

If predictions are correct, the retraining worked! 🎉

