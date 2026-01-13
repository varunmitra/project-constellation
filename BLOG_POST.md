# Training AI Models Across Multiple Devices: Building Project Constellation

A few months ago, I decided to build my own small language model. I had everything I needed—data, code, and enthusiasm. But there was one problem: training was painfully slow.

I was running training jobs on my laptop, watching epochs crawl by. What should have taken hours was going to take days, maybe even weeks. I needed more compute power, but cloud GPU instances were expensive, and I didn't have access to a dedicated training cluster.

That's when I had a simple thought: **What if I could use multiple devices I already own?**

I looked around—I had my personal laptop, my work laptop, both sitting idle most of the time. Each had processing power going unused. The math was straightforward: two laptops could train **2x faster**. Add more devices, and the speedup multiplies.

This idea became **Project Constellation**—a federated learning platform that trains AI models across multiple devices without sharing raw data.

## What is Federated Learning?

Traditional machine learning requires centralizing all your data on one server. Federated learning flips this: each device trains locally on its own data, then shares only the learned model weights. The server aggregates these updates to create a global model that combines knowledge from all devices.

The benefits are clear:
- **Privacy**: Your data never leaves your device
- **Speed**: Multiple devices work in parallel
- **Efficiency**: Leverage idle compute power you already have
- **Accessibility**: No need for expensive cloud GPUs

## Building the System

Project Constellation has three core components:

### 1. The Server (FastAPI Backend)

The central orchestrator that manages everything:
- Creates and manages training jobs
- Coordinates device registration and monitoring
- Aggregates model updates using Federated Averaging
- Stores models and tracks versions

Built with FastAPI and deployed on Render, it handles the coordination logic that makes distributed training possible.

### 2. The Dashboard (React Web UI)

A clean web interface for managing the entire system:
- Create training jobs with a few clicks
- Monitor device status and activity in real-time
- Track training progress and view metrics
- Manage models and view aggregated results

The dashboard makes it easy to see what's happening across all your devices at a glance.

### 3. The Desktop App (Swift/macOS)

A native macOS application that runs on each participating device:
- Automatically discovers and joins training jobs
- Downloads the current global model
- Trains locally using the device's compute resources
- Sends model updates back to the server

The app runs in the background, so you can continue using your laptop while it contributes to training.

## How It Works

The training flow is straightforward:

1. **Create a Job**: Using the dashboard, you specify the model architecture, dataset, and training parameters.

2. **Devices Join**: The desktop app on each device automatically discovers the job and registers with the server.

3. **Local Training**: Each device downloads the current global model, trains it locally on its own data, and sends back the updated weights.

4. **Aggregation**: The server aggregates all device updates using Federated Averaging, creating a new global model.

5. **Iterate**: The process repeats until the model converges.

## Real Results

I've successfully trained text classification models using this system. Multiple devices contributed their compute power simultaneously, significantly reducing training time. The system works—and it's surprisingly simple to use.

## Technical Challenges

Building this wasn't without challenges. One key issue was ensuring consistent tokenization between training and inference. Python's default `hash()` function is non-deterministic across runs, which caused tokenization mismatches. The solution was implementing deterministic hashing using MD5, ensuring the same word always gets the same token ID.

Another challenge was coordinating multiple devices reliably. The system needed to handle devices joining and leaving, network interruptions, and varying compute speeds. Heartbeat monitoring and automatic job discovery solved most of these issues.

## Why This Matters

Most people experimenting with AI don't have access to expensive cloud GPUs or dedicated clusters. But many of us have multiple devices—laptops, desktops, even phones—sitting idle. Project Constellation makes it possible to leverage that idle compute power.

This isn't just about speed. It's about making AI training more accessible and privacy-preserving. Your data stays on your devices. You use resources you already own. And you can train models faster than you could alone.

## What's Next

The system is working, but there's room for improvement. Better device coordination, support for more model types, and mobile device integration are all possibilities. The core idea—leveraging multiple devices for distributed training—is proven and practical.

## Try It Yourself

Project Constellation is open source and available on GitHub: [github.com/varunmitra/project-constellation](https://github.com/varunmitra/project-constellation)

The system is deployed and ready to use. You can start training models across your devices today.

---

**Project Constellation** demonstrates that you don't need expensive infrastructure to train AI models effectively. Sometimes, the solution is right in front of you—or sitting on your desk.

