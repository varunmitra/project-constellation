# Project Constellation: Presentation Story

## The Problem

I was training a small language model on my laptop. Training was slow—too slow. I needed more compute power, but cloud GPUs are expensive.

## The Solution

I realized I had multiple devices sitting idle: my personal laptop, my work laptop. What if I could use them together? Two devices could train **2x faster**. More devices, even faster.

This became **Project Constellation**—a federated learning platform that trains AI models across multiple devices without sharing raw data. Each device trains locally, then shares only model weights.

## Architecture: Three Components

**1. Server** (FastAPI Backend)
- Manages training jobs
- Coordinates devices
- Aggregates model updates

**2. Dashboard** (React Web UI)
- Create and monitor jobs
- View device status
- Track training progress

**3. Desktop App** (Swift/macOS)
- Runs on each device
- Automatically joins jobs
- Trains locally and sends updates

## How It Works

1. Create a training job via dashboard
2. Devices automatically discover and join
3. Each device trains locally on its data
4. Server aggregates updates using Federated Averaging
5. Process repeats until convergence

## Results

- **Faster training**: Multiple devices work in parallel
- **Privacy**: Data never leaves devices
- **Efficient**: Uses idle compute power
- **Scalable**: Add devices as needed

Successfully trained text classification models with multiple devices contributing simultaneously.

## Key Points

- Started from personal need: slow single-device training
- Innovation: Leverage devices you already own
- Three components: Server, Dashboard, Desktop App
- Benefits: Privacy, speed, accessibility
