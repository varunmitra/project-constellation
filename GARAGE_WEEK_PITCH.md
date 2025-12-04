# Constellation - Garage Week Pitch

## Project Title
**Constellation: Crowdsourced Computing for Machine Learning**

## Elevator Pitch (30 seconds)
Constellation transforms idle employee workstations into a distributed computing network for training machine learning models. Instead of paying for expensive cloud infrastructure, the system coordinates training jobs across available desktop machines during off-hours. This reduces costs, maximizes existing hardware, and democratizes access to compute resources across Adobe.

## The Problem
- Training ML models requires expensive cloud infrastructure or dedicated GPU clusters
- Thousands of employee workstations sit idle overnight and during breaks
- Teams face barriers to experimentation due to compute costs and availability
- Centralized training infrastructure doesn't scale with growing ML needs

## The Solution
A federated learning platform with:
- **Desktop app** that runs on employee machines and trains during idle time
- **Coordination server** that schedules jobs and aggregates results
- **Web dashboard** for monitoring and management
- **Privacy-preserving** approach that never centralizes raw data

## Why Adobe Should Care

### Cost Savings
Turn existing hardware into training infrastructure instead of buying cloud credits

### Sustainability
Maximize utilization of devices we already own, reducing energy waste

### Innovation
Give every team access to compute resources for experimentation

### Practical Use Cases
- Content recommendation engines
- Image classification for Creative Cloud
- Document understanding for Acrobat
- Customer behavior modeling
- Internal automation tools

## What's Already Built
✅ Working server with job coordination and device management  
✅ macOS desktop app with resource monitoring  
✅ React dashboard for real-time visualization  
✅ Training engine with PyTorch support  
✅ Successfully trained text classification models  

## Garage Week Goals
1. **Make it production-ready** with robust error handling and recovery
2. **Expand capabilities** with more model types and optimization
3. **Scale testing** across multiple devices and job types
4. **Document everything** so other teams can use it
5. **Gather feedback** from potential users across Adobe

## Technologies
Python • FastAPI • PyTorch • Swift • React • Federated Learning • Docker

## Ideal Team Composition
- **Cloud Architect**: Scalable infrastructure and deployment strategy
- **AI/ML Engineer**: Training optimization and federated learning techniques
- **macOS App Developer**: Native app enhancements and system integration

## Success Metrics
- 10+ devices running concurrently
- <5% job failure rate
- Sub-2-second dashboard updates
- Complete deployment documentation
- Positive feedback from demo sessions

## Long-term Vision
This could become Adobe's internal compute platform, enabling:
- On-demand model training for any team
- Automated retraining pipelines for production models
- Research experimentation without budget constraints
- Integration with Adobe Sensei workflows
- A model for sustainable corporate computing

---

**Bottom Line**: Constellation turns a fixed cost (idle computers) into productive capacity, making ML training accessible and affordable for everyone at Adobe.

