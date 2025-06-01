# Hand Gesture Recognition MLOps Project 🖐️

## Overview 🎯
This project implements a real-time hand gesture recognition system using FastAPI, with comprehensive MLOps practices including monitoring, testing, and CI/CD pipeline integration.

## Architecture 🏗️
The project uses a modern microservices architecture with the following components:
- FastAPI backend service for gesture recognition
- Prometheus for metrics collection
- Grafana for visualization and monitoring
- Docker containers for deployment
- GitHub Actions for CI/CD

## Setup and Installation 🚀

### Prerequisites
- Docker and Docker Compose
- Python 3.9+
- Git

### Quick Start
1. Clone the repository:
```bash
git clone https://github.com/SamaQaraa/MLOPS-final-project-backend.git
cd MLOPS-final-project-backend
```

2. Run with Docker Compose:
```bash
docker-compose up -d
```

3. Access the services:
- FastAPI: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Monitoring with Grafana 📊

Our Grafana dashboard provides real-time monitoring of critical metrics. Here's why we chose specific visualizations:

### 1. Total Requests Counter 📈
- **What**: Displays the total number of API requests per second
- **Why**: 
  - Helps monitor overall system usage
  - Identifies traffic patterns and potential spikes
  - Assists in capacity planning

### 2. Invalid Input Requests 🚫
- **What**: Shows the rate of invalid requests received
- **Why**: 
  - Helps detect potential API misuse
  - Identifies client-side issues
  - Monitors data quality

### 3. Model Inference Time Graph ⚡
- **What**: Visualizes model prediction latency with average and 95th percentile
- **Why**: 
  - Tracks model performance in real-time
  - Helps identify performance degradation
  - Ensures SLA compliance
  - The 95th percentile helps identify outliers and worst-case scenarios

### Metrics Collection 📉
- Using Prometheus for reliable metrics collection
- Custom metrics implemented:
  - `http_requests_total`: Counter for total HTTP requests
  - `invalid_input_requests_total`: Counter for invalid inputs
  - `model_inference_duration_seconds`: Histogram for inference time

## Testing 🧪
Run the test suite:
```bash
python run_tests.py
```

The project includes:
- Unit tests
- Integration tests
- Performance tests

## CI/CD Pipeline 🔄
Automated deployment using GitHub Actions:
- Runs tests
- Builds Docker images
- Deploys to production

## API Documentation 📚
Once running, visit http://localhost:8000/docs for the interactive API documentation.

## Contributing 🤝
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

