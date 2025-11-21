# 🐳 Docker Deployment Guide

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f ckd-predictor

# Stop
docker-compose down
```

### Option 2: Docker CLI

```bash
# Build image
docker build -t ckd-predictor:latest .

# Run container
docker run -d \
  --name ckd-stage-predictor \
  -p 7870:7870 \
  -v $(pwd)/models:/app/models:ro \
  ckd-predictor:latest

# View logs
docker logs -f ckd-stage-predictor

# Stop
docker stop ckd-stage-predictor
docker rm ckd-stage-predictor
```

---

## 📋 Prerequisites

- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+ (optional but recommended)
- **Memory**: Minimum 2GB RAM
- **Storage**: ~1GB for image + models

### Install Docker

**Windows:**
- Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
- Follow installation wizard
- Restart computer if prompted

**Linux:**
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

**macOS:**
- Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
- Install and start Docker

---

## 🏗️ Building the Image

### Standard Build

```bash
docker build -t ckd-predictor:latest .
```

### Build with Custom Tag

```bash
docker build -t ckd-predictor:v1.0.0 .
```

### Build for Different Platforms

```bash
# For ARM64 (Apple Silicon, Raspberry Pi)
docker build --platform linux/arm64 -t ckd-predictor:arm64 .

# For AMD64 (Standard x86)
docker build --platform linux/amd64 -t ckd-predictor:amd64 .
```

---

## 🚀 Running the Container

### Basic Run

```bash
docker run -d \
  --name ckd-predictor \
  -p 7870:7870 \
  ckd-predictor:latest
```

### Run with Volume Mounts

```bash
docker run -d \
  --name ckd-predictor \
  -p 7870:7870 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data:ro \
  ckd-predictor:latest
```

### Run with Environment Variables

```bash
docker run -d \
  --name ckd-predictor \
  -p 7870:7870 \
  -e GRADIO_SERVER_NAME=0.0.0.0 \
  -e GRADIO_SERVER_PORT=7870 \
  ckd-predictor:latest
```

### Run with Custom Port

```bash
docker run -d \
  --name ckd-predictor \
  -p 8080:7870 \
  ckd-predictor:latest

# Access at http://localhost:8080
```

---

## 🔧 Docker Compose Configuration

### Basic Configuration

```yaml
version: '3.8'

services:
  ckd-predictor:
    build: .
    ports:
      - "7870:7870"
    restart: unless-stopped
```

### Production Configuration

```yaml
version: '3.8'

services:
  ckd-predictor:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: ckd-stage-predictor
    ports:
      - "7870:7870"
    volumes:
      - ./models:/app/models:ro
    environment:
      - GRADIO_SERVER_NAME=0.0.0.0
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7870"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

---

## 📊 Managing the Container

### View Logs

```bash
# All logs
docker logs ckd-predictor

# Follow logs (live)
docker logs -f ckd-predictor

# Last 100 lines
docker logs --tail 100 ckd-predictor

# With timestamps
docker logs -t ckd-predictor
```

### Container Status

```bash
# Check if running
docker ps

# All containers (including stopped)
docker ps -a

# Container details
docker inspect ckd-predictor

# Resource usage
docker stats ckd-predictor
```

### Stop/Start/Restart

```bash
# Stop container
docker stop ckd-predictor

# Start stopped container
docker start ckd-predictor

# Restart container
docker restart ckd-predictor

# Remove container
docker rm ckd-predictor

# Remove with force (if running)
docker rm -f ckd-predictor
```

### Execute Commands Inside Container

```bash
# Open shell
docker exec -it ckd-predictor /bin/bash

# Run Python script
docker exec ckd-predictor python -c "print('Hello')"

# Check Python version
docker exec ckd-predictor python --version

# List files
docker exec ckd-predictor ls -la /app
```

---

## 🌐 Networking

### Access from Host

```bash
# Default
http://localhost:7870

# Custom port mapping
docker run -p 8080:7870 ...
http://localhost:8080
```

### Access from Other Machines

```bash
# Bind to all interfaces
docker run -p 0.0.0.0:7870:7870 ...

# Then access via
http://<server-ip>:7870
```

### Custom Network

```bash
# Create network
docker network create ckd-network

# Run with network
docker run --network ckd-network ...
```

---

## 💾 Data Persistence

### Volume Mounts

```bash
# Mount models directory
docker run -v $(pwd)/models:/app/models:ro ...

# Mount data directory
docker run -v $(pwd)/data:/app/data:ro ...

# Mount both
docker run \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data:ro \
  ...
```

### Named Volumes

```bash
# Create volume
docker volume create ckd-models

# Use volume
docker run -v ckd-models:/app/models ...

# List volumes
docker volume ls

# Inspect volume
docker volume inspect ckd-models

# Remove volume
docker volume rm ckd-models
```

---

## 🔒 Security Best Practices

### Non-Root User

```dockerfile
# Already configured in Dockerfile
RUN useradd -m -u 1000 ckduser
USER ckduser
```

### Read-Only Volumes

```bash
# Mount as read-only
docker run -v $(pwd)/models:/app/models:ro ...
```

### Resource Limits

```bash
# Limit CPU and memory
docker run \
  --cpus="2.0" \
  --memory="2g" \
  --memory-swap="2g" \
  ...
```

### Network Isolation

```bash
# Run on internal network only
docker run --network internal-network ...
```

---

## 🧪 Testing

### Health Check

```bash
# Manual health check
curl http://localhost:7870

# Docker health status
docker inspect --format='{{.State.Health.Status}}' ckd-predictor
```

### Smoke Test

```bash
# Test container startup
docker run --rm ckd-predictor:latest python -c "import src.utils; print('OK')"
```

---

## 🚀 Production Deployment

### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml ckd-stack

# List services
docker service ls

# Scale service
docker service scale ckd-stack_ckd-predictor=3
```

### Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ckd-predictor
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ckd-predictor
  template:
    metadata:
      labels:
        app: ckd-predictor
    spec:
      containers:
      - name: ckd-predictor
        image: ckd-predictor:latest
        ports:
        - containerPort: 7870
        resources:
          limits:
            memory: "2Gi"
            cpu: "2"
          requests:
            memory: "1Gi"
            cpu: "1"
```

---

## 🐛 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs ckd-predictor

# Check if port is in use
netstat -an | grep 7870

# Run interactively
docker run -it --rm ckd-predictor:latest /bin/bash
```

### Models Not Found

```bash
# Verify volume mount
docker inspect ckd-predictor | grep Mounts -A 10

# Check files inside container
docker exec ckd-predictor ls -la /app/models
```

### Performance Issues

```bash
# Check resource usage
docker stats ckd-predictor

# Increase limits
docker update --cpus="4" --memory="4g" ckd-predictor
```

### Network Issues

```bash
# Test network connectivity
docker exec ckd-predictor ping -c 4 google.com

# Check port binding
docker port ckd-predictor
```

---

## 📦 Image Management

### List Images

```bash
docker images
```

### Remove Images

```bash
# Remove specific image
docker rmi ckd-predictor:latest

# Remove all unused images
docker image prune -a
```

### Push to Registry

```bash
# Tag for registry
docker tag ckd-predictor:latest registry.example.com/ckd-predictor:latest

# Push
docker push registry.example.com/ckd-predictor:latest
```

### Save/Load Images

```bash
# Save to file
docker save ckd-predictor:latest > ckd-predictor.tar

# Load from file
docker load < ckd-predictor.tar
```

---

## 🔄 Updates

### Update to New Version

```bash
# Pull latest code
git pull

# Rebuild image
docker-compose build

# Restart services
docker-compose up -d

# Or with Docker CLI
docker build -t ckd-predictor:latest .
docker stop ckd-predictor
docker rm ckd-predictor
docker run -d --name ckd-predictor -p 7870:7870 ckd-predictor:latest
```

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

---

**🐳 Happy Containerizing!**
