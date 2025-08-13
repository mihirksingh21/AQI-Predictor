# 🐳 Docker Guide for AQI Prediction System

This guide covers how to use Docker with the AQI Prediction System, including building, running, and deploying containers.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Docker Commands](#docker-commands)
- [GitHub Container Registry](#github-container-registry)
- [Docker Compose](#docker-compose)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### Required Software
- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- **Git** for version control
- **GitHub account** for container registry

### System Requirements
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: At least 10GB free space
- **OS**: Windows 10+, macOS 10.15+, or Linux

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd aqi-predictor
cp .env.example .env
# Edit .env with your API keys
```

### 2. Build and Run (Linux/Mac)
```bash
chmod +x docker-build.sh
./docker-build.sh
```

### 3. Build and Run (Windows)
```cmd
docker-build.bat
```

### 4. Verify Installation
```bash
docker ps
docker logs aqi-predictor
```

## 🐳 Docker Commands

### Basic Commands

#### Build Image
```bash
# Build with default tag
docker build -t aqi-predictor:latest .

# Build with specific tag
docker build -t aqi-predictor:v1.0.0 .

# Build with no cache
docker build --no-cache -t aqi-predictor:latest .
```

#### Run Container
```bash
# Run in background
docker run -d --name aqi-predictor \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/images:/app/images \
  -p 8000:8000 \
  aqi-predictor:latest

# Run interactively
docker run -it --rm --name aqi-predictor \
  --env-file .env \
  aqi-predictor:latest bash
```

#### Container Management
```bash
# List containers
docker ps -a

# Start container
docker start aqi-predictor

# Stop container
docker stop aqi-predictor

# Remove container
docker rm aqi-predictor

# View logs
docker logs aqi-predictor
docker logs -f aqi-predictor  # Follow logs
```

#### Image Management
```bash
# List images
docker images

# Remove image
docker rmi aqi-predictor:latest

# Save image to file
docker save aqi-predictor:latest > aqi-predictor.tar

# Load image from file
docker load < aqi-predictor.tar
```

## 🏗️ GitHub Container Registry

### 1. Authenticate to GitHub Container Registry

#### Using Personal Access Token
```bash
# Set your token as environment variable
export CR_PAT=YOUR_GITHUB_TOKEN

# Login to registry
echo $CR_PAT | docker login ghcr.io -u YOUR_USERNAME --password-stdin
```

#### Using GitHub CLI
```bash
# Install GitHub CLI
gh auth login

# Login to container registry
gh auth token | docker login ghcr.io -u YOUR_USERNAME --password-stdin
```

### 2. Tag and Push Images

#### Tag for GitHub Container Registry
```bash
# Tag with your username/organization
docker tag aqi-predictor:latest ghcr.io/YOUR_USERNAME/aqi-predictor:latest

# Tag with version
docker tag aqi-predictor:latest ghcr.io/YOUR_USERNAME/aqi-predictor:v1.0.0
```

#### Push to Registry
```bash
# Push latest
docker push ghcr.io/YOUR_USERNAME/aqi-predictor:latest

# Push specific version
docker push ghcr.io/YOUR_USERNAME/aqi-predictor:v1.0.0

# Push all tags
docker push ghcr.io/YOUR_USERNAME/aqi-predictor --all-tags
```

### 3. Pull from Registry
```bash
# Pull latest
docker pull ghcr.io/YOUR_USERNAME/aqi-predictor:latest

# Pull specific version
docker pull ghcr.io/YOUR_USERNAME/aqi-predictor:v1.0.0

# Pull by digest (most secure)
docker pull ghcr.io/YOUR_USERNAME/aqi-predictor@sha256:digest_hash
```

## 🐙 Docker Compose

### 1. Basic Usage
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build -d
```

### 2. Environment Variables
```bash
# Use .env file (default)
docker-compose up

# Use specific env file
docker-compose --env-file .env.production up

# Override environment variables
OPENWEATHER_API_KEY=your_key docker-compose up
```

### 3. Service Management
```bash
# Start specific service
docker-compose up -d aqi-predictor

# Scale service
docker-compose up -d --scale aqi-predictor=3

# View service status
docker-compose ps
```

## ⚙️ Advanced Configuration

### 1. Multi-Stage Builds
The Dockerfile uses multi-stage builds for optimization:

```dockerfile
# Base stage with dependencies
FROM python:3.11-slim as base
# ... install system dependencies

# Final stage
FROM base as final
# ... copy application code
```

### 2. Build Arguments
```bash
# Build with custom arguments
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  -t aqi-predictor:latest .
```

### 3. Multi-Platform Builds
```bash
# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t aqi-predictor:latest \
  --push .
```

### 4. Health Checks
The container includes health checks:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import config; print('System healthy')" || exit 1
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Permission Denied
```bash
# Fix volume permissions
sudo chown -R $USER:$USER data models results images

# Or run container as current user
docker run -u $(id -u):$(id -g) ...
```

#### 2. Port Already in Use
```bash
# Check what's using the port
lsof -i :8000

# Use different port
docker run -p 8001:8000 ...
```

#### 3. Out of Memory
```bash
# Increase Docker memory limit in Docker Desktop
# Or use memory limits
docker run --memory=2g --memory-swap=4g ...
```

#### 4. Build Failures
```bash
# Clear Docker cache
docker system prune -a

# Check Dockerfile syntax
docker build --no-cache -t test .
```

### Debug Commands
```bash
# Inspect container
docker inspect aqi-predictor

# Execute commands in running container
docker exec -it aqi-predictor bash

# View container resource usage
docker stats aqi-predictor

# Check container logs
docker logs --tail 100 aqi-predictor
```

## 📊 Performance Optimization

### 1. Image Size Optimization
- Use multi-stage builds
- Remove unnecessary packages
- Use .dockerignore effectively
- Use slim base images

### 2. Build Speed
- Use Docker layer caching
- Optimize Dockerfile order
- Use build cache mounts
- Parallel builds with buildx

### 3. Runtime Performance
- Use volume mounts for data
- Limit container resources
- Use health checks
- Monitor resource usage

## 🔒 Security Best Practices

### 1. Image Security
```bash
# Scan for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image aqi-predictor:latest

# Use signed base images
FROM python:3.11-slim@sha256:digest
```

### 2. Runtime Security
```bash
# Run as non-root user
USER aqi_user

# Use read-only filesystem
docker run --read-only ...

# Limit capabilities
docker run --cap-drop=ALL ...
```

### 3. Network Security
```bash
# Use custom networks
docker network create aqi-network

# Limit port exposure
docker run -p 127.0.0.1:8000:8000 ...
```

## 📚 Additional Resources

### Documentation
- [Docker Official Documentation](https://docs.docker.com/)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

### Tools
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Docker Hub](https://hub.docker.com/)
- [Trivy Security Scanner](https://aquasecurity.github.io/trivy/)

### Community
- [Docker Community Forums](https://forums.docker.com/)
- [GitHub Discussions](https://github.com/yourusername/aqi-predictor/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/docker)

---

## 🎯 Next Steps

1. **Build and test** your Docker image locally
2. **Push to GitHub Container Registry** for distribution
3. **Set up CI/CD** with GitHub Actions
4. **Deploy to production** environments
5. **Monitor and optimize** container performance

For questions or issues, please open an issue on GitHub or join our community discussions! 