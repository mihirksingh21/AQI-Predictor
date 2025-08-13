# 🐳 Docker Setup Guide for AQI Prediction System

This guide will help you install Docker and set up the AQI Prediction System in containers.

## 📋 Prerequisites

- **Windows 10/11** (64-bit) or **Windows Server 2019+**
- **WSL 2** (Windows Subsystem for Linux 2) - Required for Docker Desktop
- **Virtualization enabled** in BIOS (Intel VT-x/AMD-V)
- **Administrator privileges** for installation

## 🔧 Installation Steps

### Step 1: Enable WSL 2

1. **Open PowerShell as Administrator** and run:
   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   ```

2. **Restart your computer**

3. **Download and install WSL 2 Linux kernel update**:
   - Go to: https://aka.ms/wsl2kernel
   - Download and install the package

4. **Set WSL 2 as default**:
   ```powershell
   wsl --set-default-version 2
   ```

### Step 2: Install Docker Desktop

1. **Download Docker Desktop for Windows**:
   - Go to: https://www.docker.com/products/docker-desktop
   - Click "Download for Windows"

2. **Run the installer**:
   - Double-click `Docker Desktop Installer.exe`
   - Follow the installation wizard
   - **Important**: Check "Use WSL 2 instead of Hyper-V" when prompted

3. **Restart your computer**

4. **Start Docker Desktop**:
   - Search for "Docker Desktop" in Start menu
   - Click to launch
   - Wait for Docker to start (whale icon in system tray)

### Step 3: Verify Installation

1. **Open PowerShell** and run:
   ```powershell
   docker --version
   docker-compose --version
   ```

2. **Expected output**:
   ```
   Docker version 20.10.x, build xxxxxxx
   docker-compose version 1.29.x, build xxxxxxx
   ```

3. **Test Docker**:
   ```powershell
   docker run hello-world
   ```

## 🚀 Quick Start with Docker

### 1. Build the AQI Predictor Image

```powershell
# Navigate to your project directory
cd "D:\AQI Predictor"

# Build the Docker image
docker build -t aqi-predictor:latest .
```

### 2. Run the Container

```powershell
# Create necessary directories
mkdir -p data, models, results, images

# Run the container
docker run -d --name aqi-predictor `
  --env-file .env `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/models:/app/models `
  -v ${PWD}/results:/app/results `
  -v ${PWD}/images:/app/images `
  -p 8000:8000 `
  aqi-predictor:latest
```

### 3. Check Container Status

```powershell
# View running containers
docker ps

# View container logs
docker logs aqi-predictor

# Stop container
docker stop aqi-predictor

# Remove container
docker rm aqi-predictor
```

## 🐙 Using Docker Compose

### 1. Start Services

```powershell
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. Rebuild and Restart

```powershell
# Rebuild and start
docker-compose up --build -d

# View service status
docker-compose ps
```

## 🔍 Troubleshooting

### Common Issues

#### 1. "Docker is not recognized"
- **Solution**: Restart PowerShell after Docker Desktop installation
- **Alternative**: Restart your computer

#### 2. "WSL 2 installation is incomplete"
- **Solution**: Update WSL to latest version
  ```powershell
  wsl --update
  ```

#### 3. "Virtualization is disabled"
- **Solution**: Enable virtualization in BIOS
  - Restart computer and enter BIOS (usually F2, F10, or Del)
  - Look for "Virtualization Technology", "Intel VT-x", "AMD-V"
  - Enable and save

#### 4. "Port already in use"
- **Solution**: Change port or stop conflicting service
  ```powershell
  # Use different port
  docker run -p 8001:8000 ...
  ```

#### 5. "Permission denied"
- **Solution**: Run PowerShell as Administrator
- **Alternative**: Fix volume permissions
  ```powershell
  # Create directories with proper permissions
  New-Item -ItemType Directory -Force -Path data, models, results, images
  ```

### Docker Desktop Issues

#### 1. Docker Desktop won't start
- **Check**: Windows Defender Firewall settings
- **Solution**: Allow Docker Desktop through firewall

#### 2. WSL 2 connection failed
- **Solution**: Restart WSL
  ```powershell
  wsl --shutdown
  wsl
  ```

#### 3. Out of memory errors
- **Solution**: Increase Docker Desktop memory limit
  - Open Docker Desktop
  - Go to Settings → Resources → Advanced
  - Increase memory limit (recommended: 4GB+)

## 📚 Alternative Installation Methods

### Option 1: Chocolatey Package Manager

```powershell
# Install Chocolatey first (if not installed)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Docker Desktop
choco install docker-desktop
```

### Option 2: Winget Package Manager

```powershell
# Install Docker Desktop
winget install Docker.DockerDesktop
```

### Option 3: Manual Download

1. Go to https://github.com/docker/docker-install
2. Download the latest release
3. Follow manual installation instructions

## 🔒 Security Considerations

### 1. Docker Desktop Settings
- **Enable**: "Use the WSL 2 based engine"
- **Enable**: "Use Docker Compose V2"
- **Enable**: "Use BuildKit"

### 2. Resource Limits
- **Memory**: 4GB minimum, 8GB recommended
- **CPU**: 2 cores minimum, 4 cores recommended
- **Disk**: 20GB minimum free space

### 3. Network Security
- **Firewall**: Allow Docker through Windows Defender
- **Ports**: Only expose necessary ports (8000 for AQI system)

## 📊 Performance Optimization

### 1. WSL 2 Configuration
Create `.wslconfig` in your user directory:
```ini
[wsl2]
memory=8GB
processors=4
swap=2GB
localhostForwarding=true
```

### 2. Docker Desktop Settings
- **Resources**: Allocate sufficient memory and CPU
- **Advanced**: Enable BuildKit for faster builds
- **Docker Engine**: Optimize daemon configuration

### 3. Volume Mounts
- Use volume mounts for persistent data
- Avoid binding large directories
- Use `.dockerignore` to exclude unnecessary files

## 🎯 Next Steps After Installation

1. **Test basic Docker functionality**:
   ```powershell
   docker run hello-world
   ```

2. **Build your AQI Predictor image**:
   ```powershell
   docker build -t aqi-predictor:latest .
   ```

3. **Run the system in container**:
   ```powershell
   docker run -d --name aqi-predictor --env-file .env aqi-predictor:latest
   ```

4. **Set up GitHub Container Registry** (optional):
   - Create personal access token
   - Authenticate to ghcr.io
   - Push your images

5. **Explore Docker Compose**:
   - Multi-service orchestration
   - Environment management
   - Volume persistence

## 📞 Getting Help

### Official Resources
- [Docker Desktop for Windows](https://docs.docker.com/desktop/windows/)
- [WSL 2 Installation Guide](https://docs.microsoft.com/en-us/windows/wsl/install)
- [Docker Community Forums](https://forums.docker.com/)

### Community Support
- [Stack Overflow](https://stackoverflow.com/questions/tagged/docker+windows)
- [GitHub Issues](https://github.com/yourusername/aqi-predictor/issues)
- [Docker Discord](https://discord.gg/docker)

### Troubleshooting Commands
```powershell
# Check Docker status
docker info

# Check WSL status
wsl --list --verbose

# Check system resources
Get-ComputerInfo | Select-Object WindowsVersion, TotalPhysicalMemory

# Check virtualization
Get-ComputerInfo | Select-Object HyperVRequirementVirtualizationFirmwareEnabled
```

---

## 🎉 Success!

Once Docker is installed and working, you'll be able to:
- ✅ **Containerize** your AQI Prediction System
- ✅ **Deploy** consistently across environments
- ✅ **Scale** your application easily
- ✅ **Share** your system via GitHub Container Registry
- ✅ **Automate** builds with GitHub Actions

Happy containerizing! 🐳 