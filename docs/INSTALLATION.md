# 📦 Installation Guide

## 🚀 Quick Install (Windows PowerShell)

### One-Command Install

```powershell
# Download and run installer
.\install.ps1
```

This will automatically:
- ✅ Check Python 3.11+ installation
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Verify installation
- ✅ Launch the application

---

## 📋 Manual Installation

### Prerequisites

- **Python 3.11+** ([Download](https://www.python.org/downloads/))
- **pip** (included with Python)
- **Git** (optional, for cloning)

### Step 1: Get the Code

**Option A: Clone Repository**
```bash
git clone https://github.com/e-RickReyJim/CDK.git
cd CDK
```

**Option B: Download ZIP**
1. Download from GitHub
2. Extract to desired location
3. Open PowerShell/Terminal in extracted folder

### Step 2: Create Virtual Environment

```powershell
# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Check if all packages installed
pip list

# Test imports
python -c "import gradio, sklearn, pandas; print('✅ All good!')"
```

### Step 5: Run Application

```bash
python src/app.py
```

Open browser to: `http://127.0.0.1:7870`

---

## 🐳 Docker Installation

For containerized deployment, see [docs/DOCKER.md](DOCKER.md)

**Quick Docker Start:**

```bash
# Using Docker Compose
docker-compose up -d

# Using Docker CLI
docker build -t ckd-predictor .
docker run -d -p 7870:7870 ckd-predictor
```

---

## 🛠️ Advanced Installation Options

### Install with Development Tools

```powershell
# Use install script with dev mode
.\install.ps1 -DevMode

# Or manually
pip install -r requirements.txt
pip install pytest jupyter black flake8
```

### Install in Specific Location

```powershell
# Create installation directory
mkdir C:\Apps\CKD-Predictor
cd C:\Apps\CKD-Predictor

# Clone/extract project here
# Then follow installation steps
```

### Install for Production

```bash
# Create system service (Linux)
sudo nano /etc/systemd/system/ckd-predictor.service

# Add service configuration
[Unit]
Description=CKD Stage Predictor
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/ckd-predictor
ExecStart=/var/www/ckd-predictor/.venv/bin/python src/app.py
Restart=always

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable ckd-predictor
sudo systemctl start ckd-predictor
```

---

## 🔧 Installation Script Options

### Script Parameters

```powershell
# Skip Python version check
.\install.ps1 -SkipPythonCheck

# Install but don't launch
.\install.ps1 -NoLaunch

# Install with dev tools
.\install.ps1 -DevMode

# Combine options
.\install.ps1 -NoLaunch -DevMode
```

### What the Script Does

1. ✅ Checks Python 3.11+ installation
2. ✅ Verifies pip availability
3. ✅ Creates virtual environment (.venv)
4. ✅ Activates virtual environment
5. ✅ Upgrades pip to latest
6. ✅ Installs all dependencies
7. ✅ Verifies project structure
8. ✅ Tests package imports
9. ✅ Provides installation summary
10. ✅ (Optional) Launches application

---

## 📱 Installation on Different Platforms

### Windows 10/11

```powershell
# Install Python from https://python.org
# Check "Add Python to PATH"

# Clone/Download project
cd C:\Projects\CDK

# Run installer
.\install.ps1
```

### Ubuntu/Debian Linux

```bash
# Install Python 3.11
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# Clone project
cd ~/projects/CDK

# Create venv and install
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run application
python src/app.py
```

### macOS

```bash
# Install Python via Homebrew
brew install python@3.11

# Clone project
cd ~/Projects/CDK

# Create venv and install
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run application
python src/app.py
```

### Raspberry Pi (ARM)

```bash
# Install dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv
sudo apt install python3-dev build-essential

# Clone project
cd ~/CDK

# Install (may take longer on ARM)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
python src/app.py
```

---

## 🧪 Verify Installation

### Quick Test

```bash
# Activate venv (if not active)
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate    # Linux/Mac

# Test Python
python --version
# Should show: Python 3.11.x

# Test packages
python -c "import gradio; print(f'Gradio {gradio.__version__}')"
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__}')"

# Test application files
python -c "from src.utils import calculate_egfr; print('✅ Utils OK')"
```

### Full Verification

```bash
# Run all imports
python -c "
import gradio
import sklearn
import pandas
import numpy
import matplotlib
import joblib
print('✅ All packages verified')
"

# Check models
ls models/*.pkl  # Should list 9 .pkl files

# Check structure
ls -R src/  # Should show app.py, utils.py, __init__.py
```

---

## 🐛 Troubleshooting

### Python Not Found

**Windows:**
1. Install from [python.org](https://python.org)
2. Check "Add Python to PATH" during install
3. Restart PowerShell/CMD

**Linux:**
```bash
sudo apt install python3.11
```

### pip Not Found

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

### Virtual Environment Won't Activate

**Windows:**
```powershell
# Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
# Ensure venv was created
python3 -m venv .venv

# Activate with source
source .venv/bin/activate
```

### Package Installation Fails

```bash
# Upgrade pip first
python -m pip install --upgrade pip setuptools wheel

# Try installing again
pip install -r requirements.txt

# If specific package fails, install separately
pip install package-name --no-cache-dir
```

### Models Not Found

```bash
# Verify models directory exists
ls models/

# Should contain 9 .pkl files
# If missing, run the Jupyter notebook:
jupyter notebook notebooks/CKD_PCA_Models.ipynb
# Run all cells to generate models
```

### Port Already in Use

```bash
# Find process using port 7870
netstat -ano | findstr :7870  # Windows
lsof -i :7870                 # Linux/Mac

# Kill process or use different port
# Edit src/app.py, change server_port
```

---

## 🔄 Updating Installation

### Update Code

```bash
# Pull latest changes
git pull origin main

# Activate venv
.venv\Scripts\Activate.ps1

# Update dependencies
pip install -r requirements.txt --upgrade
```

### Update Models

```bash
# Retrain models
jupyter notebook notebooks/CKD_PCA_Models.ipynb
# Run all cells
```

---

## 🗑️ Uninstallation

### Remove Virtual Environment

```powershell
# Deactivate first
deactivate

# Remove directory
Remove-Item -Recurse -Force .venv
```

### Complete Removal

```powershell
# Windows
Remove-Item -Recurse -Force C:\Path\To\CDK

# Linux/Mac
rm -rf ~/path/to/CDK
```

---

## 📚 Next Steps After Installation

1. **📖 Read Documentation**
   - [README.md](../README.md) - Complete guide
   - [QUICKSTART.md](QUICKSTART.md) - Quick reference
   - [FLOWCHART.md](FLOWCHART.md) - System architecture

2. **🎯 Try Examples**
   - Launch app: `python src/app.py`
   - Try example cases in web interface
   - Experiment with different patient data

3. **🔬 Explore Notebook**
   - `jupyter notebook notebooks/CKD_PCA_Models.ipynb`
   - See complete ML pipeline
   - Retrain models with custom parameters

4. **🚀 Deploy**
   - See [DOCKER.md](DOCKER.md) for containerization
   - Configure for production environment
   - Set up monitoring and logging

---

## 💬 Need Help?

- **Documentation**: Check all .md files in docs/
- **Issues**: [GitHub Issues](https://github.com/e-RickReyJim/CDK/issues)
- **Community**: GitHub Discussions

---

**✅ Installation Complete! Ready to predict CKD stages!**
