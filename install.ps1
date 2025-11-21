<#
.SYNOPSIS
    Complete installation script for CKD Stage Predictor

.DESCRIPTION
    This script automates the complete setup of the CKD Stage Predictor on a new machine.
    It handles Python installation check, virtual environment creation, dependency installation,
    and application launch.

.PARAMETER SkipPythonCheck
    Skip Python version verification (use with caution)

.PARAMETER NoLaunch
    Install but don't launch the application

.PARAMETER DevMode
    Install development dependencies (pytest, jupyter, etc.)

.EXAMPLE
    .\install.ps1
    Standard installation with all checks

.EXAMPLE
    .\install.ps1 -NoLaunch
    Install without launching the application

.EXAMPLE
    .\install.ps1 -DevMode
    Install with development tools

.NOTES
    Author: CKD ML Team
    Version: 1.0.0
    Requires: PowerShell 5.1+, Python 3.11+
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$false)]
    [switch]$SkipPythonCheck,
    
    [Parameter(Mandatory=$false)]
    [switch]$NoLaunch,
    
    [Parameter(Mandatory=$false)]
    [switch]$DevMode
)

# Color functions
function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Info { param($Message) Write-Host "ℹ️  $Message" -ForegroundColor Cyan }
function Write-Warning { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }
function Write-Step { param($Message) Write-Host "`n🔹 $Message" -ForegroundColor Magenta }

# Banner
function Show-Banner {
    Write-Host @"

╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║       🏥 CKD Stage Predictor - Installation Script            ║
║                                                                ║
║       AI-Powered 5-Stage CKD Classification                    ║
║       Version 1.0.0                                            ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan
}

# Error handler
$ErrorActionPreference = "Stop"
trap {
    Write-Error "Installation failed: $_"
    Write-Host "`nPlease check the error message above and try again." -ForegroundColor Yellow
    exit 1
}

# Main installation function
function Install-CKDPredictor {
    Show-Banner
    
    # Step 1: Check prerequisites
    Write-Step "Checking Prerequisites"
    
    if (-not $SkipPythonCheck) {
        Write-Info "Checking Python installation..."
        
        try {
            $pythonVersion = python --version 2>&1
            Write-Success "Found: $pythonVersion"
            
            # Extract version number
            if ($pythonVersion -match "Python (\d+)\.(\d+)\.(\d+)") {
                $major = [int]$matches[1]
                $minor = [int]$matches[2]
                
                if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
                    Write-Error "Python 3.11+ required. Found: $pythonVersion"
                    Write-Warning "Please install Python 3.11+ from https://www.python.org/downloads/"
                    exit 1
                }
            }
        }
        catch {
            Write-Error "Python not found in PATH"
            Write-Warning "Please install Python 3.11+ from https://www.python.org/downloads/"
            Write-Info "Make sure to check 'Add Python to PATH' during installation"
            exit 1
        }
    }
    
    # Check pip
    Write-Info "Checking pip..."
    try {
        $pipVersion = python -m pip --version 2>&1
        Write-Success "pip is available: $pipVersion"
    }
    catch {
        Write-Error "pip not found"
        Write-Warning "Installing pip..."
        python -m ensurepip --upgrade
    }
    
    # Step 2: Create virtual environment
    Write-Step "Setting Up Virtual Environment"
    
    if (Test-Path ".venv") {
        Write-Warning "Virtual environment already exists"
        $response = Read-Host "Recreate? (y/N)"
        if ($response -eq 'y' -or $response -eq 'Y') {
            Write-Info "Removing existing virtual environment..."
            Remove-Item -Recurse -Force .venv
            Write-Success "Removed"
        }
        else {
            Write-Info "Using existing virtual environment"
        }
    }
    
    if (-not (Test-Path ".venv")) {
        Write-Info "Creating virtual environment..."
        python -m venv .venv
        Write-Success "Virtual environment created"
    }
    
    # Step 3: Activate virtual environment
    Write-Step "Activating Virtual Environment"
    
    $activateScript = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        Write-Info "Activating virtual environment..."
        & $activateScript
        Write-Success "Virtual environment activated"
    }
    else {
        Write-Error "Activation script not found: $activateScript"
        exit 1
    }
    
    # Step 4: Upgrade pip
    Write-Step "Upgrading pip"
    
    Write-Info "Upgrading pip to latest version..."
    python -m pip install --upgrade pip --quiet
    Write-Success "pip upgraded"
    
    # Step 5: Install dependencies
    Write-Step "Installing Dependencies"
    
    if (Test-Path "requirements.txt") {
        Write-Info "Installing Python packages from requirements.txt..."
        Write-Host "This may take several minutes..." -ForegroundColor Yellow
        
        python -m pip install -r requirements.txt --quiet
        Write-Success "All dependencies installed"
    }
    else {
        Write-Error "requirements.txt not found"
        exit 1
    }
    
    # Step 6: Install development dependencies (optional)
    if ($DevMode) {
        Write-Step "Installing Development Tools"
        
        Write-Info "Installing development dependencies..."
        python -m pip install --quiet pytest pytest-mock jupyter black flake8
        Write-Success "Development tools installed"
    }
    
    # Step 7: Verify installation
    Write-Step "Verifying Installation"
    
    Write-Info "Checking project structure..."
    
    $requiredDirs = @("src", "models", "data", "notebooks")
    $requiredFiles = @("src\app.py", "src\utils.py", "models\best_model.pkl")
    
    foreach ($dir in $requiredDirs) {
        if (Test-Path $dir) {
            Write-Success "Found directory: $dir"
        }
        else {
            Write-Warning "Missing directory: $dir"
        }
    }
    
    foreach ($file in $requiredFiles) {
        if (Test-Path $file) {
            Write-Success "Found file: $file"
        }
        else {
            Write-Error "Missing required file: $file"
            exit 1
        }
    }
    
    # Step 8: Test import
    Write-Info "Testing Python imports..."
    
    $testScript = @"
import sys
try:
    import gradio
    import sklearn
    import pandas
    import numpy
    import matplotlib
    print('✅ All imports successful')
    sys.exit(0)
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"@
    
    $testResult = python -c $testScript
    if ($LASTEXITCODE -eq 0) {
        Write-Success "All Python packages verified"
    }
    else {
        Write-Error "Package verification failed"
        exit 1
    }
    
    # Step 9: Installation summary
    Write-Host "`n" -NoNewline
    Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║                                                                ║" -ForegroundColor Green
    Write-Host "║                  ✅ INSTALLATION COMPLETE!                     ║" -ForegroundColor Green
    Write-Host "║                                                                ║" -ForegroundColor Green
    Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    
    Write-Info "Installation Summary:"
    Write-Host "  📦 Virtual environment: .venv" -ForegroundColor White
    Write-Host "  🐍 Python version: $(python --version)" -ForegroundColor White
    Write-Host "  📚 Packages installed: $(python -m pip list --format=freeze | Measure-Object -Line | Select-Object -ExpandProperty Lines)" -ForegroundColor White
    Write-Host "  📂 Project directory: $PSScriptRoot" -ForegroundColor White
    
    # Step 10: Launch application (optional)
    if (-not $NoLaunch) {
        Write-Host ""
        Write-Step "Launching Application"
        
        $response = Read-Host "Launch CKD Stage Predictor now? (Y/n)"
        if ($response -ne 'n' -and $response -ne 'N') {
            Write-Info "Starting Gradio application..."
            Write-Host "Opening http://127.0.0.1:7870 in your browser..." -ForegroundColor Cyan
            Write-Host "Press Ctrl+C to stop the application`n" -ForegroundColor Yellow
            
            Start-Sleep -Seconds 2
            python src\app.py
        }
    }
    
    # Step 11: Next steps
    Write-Host "`n" -NoNewline
    Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║                     📝 NEXT STEPS                              ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To run the application:" -ForegroundColor Yellow
    Write-Host "  1. Activate virtual environment:" -ForegroundColor White
    Write-Host "     .venv\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  2. Run the application:" -ForegroundColor White
    Write-Host "     python src\app.py" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  3. Open your browser to:" -ForegroundColor White
    Write-Host "     http://127.0.0.1:7870" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "For more information:" -ForegroundColor Yellow
    Write-Host "  📖 Read README.md for complete documentation" -ForegroundColor White
    Write-Host "  🚀 See docs/QUICKSTART.md for quick reference" -ForegroundColor White
    Write-Host "  📊 Check docs/FLOWCHART.md for system architecture" -ForegroundColor White
    Write-Host ""
    Write-Host "Need help? Open an issue on GitHub!" -ForegroundColor Green
    Write-Host ""
}

# Run installation
try {
    Install-CKDPredictor
    exit 0
}
catch {
    Write-Error "Installation failed: $_"
    Write-Host "`nFor help, see README.md or open an issue on GitHub" -ForegroundColor Yellow
    exit 1
}
