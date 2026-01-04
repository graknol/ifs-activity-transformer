@echo off
REM IFS Activity Transformer - Windows Setup Script
REM This script automates the setup process for Windows 11 users

echo ======================================================================
echo IFS Activity Transformer - Windows Setup
echo ======================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or 3.11 first
    pause
    exit /b 1
)

echo [1/6] Python found: 
python --version
echo.

REM Check if we're in a virtual environment
if defined VIRTUAL_ENV (
    echo [2/6] Virtual environment already active: %VIRTUAL_ENV%
) else (
    echo [2/6] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
)
echo.

REM Activate virtual environment
if not defined VIRTUAL_ENV (
    echo [3/6] Activating virtual environment...
    call venv\Scripts\activate.bat
    if %errorlevel% neq 0 (
        echo ERROR: Failed to activate virtual environment
        pause
        exit /b 1
    )
)
echo.

REM Install PyTorch with CUDA support
echo [4/6] Installing PyTorch with CUDA 13.0 support...
echo This may take a few minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
if %errorlevel% neq 0 (
    echo WARNING: PyTorch installation failed
    echo Continuing with CPU-only version...
    pip install torch
)
echo.

REM Verify GPU detection
echo [5/6] Verifying GPU detection...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'None')"
echo.

REM Install application dependencies
echo [6/6] Installing application dependencies...
echo This may take a few minutes...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file from template...
    copy .env.example .env
    echo.
    echo IMPORTANT: Please edit .env file with your configuration
    echo Opening .env in notepad...
    timeout /t 2 >nul
    notepad .env
)

echo ======================================================================
echo Setup Complete!
echo ======================================================================
echo.
echo To start the application, run:
echo    python run.py
echo.
echo The application will:
echo  - Automatically detect your GPU
echo  - Download models from HuggingFace on first run
echo  - Start the web interface on http://localhost:5000
echo.
echo For detailed documentation, see: WINDOWS_SETUP.md
echo ======================================================================
echo.
pause
