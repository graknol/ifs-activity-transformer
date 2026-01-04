# Windows Setup Guide for IFS Activity Transformer

Complete guide for running this application on Windows 11 with GPU support (RTX 5070 Ti).

## Prerequisites

- **Windows 11**
- **Python 3.10 or 3.11** (managed by uv)
- **NVIDIA RTX 5070 Ti** (or any CUDA-compatible GPU)
- **32GB RAM** (you have this - perfect for training!)
- **~10GB free disk space** (for models and dependencies)

## Quick Start with uv sync (Recommended - One Command Setup!)

### 1. Install uv (Python Package Manager)

```powershell
# Install uv if you haven't already
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Navigate to Project

```powershell
git clone https://github.com/graknol/ifs-activity-transformer.git
cd ifs-activity-transformer
```

### 3. Install PyTorch with CUDA (One-Time Setup for GPU)

For your RTX 5070 Ti, install PyTorch with CUDA 12.1 support first:

```powershell
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Verify GPU detection:**
```powershell
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 5070 Ti
```

### 4. Automatic Environment Setup (Magic Command!)

```powershell
# This ONE command does EVERYTHING:
# - Creates .venv with Python 3.11 (from .python-version)
# - Installs all dependencies (from pyproject.toml)
# - Installs dev/test tools automatically
# - Creates uv.lock for reproducible builds
uv sync
```

**That's it!** No need to manually:
- Create virtual environments
- Install dependencies one by one
- Manage package versions
- Track what's installed

`uv sync` handles everything automatically based on `pyproject.toml`!

**What gets installed:**
- Core dependencies (Flask, Transformers, etc.)
- Database drivers (Oracle DB)
- Cloud storage (Azure Blob)
- Dev tools (pytest, black, flake8, etc.) - installed automatically!

### 5. Configure Environment Variables

```powershell
# Copy the example environment file
copy .env.example .env

# Edit .env with your settings (use notepad or your favorite editor)
notepad .env
```

**Minimal Configuration (to get started):**
```env
# Flask
FLASK_ENV=development
FLASK_SECRET_KEY=your-secret-key-here

# Model (default is fine)
MODEL_NAME=google/bigbird-roberta-base
BATCH_SIZE=12
```

**Optional - Add later if needed:**
- Oracle database credentials
- Azure Blob Storage for backups

### 7. Run the Application

```powershell
python run.py
```

**First Run Behavior:**
- The application will automatically detect your GPU
- It will download the RoBERTa BigBird model from HuggingFace (~1.5GB)
- Models are cached in `C:\Users\YourName\.cache\huggingface\`
- Subsequent runs will use the cached models (instant startup)

### 8. Open Your Browser

Navigate to: **http://localhost:5000**

## What Happens on First Run

When you start the application for the first time, you'll see:

```
Starting IFS Activity Transformer...
======================================================================
SYSTEM INFORMATION
======================================================================

Python Environment:
  PyTorch version: 2.1.0+cu121

Compute Device:
  Selected device: cuda
  CUDA available: True
  CUDA version: 12.1
  cuDNN version: 8902
  GPU count: 1

  GPU 0: NVIDIA GeForce RTX 5070 Ti
    Total memory: 16.00 GB
    Compute capability: 8.9

Model Cache:
  Location: C:\Users\YourName\.cache\huggingface
  Size: 0 GB (empty)

Recommended Settings:
  Batch size: 12
  Mixed precision: Enabled (float16)

======================================================================

Initializing model manager...
Preloading model: google/bigbird-roberta-base
======================================================================
PRELOADING MODELS
======================================================================

Processing: google/bigbird-roberta-base
  Downloading/verifying tokenizer...
  ✓ Tokenizer ready
  Downloading/verifying model weights...
  ✓ Model ready
✓ google/bigbird-roberta-base ready

======================================================================
MODEL PRELOADING COMPLETE
======================================================================

Application initialization complete!
 * Running on http://0.0.0.0:5000
```

## Troubleshooting

### GPU Not Detected

If you see `CUDA available: False`:

1. **Verify NVIDIA drivers are installed:**
   ```powershell
   nvidia-smi
   ```
   This should show your GPU and driver version.

2. **Reinstall PyTorch with CUDA:**
   ```powershell
   uv pip uninstall torch torchvision torchaudio
   uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Check CUDA installation:**
   - Your GPU drivers include CUDA runtime
   - PyTorch bundles cuDNN
   - You don't need to manually install CUDA Toolkit

### Model Download Issues

If model download fails:

1. **Check internet connection**
2. **Try manually downloading:**
   ```powershell
   python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('google/bigbird-roberta-base'); AutoModel.from_pretrained('google/bigbird-roberta-base')"
   ```
3. **Check disk space** (~2GB needed for model)

### Port Already in Use

If port 5000 is busy:

Edit `.env`:
```env
FLASK_PORT=8080
```

Then access at http://localhost:8080

### Memory Issues

If you run out of memory during training:

Edit `.env` to reduce batch size:
```env
BATCH_SIZE=4
```

Your RTX 5070 Ti with 16GB should handle batch size 12 comfortably, but reduce if needed.

## Performance Tips

### Optimal Settings for RTX 5070 Ti (16GB VRAM)

```env
# .env settings optimized for your hardware
BATCH_SIZE=12
NUM_EPOCHS=3
MAX_LENGTH=512
LEARNING_RATE=2e-5
```

### Training Speed

Expected training speeds on your system:
- **CPU only**: ~5-10 samples/second
- **With RTX 5070 Ti**: ~50-100 samples/second (10-20x faster!)

### Model Cache Location

Models are cached at:
```
C:\Users\YourName\.cache\huggingface\hub\
```

To change cache location, set environment variable:
```powershell
$env:TRANSFORMERS_CACHE="D:\models\huggingface"
```

## Using with uv (Best Practices)

### Install specific package
```powershell
uv pip install package-name
```

### Update all dependencies
```powershell
uv pip install -r requirements.txt --upgrade
```

### List installed packages
```powershell
uv pip list
```

### Create project.toml for uv (optional)
```toml
[project]
name = "ifs-activity-transformer"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
    "flask>=3.0.0",
    "transformers>=4.36.0",
    "torch>=2.1.0",
    # ... other deps
]
```

## Running Tests (Windows-Friendly)

We provide cross-platform test runners that work on Windows without requiring MinGW or WSL:

### Run Tests

```powershell
# Basic test run
python run_tests.py

# Run with coverage report
python run_tests.py --cov

# Run only fast tests (skip slow integration tests)
python run_tests.py --fast

# Alternatively, use the batch file
run_tests.bat
run_tests.bat --cov
```

### Other Test Commands

```powershell
# Run linters
python run_tests.py --lint

# Format code
python run_tests.py --format

# Type checking
python run_tests.py --type-check

# Security checks
python run_tests.py --security

# Clean up temporary files (cross-platform)
python run_tests.py --clean

# Run all checks (CI mode)
python run_tests.py --all
```

### Direct pytest (also works)

```powershell
# Run all tests
pytest

# Run specific test file
pytest tests/test_config.py

# Run with verbose output
pytest -v
```

**Note:** We intentionally avoid using Makefiles on Windows since they require MinGW or WSL. The `run_tests.py` script provides all the same functionality using pure Python.

## Features That "Just Work"

✅ **Automatic GPU Detection**: Detects and uses your RTX 5070 Ti automatically  
✅ **Model Auto-Download**: Downloads models from HuggingFace on first run  
✅ **Model Caching**: Reuses downloaded models on subsequent runs  
✅ **Memory Optimization**: Automatically sets optimal batch sizes  
✅ **Mixed Precision**: Uses float16 on GPU for 2x speed boost  
✅ **Progress Indication**: Shows download progress for models  
✅ **Error Recovery**: Graceful fallback to CPU if GPU unavailable  

## Next Steps

1. **Load your data**: Go to Data page and connect to Oracle or upload CSV
2. **Annotate activities**: Use the Annotate page with active learning
3. **Train the model**: Go to Train page and start training
4. **Make predictions**: Use the Predict page for classifications

## Support

For issues specific to:
- **Windows**: Check Windows Event Viewer for errors
- **GPU**: Run `nvidia-smi` to verify GPU is accessible
- **Python**: Ensure using Python 3.10 or 3.11 with uv
- **Models**: Check `%USERPROFILE%\.cache\huggingface\` for cached models

## Architecture Benefits

This application is designed to "just work":
- **Zero manual file placement**: No need to download and place model files
- **Automatic dependency resolution**: uv handles all package versions
- **Smart defaults**: Optimized settings for your RTX 5070 Ti
- **Graceful degradation**: Falls back to CPU if GPU unavailable
- **Clear feedback**: Detailed logging shows exactly what's happening

Enjoy training your activity classifier! 🚀
