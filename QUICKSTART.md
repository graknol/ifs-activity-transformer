# Quick Start Guide

Get up and running with IFS Activity Transformer in 5 minutes.

## Windows 11 + GPU (Recommended for your RTX 5070 Ti)

### 1. Install uv Package Manager
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and Setup (Automatic with uv sync!)
```powershell
git clone https://github.com/graknol/ifs-activity-transformer.git
cd ifs-activity-transformer

# Install PyTorch with CUDA first (for GPU support)
uv pip install torch --index-url https://download.pytorch.org/whl/cu130

# Automatically create venv and install ALL dependencies!
uv sync

# Configure
copy .env.example .env
```

**That's it!** `uv sync` automatically:
- ✅ Creates a virtual environment (`.venv`)
- ✅ Installs Python 3.11 (specified in `.python-version`)
- ✅ Installs all dependencies from `pyproject.toml`
- ✅ Installs dev/test dependencies
- ✅ Creates a lockfile (`uv.lock`) for reproducible builds

### Alternative: Manual Setup
```powershell
# If you prefer step-by-step control
uv venv --python 3.11
.venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt
copy .env.example .env
```

### 3. Edit .env (Minimal Config)
```env
FLASK_SECRET_KEY=your-secret-key-here
FLASK_ENV=development
MODEL_NAME=google/bigbird-roberta-base
BATCH_SIZE=12
```

### 4. Activate Environment and Run!
```powershell
# Activate the virtual environment created by uv sync
.venv\Scripts\activate

# Run the application
python run.py
```

Open browser: **http://localhost:5000**

## Alternative: Automated Setup Script (Windows)

Just run:
```powershell
setup_windows.bat
```

This script:
- ✅ Creates virtual environment
- ✅ Installs PyTorch with CUDA
- ✅ Installs all dependencies
- ✅ Creates .env file
- ✅ Verifies GPU detection

Then:
```powershell
python run.py
```

## What Happens on First Run?

```
Starting IFS Activity Transformer...
======================================================================
SYSTEM INFORMATION
======================================================================

Python Environment:
  PyTorch version: 2.1.0+cu130

Compute Device:
  Selected device: cuda
  ✓ GPU detected: NVIDIA GeForce RTX 5070 Ti
  CUDA version: 13.0
  GPU memory: 16.00 GB

Model Cache:
  Location: C:\Users\YourName\.cache\huggingface
  Size: 0 GB (empty)

Recommended Settings:
  Batch size: 12
  Mixed precision: Enabled (float16)

======================================================================

Preloading model: google/bigbird-roberta-base
  Downloading/verifying tokenizer...
  ✓ Tokenizer ready
  Downloading/verifying model weights...
  [Download progress shown here - ~1.5GB]
  ✓ Model ready

Application initialization complete!
 * Running on http://0.0.0.0:5000
```

**First run downloads ~1.5GB of models. Subsequent runs start instantly!**

## Using the Application

### 1. Load Data
- **Data Page**: Connect to Oracle DB or upload CSV
- SQL query is customizable on-the-fly

### 2. Annotate Activities
- **Annotate Page**: Label activities with categories
- Model shows predictions alongside your labels
- Active learning prioritizes uncertain samples
- Keyboard shortcuts: 1-8 for categories, Ctrl+arrows for navigation

### 3. Train Model
- **Train Page**: Configure and start training
- Real-time progress monitoring
- Your RTX 5070 Ti will train 10-20x faster than CPU!

### 4. Make Predictions
- **Predict Page**: Classify new activities
- Single or batch predictions
- Confidence scores included
- Export results to CSV

## Performance on Your Hardware

**RTX 5070 Ti (16GB) + Ryzen 9950X + 32GB RAM:**
- **Training speed**: ~50-100 samples/second (vs ~5-10 on CPU)
- **Recommended batch size**: 12 (auto-configured)
- **Training time**: ~10-15 minutes for 1000 samples
- **Memory usage**: ~8-10GB VRAM during training

## Troubleshooting

### GPU Not Detected?
```powershell
# Check NVIDIA drivers
nvidia-smi

# Reinstall PyTorch with CUDA
uv pip uninstall torch
uv pip install torch --index-url https://download.pytorch.org/whl/cu130

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Model Download Failed?
Check internet connection and disk space (~2GB needed).
Models auto-retry on next run.

### Port 5000 Busy?
Edit `.env`:
```env
FLASK_PORT=8080
```

## Next Steps

- 📖 Read [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for detailed Windows setup
- 📖 Read [README.md](README.md) for full documentation
- 📖 Check [.github/instructions.md](.github/instructions.md) for development patterns

## Support

**For your specific setup (Windows 11 + RTX 5070 Ti + uv):**
- All dependencies auto-resolve with uv
- GPU auto-detected and configured
- Models auto-download from HuggingFace
- Everything should "just work" 🚀

If you encounter issues:
1. Check `nvidia-smi` shows your GPU
2. Verify CUDA installed: `python -c "import torch; print(torch.version.cuda)"`
3. Check model cache: `%USERPROFILE%\.cache\huggingface\`
