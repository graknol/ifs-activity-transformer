"""
Model management and initialization for Windows/GPU support.

This module handles automatic model downloading, GPU detection,
and ensures a smooth "just works" experience on Windows with RTX GPUs.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Centralized model management for HuggingFace models.
    
    Handles:
    - Automatic model downloading from HuggingFace Hub
    - GPU detection and configuration
    - Model caching and validation
    - Progress indication for downloads
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model manager.
        
        Args:
            cache_dir: Optional cache directory for models. 
                      If None, uses HuggingFace default (~/.cache/huggingface)
        """
        self.cache_dir = cache_dir
        self.device = self._detect_device()
        
        logger.info(f"Model Manager initialized - Device: {self.device}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def _has_gpu(self) -> bool:
        """Check if GPU is available and accessible."""
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    
    def _detect_device(self) -> str:
        """
        Detect available compute device.
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if self._has_gpu():
            device = 'cuda'
            logger.info(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = 'cpu'
            logger.info("⚠ No GPU detected, using CPU (training will be slower)")
            logger.info("  For GPU support, install: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        
        return device
    
    def get_device(self) -> str:
        """Get the current device."""
        return self.device
    
    def ensure_model_available(
        self, 
        model_name: str,
        model_type: str = 'both'
    ) -> Tuple[bool, str]:
        """
        Ensure model is downloaded and available.
        
        Args:
            model_name: HuggingFace model identifier (e.g., 'google/bigbird-roberta-base')
            model_type: Type to check ('tokenizer', 'model', or 'both')
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            logger.info(f"Checking model availability: {model_name}")
            
            if model_type in ('tokenizer', 'both'):
                logger.info("  Downloading/verifying tokenizer...")
                AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir
                )
                logger.info("  ✓ Tokenizer ready")
            
            if model_type in ('model', 'both'):
                logger.info("  Downloading/verifying model weights...")
                AutoModel.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir
                )
                logger.info("  ✓ Model ready")
            
            return True, f"Model {model_name} is ready to use"
            
        except Exception as e:
            error_msg = f"Failed to download model {model_name}: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def preload_models(self, model_names: list) -> Dict[str, Tuple[bool, str]]:
        """
        Preload multiple models at application startup.
        
        Args:
            model_names: List of HuggingFace model identifiers
            
        Returns:
            Dictionary mapping model names to (success, message) tuples
        """
        results = {}
        
        logger.info("=" * 60)
        logger.info("PRELOADING MODELS")
        logger.info("=" * 60)
        
        for model_name in model_names:
            logger.info(f"\nProcessing: {model_name}")
            success, message = self.ensure_model_available(model_name)
            results[model_name] = (success, message)
            
            if success:
                logger.info(f"✓ {model_name} ready")
            else:
                logger.error(f"✗ {model_name} failed: {message}")
        
        logger.info("\n" + "=" * 60)
        logger.info("MODEL PRELOADING COMPLETE")
        logger.info("=" * 60)
        
        return results
    
    def get_optimal_batch_size(self) -> int:
        """
        Recommend optimal batch size based on available hardware.
        
        Returns:
            Recommended batch size
        """
        if self.device == 'cpu':
            return 4  # Smaller batch size for CPU
        
        # GPU - check memory
        if self._has_gpu():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if gpu_memory_gb >= 24:  # High-end GPU
                return 16
            elif gpu_memory_gb >= 12:  # Mid-range GPU (like RTX 5070 Ti)
                return 12
            elif gpu_memory_gb >= 8:  # Entry GPU
                return 8
            else:
                return 4
        
        return 8  # Default
    
    def print_system_info(self):
        """Print comprehensive system information for debugging."""
        print("\n" + "=" * 70)
        print("SYSTEM INFORMATION")
        print("=" * 70)
        
        print(f"\nPython Environment:")
        print(f"  PyTorch version: {torch.__version__}")
        
        print(f"\nCompute Device:")
        print(f"  Selected device: {self.device}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n  GPU {i}: {props.name}")
                print(f"    Total memory: {props.total_memory / 1e9:.2f} GB")
                print(f"    Compute capability: {props.major}.{props.minor}")
        else:
            print("  ⚠ No GPU detected")
            print("  To enable GPU support on Windows:")
            print("    pip install torch --index-url https://download.pytorch.org/whl/cu121")
        
        print(f"\nModel Cache:")
        default_cache = os.path.expanduser("~/.cache/huggingface")
        cache_size = self._get_dir_size(default_cache)
        print(f"  Location: {default_cache}")
        print(f"  Size: {cache_size / 1e9:.2f} GB" if cache_size > 0 else "  Size: 0 GB (empty)")
        
        print(f"\nRecommended Settings:")
        print(f"  Batch size: {self.get_optimal_batch_size()}")
        print(f"  Mixed precision: {'Enabled (float16)' if self.device == 'cuda' else 'Disabled (float32)'}")
        
        print("\n" + "=" * 70 + "\n")
    
    def _get_dir_size(self, path: str) -> int:
        """Get total size of directory in bytes."""
        total = 0
        try:
            for entry in os.scandir(path):
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += self._get_dir_size(entry.path)
        except (FileNotFoundError, PermissionError):
            pass
        return total


# Singleton instance
_model_manager = None


def get_model_manager(cache_dir: Optional[str] = None) -> ModelManager:
    """
    Get or create the global ModelManager instance.
    
    Args:
        cache_dir: Optional cache directory for models
        
    Returns:
        ModelManager instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(cache_dir)
    return _model_manager
