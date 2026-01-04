"""
Unit tests for model manager (app/model_manager.py).

Tests the automatic model management and GPU detection.
"""
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from app.model_manager import ModelManager


class TestModelManager:
    """Test ModelManager class."""
    
    @pytest.fixture
    def model_manager(self):
        """Create a model manager for testing."""
        return ModelManager()
    
    def test_initialization(self, model_manager):
        """Test ModelManager initialization."""
        assert model_manager.cache_dir is None
        assert model_manager.device in ['cuda', 'cpu']
    
    def test_has_gpu_with_cuda(self, model_manager):
        """Test GPU detection when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                assert model_manager._has_gpu() is True
    
    def test_has_gpu_without_cuda(self, model_manager):
        """Test GPU detection when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            assert model_manager._has_gpu() is False
    
    def test_has_gpu_with_no_devices(self, model_manager):
        """Test GPU detection when CUDA available but no devices."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=0):
                assert model_manager._has_gpu() is False
    
    def test_detect_device_cuda(self, model_manager):
        """Test device detection returns cuda when available."""
        with patch.object(model_manager, '_has_gpu', return_value=True):
            with patch('torch.cuda.get_device_name', return_value='NVIDIA RTX 5070 Ti'):
                with patch('torch.version.cuda', '12.1'):
                    with patch('torch.cuda.get_device_properties') as mock_props:
                        mock_props.return_value.total_memory = 16 * 1024**3
                        device = model_manager._detect_device()
                        assert device == 'cuda'
    
    def test_detect_device_cpu(self, model_manager):
        """Test device detection returns cpu when GPU not available."""
        with patch.object(model_manager, '_has_gpu', return_value=False):
            device = model_manager._detect_device()
            assert device == 'cpu'
    
    def test_get_device(self, model_manager):
        """Test getting current device."""
        device = model_manager.get_device()
        assert device in ['cuda', 'cpu']
    
    def test_get_system_info(self, model_manager):
        """Test getting system information."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                with patch('torch.cuda.get_device_name', return_value='Test GPU'):
                    info = model_manager.get_system_info()
                    
                    assert 'device' in info
                    assert 'cuda_available' in info
                    assert 'pytorch_version' in info
    
    def test_get_system_info_cpu_only(self, model_manager):
        """Test system info when only CPU is available."""
        with patch('torch.cuda.is_available', return_value=False):
            info = model_manager.get_system_info()
            
            assert info['device'] == 'cpu'
            assert info['cuda_available'] is False
            assert 'cuda_version' not in info or info['cuda_version'] is None
    
    def test_recommend_batch_size_with_gpu(self, model_manager):
        """Test batch size recommendation with GPU."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                # Simulate 16GB VRAM
                mock_props.return_value.total_memory = 16 * 1024**3
                batch_size = model_manager.recommend_batch_size()
                assert batch_size > 0
                assert batch_size <= 32  # Reasonable upper bound
    
    def test_recommend_batch_size_with_cpu(self, model_manager):
        """Test batch size recommendation with CPU only."""
        with patch('torch.cuda.is_available', return_value=False):
            batch_size = model_manager.recommend_batch_size()
            assert batch_size > 0
            assert batch_size <= 8  # CPU should recommend smaller batches
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_ensure_model_available(self, mock_model, mock_tokenizer, model_manager):
        """Test ensuring model is available."""
        mock_tokenizer.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        success, message = model_manager.ensure_model_available('bert-base-uncased')
        
        assert success is True
        assert 'available' in message.lower() or 'downloaded' in message.lower()
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_tokenizer(self, mock_from_pretrained, model_manager):
        """Test loading tokenizer."""
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer
        
        tokenizer = model_manager.load_tokenizer('bert-base-uncased')
        
        assert tokenizer is not None
        mock_from_pretrained.assert_called_once()
    
    @patch('transformers.AutoModel.from_pretrained')
    def test_load_model(self, mock_from_pretrained, model_manager):
        """Test loading model."""
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model
        
        model = model_manager.load_model('bert-base-uncased')
        
        assert model is not None
        mock_from_pretrained.assert_called_once()
    
    def test_custom_cache_dir(self):
        """Test ModelManager with custom cache directory."""
        custom_cache = '/custom/cache/dir'
        manager = ModelManager(cache_dir=custom_cache)
        assert manager.cache_dir == custom_cache


class TestModelManagerIntegration:
    """Integration tests for ModelManager (may require network)."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_download_small_model(self):
        """Test downloading a small model (requires network)."""
        manager = ModelManager()
        # Use a very small model for testing
        success, message = manager.ensure_model_available('prajjwal1/bert-tiny')
        assert success is True
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_load_small_model(self):
        """Test loading a small model (requires network)."""
        manager = ModelManager()
        tokenizer = manager.load_tokenizer('prajjwal1/bert-tiny')
        model = manager.load_model('prajjwal1/bert-tiny')
        
        assert tokenizer is not None
        assert model is not None
