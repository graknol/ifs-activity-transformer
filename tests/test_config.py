"""
Unit tests for configuration module (app/config.py).

Tests the type-safe configuration management with environment variable parsing.
"""
import pytest
import os
from app.config import (
    Config, DatabaseConfig, ModelConfig, FlaskConfig, AzureConfig,
    get_int_env, get_float_env
)


class TestEnvironmentHelpers:
    """Test environment variable helper functions."""
    
    def test_get_int_env_with_value(self, monkeypatch):
        """Test get_int_env with a valid integer."""
        monkeypatch.setenv('TEST_INT', '42')
        assert get_int_env('TEST_INT', 10) == 42
    
    def test_get_int_env_with_default(self):
        """Test get_int_env returns default when not set."""
        assert get_int_env('NONEXISTENT_VAR', 100) == 100
    
    def test_get_float_env_with_value(self, monkeypatch):
        """Test get_float_env with a valid float."""
        monkeypatch.setenv('TEST_FLOAT', '3.14')
        assert get_float_env('TEST_FLOAT', 1.0) == pytest.approx(3.14)
    
    def test_get_float_env_with_default(self):
        """Test get_float_env returns default when not set."""
        assert get_float_env('NONEXISTENT_VAR', 2.5) == pytest.approx(2.5)
    
    def test_get_int_env_with_scientific_notation(self, monkeypatch):
        """Test get_int_env with scientific notation."""
        monkeypatch.setenv('TEST_SCI', '1e2')
        assert get_int_env('TEST_SCI', 0) == 100


class TestDatabaseConfig:
    """Test DatabaseConfig class."""
    
    def test_from_env_with_all_values(self, mock_env):
        """Test creating config from environment with all values set."""
        config = DatabaseConfig.from_env()
        assert config.user == 'test_user'
        assert config.password == 'test_password'
        assert config.host == 'localhost'
        assert config.port == 1521
        assert config.service == 'test_service'
    
    def test_from_env_with_defaults(self, monkeypatch):
        """Test creating config with default values."""
        # Clear all Oracle env vars
        for key in ['ORACLE_USER', 'ORACLE_PASSWORD', 'ORACLE_HOST', 'ORACLE_SERVICE']:
            monkeypatch.delenv(key, raising=False)
        
        config = DatabaseConfig.from_env()
        assert config.user is None
        assert config.password is None
        assert config.host is None
        assert config.port == 1521  # Default port
        assert config.service is None
    
    def test_to_dict(self, database_config):
        """Test converting config to dictionary."""
        config_dict = database_config.to_dict()
        assert config_dict['user'] == 'test_user'
        assert config_dict['password'] == 'test_password'
        assert config_dict['host'] == 'localhost'
        assert config_dict['port'] == 1521
        assert config_dict['service'] == 'test_service'
    
    def test_custom_port(self, monkeypatch):
        """Test custom database port."""
        monkeypatch.setenv('ORACLE_PORT', '1530')
        config = DatabaseConfig.from_env()
        assert config.port == 1530


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_from_env_with_all_values(self, mock_env):
        """Test creating config from environment with all values set."""
        config = ModelConfig.from_env()
        assert config.model_name == 'google/bigbird-roberta-base'
        assert config.max_length == 512
        assert config.batch_size == 8
        assert config.learning_rate == pytest.approx(2e-5)
        assert config.num_epochs == 3
    
    def test_from_env_with_defaults(self, monkeypatch):
        """Test creating config with default values."""
        # Clear all model env vars
        for key in ['MODEL_NAME', 'MAX_LENGTH', 'BATCH_SIZE', 'LEARNING_RATE', 'NUM_EPOCHS']:
            monkeypatch.delenv(key, raising=False)
        
        config = ModelConfig.from_env()
        assert config.model_name == 'google/bigbird-roberta-base'
        assert config.max_length == 512
        assert config.batch_size == 8
        assert config.learning_rate == pytest.approx(2e-5)
        assert config.num_epochs == 3
    
    def test_to_dict(self, model_config):
        """Test converting config to dictionary."""
        config_dict = model_config.to_dict()
        assert config_dict['model_name'] == 'google/bigbird-roberta-base'
        assert config_dict['max_length'] == 512
        assert config_dict['batch_size'] == 8
        assert config_dict['learning_rate'] == pytest.approx(2e-5)
        assert config_dict['num_epochs'] == 3
    
    def test_custom_hyperparameters(self, monkeypatch):
        """Test custom hyperparameters."""
        monkeypatch.setenv('BATCH_SIZE', '16')
        monkeypatch.setenv('LEARNING_RATE', '0.0001')
        monkeypatch.setenv('NUM_EPOCHS', '5')
        
        config = ModelConfig.from_env()
        assert config.batch_size == 16
        assert config.learning_rate == pytest.approx(1e-4)
        assert config.num_epochs == 5


class TestFlaskConfig:
    """Test FlaskConfig class."""
    
    def test_from_env_development(self, monkeypatch):
        """Test Flask config in development mode."""
        monkeypatch.setenv('FLASK_ENV', 'development')
        config = FlaskConfig.from_env()
        assert config.debug is True
        assert config.testing is False
    
    def test_from_env_production(self, monkeypatch):
        """Test Flask config in production mode."""
        monkeypatch.setenv('FLASK_ENV', 'production')
        config = FlaskConfig.from_env()
        assert config.debug is False
        assert config.testing is False
    
    def test_from_env_testing(self, monkeypatch):
        """Test Flask config in testing mode."""
        monkeypatch.setenv('TESTING', 'true')
        config = FlaskConfig.from_env()
        assert config.testing is True
    
    def test_custom_host_and_port(self, monkeypatch):
        """Test custom host and port."""
        monkeypatch.setenv('FLASK_HOST', '127.0.0.1')
        monkeypatch.setenv('FLASK_PORT', '8080')
        config = FlaskConfig.from_env()
        assert config.host == '127.0.0.1'
        assert config.port == 8080
    
    def test_default_secret_key(self, monkeypatch):
        """Test default secret key."""
        monkeypatch.delenv('FLASK_SECRET_KEY', raising=False)
        config = FlaskConfig.from_env()
        assert config.secret_key == 'dev-secret-key-change-in-production'


class TestAzureConfig:
    """Test AzureConfig class."""
    
    def test_from_env_with_connection_string(self, mock_env):
        """Test config with Azure connection string."""
        config = AzureConfig.from_env()
        assert config.connection_string == 'test_connection_string'
        assert config.container_name == 'test-container'
        assert config.backup_enabled is True
        assert config.backup_threshold == 10
    
    def test_from_env_without_connection_string(self, monkeypatch):
        """Test config without Azure connection string."""
        monkeypatch.delenv('AZURE_STORAGE_CONNECTION_STRING', raising=False)
        monkeypatch.delenv('BACKUP_ENABLED', raising=False)
        config = AzureConfig.from_env()
        assert config.connection_string is None
        assert config.backup_enabled is False  # Auto-disabled without connection
    
    def test_explicit_backup_enabled(self, monkeypatch):
        """Test explicit backup enabled/disabled setting."""
        monkeypatch.setenv('AZURE_STORAGE_CONNECTION_STRING', 'test_string')
        monkeypatch.setenv('BACKUP_ENABLED', 'false')
        config = AzureConfig.from_env()
        assert config.backup_enabled is False  # Explicit setting overrides
    
    def test_backup_enabled_true_without_connection(self, monkeypatch):
        """Test backup_enabled=true without connection string."""
        monkeypatch.delenv('AZURE_STORAGE_CONNECTION_STRING', raising=False)
        monkeypatch.setenv('BACKUP_ENABLED', 'true')
        config = AzureConfig.from_env()
        assert config.backup_enabled is True  # Explicit user setting honored
    
    def test_custom_container_name(self, monkeypatch):
        """Test custom container name."""
        monkeypatch.setenv('AZURE_CONTAINER_NAME', 'my-custom-container')
        config = AzureConfig.from_env()
        assert config.container_name == 'my-custom-container'
    
    def test_custom_backup_threshold(self, monkeypatch):
        """Test custom backup threshold."""
        monkeypatch.setenv('AUTO_BACKUP_THRESHOLD', '25')
        config = AzureConfig.from_env()
        assert config.backup_threshold == 25


class TestConfig:
    """Test main Config class."""
    
    def test_from_env_creates_all_configs(self, mock_env):
        """Test that Config.from_env creates all sub-configurations."""
        config = Config.from_env()
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.flask, FlaskConfig)
        assert isinstance(config.azure, AzureConfig)
    
    def test_config_repr(self, mock_env):
        """Test config string representation."""
        config = Config.from_env()
        repr_str = repr(config)
        assert 'Config(' in repr_str
        assert 'database=<DatabaseConfig>' in repr_str
        assert 'flask=<FlaskConfig' in repr_str
        assert 'azure=<AzureConfig' in repr_str
    
    def test_config_initialization(self):
        """Test direct config initialization."""
        config = Config()
        assert hasattr(config, 'database')
        assert hasattr(config, 'model')
        assert hasattr(config, 'flask')
        assert hasattr(config, 'azure')
