"""
Pytest configuration and shared fixtures.

This module provides reusable fixtures and configuration for all tests,
following the dependency injection pattern used throughout the application.
"""
import pytest
import os
import tempfile
import pandas as pd
from unittest.mock import Mock, MagicMock
from pathlib import Path

# Add app to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import Config, DatabaseConfig, ModelConfig, FlaskConfig, AzureConfig
from services.container import ServiceContainer
from app.utils import DataManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_env(monkeypatch):
    """Set up mock environment variables for testing."""
    test_env = {
        'ORACLE_USER': 'test_user',
        'ORACLE_PASSWORD': 'test_password',
        'ORACLE_HOST': 'localhost',
        'ORACLE_PORT': '1521',
        'ORACLE_SERVICE': 'test_service',
        'MODEL_NAME': 'google/bigbird-roberta-base',
        'MAX_LENGTH': '512',
        'BATCH_SIZE': '8',
        'LEARNING_RATE': '0.00002',
        'NUM_EPOCHS': '3',
        'FLASK_SECRET_KEY': 'test-secret-key',
        'FLASK_ENV': 'testing',
        'FLASK_HOST': '0.0.0.0',
        'FLASK_PORT': '5000',
        'AZURE_STORAGE_CONNECTION_STRING': 'test_connection_string',
        'AZURE_CONTAINER_NAME': 'test-container',
        'BACKUP_ENABLED': 'true',
        'AUTO_BACKUP_THRESHOLD': '10'
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    return test_env


@pytest.fixture
def database_config():
    """Create a test database configuration."""
    return DatabaseConfig(
        user='test_user',
        password='test_password',
        host='localhost',
        port=1521,
        service='test_service'
    )


@pytest.fixture
def model_config():
    """Create a test model configuration."""
    return ModelConfig(
        model_name='google/bigbird-roberta-base',
        max_length=512,
        batch_size=8,
        learning_rate=2e-5,
        num_epochs=3,
        save_dir='./test_models'
    )


@pytest.fixture
def flask_config():
    """Create a test Flask configuration."""
    return FlaskConfig(
        secret_key='test-secret-key',
        debug=False,
        testing=True,
        host='0.0.0.0',
        port=5000
    )


@pytest.fixture
def azure_config():
    """Create a test Azure configuration."""
    return AzureConfig(
        connection_string='test_connection_string',
        container_name='test-container',
        backup_enabled=True,
        backup_threshold=10
    )


@pytest.fixture
def service_container():
    """Create a fresh service container for each test."""
    container = ServiceContainer()
    yield container
    container.clear()


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'activity_id': ['ACT001', 'ACT002', 'ACT003'],
        'activity_desc': ['Design phase', 'Construction work', 'Testing activities'],
        'category': ['engineering', 'construction', 'milestone'],
        'num_jobcards': [5, 12, 3],
        'num_procurement': [2, 8, 1]
    })


@pytest.fixture
def sample_annotations():
    """Create a sample annotations DataFrame."""
    return pd.DataFrame({
        'activity_id': ['ACT001', 'ACT002'],
        'categories': ['engineering,procurement', 'construction'],
        'uncertainty_score': [0.25, 0.15],
        'timestamp': ['2026-01-01T10:00:00', '2026-01-01T10:05:00'],
        'label': [1, 1]
    })


@pytest.fixture
def mock_flask_app():
    """Create a mock Flask application for testing routes."""
    from flask import Flask
    app = Flask(__name__)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def mock_db_connection():
    """Create a mock database connection."""
    mock_conn = MagicMock()
    mock_conn.is_connected.return_value = True
    mock_conn.test_connection.return_value = (True, "Connection successful")
    mock_conn.execute_query.return_value = pd.DataFrame({
        'activity_id': ['ACT001'],
        'activity_desc': ['Test activity']
    })
    return mock_conn


@pytest.fixture
def mock_model():
    """Create a mock ML model for testing."""
    import numpy as np
    
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.1, 0.7, 0.2]])
    mock_model.train.return_value = {'loss': 0.5, 'accuracy': 0.85}
    return mock_model


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Cleanup test data files after each test."""
    yield
    # Cleanup any test files created
    test_files = [
        'data/test_training_data.csv',
        'data/test_annotations.csv',
        'test_models/'
    ]
    for file_path in test_files:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
