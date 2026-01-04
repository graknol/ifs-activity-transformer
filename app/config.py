"""
Configuration classes for the application.
Following the configuration management pattern documented in .github/instructions.md
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def get_int_env(key: str, default: int) -> int:
    """
    Get integer environment variable with default.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        
    Returns:
        Integer value from environment or default
    """
    return int(os.getenv(key, str(default)))


def get_float_env(key: str, default: float) -> float:
    """
    Get float environment variable with default.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        
    Returns:
        Float value from environment or default
    """
    return float(os.getenv(key, str(default)))


@dataclass
class DatabaseConfig:
    """Database configuration with environment variable defaults."""
    
    user: Optional[str] = None
    password: Optional[str] = None
    host: Optional[str] = None
    port: int = 1521
    service: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """
        Create configuration from environment variables.
        
        Returns:
            DatabaseConfig instance with values from environment
        """
        return cls(
            user=os.getenv('ORACLE_USER'),
            password=os.getenv('ORACLE_PASSWORD'),
            host=os.getenv('ORACLE_HOST'),
            port=get_int_env('ORACLE_PORT', 1521),
            service=os.getenv('ORACLE_SERVICE')
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'user': self.user,
            'password': self.password,
            'host': self.host,
            'port': self.port,
            'service': self.service
        }


@dataclass
class ModelConfig:
    """Model training configuration with environment variable defaults."""
    
    model_name: str = 'google/bigbird-roberta-base'
    max_length: int = 512
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    save_dir: str = './models/saved'
    
    @classmethod
    def from_env(cls) -> 'ModelConfig':
        """
        Create configuration from environment variables.
        
        Returns:
            ModelConfig instance with values from environment
        """
        return cls(
            model_name=os.getenv('MODEL_NAME', cls.model_name),
            max_length=get_int_env('MAX_LENGTH', cls.max_length),
            batch_size=get_int_env('BATCH_SIZE', cls.batch_size),
            learning_rate=get_float_env('LEARNING_RATE', cls.learning_rate),
            num_epochs=get_int_env('NUM_EPOCHS', cls.num_epochs),
            save_dir=os.getenv('SAVE_DIR', cls.save_dir)
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'save_dir': self.save_dir
        }


@dataclass
class FlaskConfig:
    """Flask application configuration."""
    
    secret_key: str = 'dev-secret-key-change-in-production'
    debug: bool = False
    testing: bool = False
    host: str = '0.0.0.0'
    port: int = 5000
    
    @classmethod
    def from_env(cls) -> 'FlaskConfig':
        """
        Create configuration from environment variables.
        
        Returns:
            FlaskConfig instance with values from environment
        """
        return cls(
            secret_key=os.getenv('FLASK_SECRET_KEY', cls.secret_key),
            debug=os.getenv('FLASK_ENV') == 'development',
            testing=os.getenv('TESTING', 'False').lower() == 'true',
            host=os.getenv('FLASK_HOST', cls.host),
            port=get_int_env('FLASK_PORT', cls.port)
        )


@dataclass
class AzureConfig:
    """Azure Blob Storage configuration."""
    
    connection_string: Optional[str] = None
    container_name: str = 'ifs-annotations-backup'
    backup_enabled: bool = False
    backup_threshold: int = 10
    
    @classmethod
    def from_env(cls) -> 'AzureConfig':
        """
        Create configuration from environment variables.
        
        Returns:
            AzureConfig instance with values from environment
        """
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        return cls(
            connection_string=connection_string,
            container_name=os.getenv('AZURE_CONTAINER_NAME', cls.container_name),
            backup_enabled=connection_string is not None and os.getenv('BACKUP_ENABLED', 'true').lower() == 'true',
            backup_threshold=get_int_env('AUTO_BACKUP_THRESHOLD', cls.backup_threshold)
        )


class Config:
    """
    Application configuration container.
    
    This class follows the configuration management pattern for Python applications,
    providing a centralized, type-safe configuration management system.
    """
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self.database = DatabaseConfig.from_env()
        self.model = ModelConfig.from_env()
        self.flask = FlaskConfig.from_env()
        self.azure = AzureConfig.from_env()
    
    @classmethod
    def from_env(cls) -> 'Config':
        """
        Create configuration from environment variables.
        
        Returns:
            Config instance with all sub-configurations initialized
        """
        return cls()
    
    def __repr__(self) -> str:
        """String representation of configuration (without sensitive data)."""
        return (
            f"Config("
            f"database=<DatabaseConfig>, "
            f"model={self.model}, "
            f"flask=<FlaskConfig debug={self.flask.debug}>, "
            f"azure=<AzureConfig backup_enabled={self.azure.backup_enabled}>"
            f")"
        )
