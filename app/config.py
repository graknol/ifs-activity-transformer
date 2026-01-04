"""
Configuration classes for the application.
Following the configuration management pattern documented in .github/instructions.md
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


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
            port=int(os.getenv('ORACLE_PORT', '1521')),
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
            max_length=int(os.getenv('MAX_LENGTH', str(cls.max_length))),
            batch_size=int(os.getenv('BATCH_SIZE', str(cls.batch_size))),
            learning_rate=float(os.getenv('LEARNING_RATE', str(cls.learning_rate))),
            num_epochs=int(os.getenv('NUM_EPOCHS', str(cls.num_epochs))),
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
            port=int(os.getenv('FLASK_PORT', str(cls.port)))
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
            f"flask=<FlaskConfig debug={self.flask.debug}>"
            f")"
        )
