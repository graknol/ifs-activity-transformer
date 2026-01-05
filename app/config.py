"""
Configuration classes for the application.
Following the configuration management pattern documented in .github/instructions.md
"""
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from dotenv import load_dotenv

load_dotenv()

# Default path to discipline codes JSON
DEFAULT_DISCIPLINE_CONFIG_PATH = Path(__file__).parent / "discipline_codes.json"


def get_int_env(key: str, default: int) -> int:
    """
    Get integer environment variable with default.
    
    Args:
        key: Environment variable name
        default: Default value if not set
        
    Returns:
        Integer value from environment or default
    """
    value = os.getenv(key, str(default))
    # Handle scientific notation by converting through float first
    try:
        return int(value)
    except ValueError:
        return int(float(value))


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
        backup_enabled = cls._get_backup_enabled(connection_string)
            
        return cls(
            connection_string=connection_string,
            container_name=os.getenv('AZURE_CONTAINER_NAME', cls.container_name),
            backup_enabled=backup_enabled,
            backup_threshold=get_int_env('AUTO_BACKUP_THRESHOLD', cls.backup_threshold)
        )
    
    @staticmethod
    def _get_backup_enabled(connection_string: Optional[str]) -> bool:
        """
        Determine if backup should be enabled.
        
        Args:
            connection_string: Azure connection string (None if not configured)
            
        Returns:
            True if backups should be enabled, False otherwise
        """
        backup_enabled_env = os.getenv('BACKUP_ENABLED')
        if backup_enabled_env is not None:
            # Explicit user setting takes precedence
            return backup_enabled_env.lower() == 'true'
        # Default: enable only if connection string is present
        return connection_string is not None


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
        self.discipline = DisciplineConfig.load()
    
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
            f"azure=<AzureConfig backup_enabled={self.azure.backup_enabled}>, "
            f"discipline=<DisciplineConfig loaded={self.discipline.is_loaded}>"
            f")"
        )


class DisciplineConfig:
    """
    Configuration for discipline codes, phases, and label mappings.
    
    Loads from discipline_codes.json and provides:
    - Discipline code lookups
    - Phase mappings
    - Sub-project level 1 structure
    - Code-to-labels mappings for ML training
    """
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize with loaded configuration data.
        
        Args:
            data: Parsed JSON configuration dictionary
        """
        self._data = data
        self.is_loaded = bool(data)
        
    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'DisciplineConfig':
        """
        Load discipline configuration from JSON file.
        
        Args:
            path: Path to JSON file. Uses default if None.
            
        Returns:
            DisciplineConfig instance
        """
        config_path = path or DEFAULT_DISCIPLINE_CONFIG_PATH
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load discipline config from {config_path}: {e}")
            return cls({})
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get configuration metadata."""
        return self._data.get('metadata', {})
    
    @property
    def discipline_codes(self) -> Dict[str, Dict[str, str]]:
        """
        Get all discipline codes organized by phase prefix.
        
        Returns:
            Dict mapping phase prefix to {code: description}
        """
        return self._data.get('discipline_codes', {})
    
    @property
    def code_to_labels(self) -> Dict[str, Dict[str, Any]]:
        """
        Get code-to-labels mapping for ML training.
        
        Returns:
            Dict mapping discipline code to label dictionary
        """
        return self._data.get('code_to_labels_mapping', {}).get('mappings', {})
    
    @property
    def phase_labels(self) -> List[str]:
        """Get list of phase label names."""
        return self._data.get('classification_labels', {}).get('phase_labels', [])
    
    @property
    def discipline_labels(self) -> List[str]:
        """Get list of discipline label names."""
        return self._data.get('classification_labels', {}).get('discipline_labels', [])
    
    @property
    def work_type_labels(self) -> List[str]:
        """Get list of work type label names."""
        return self._data.get('classification_labels', {}).get('work_type_labels', [])
    
    @property
    def location_labels(self) -> List[str]:
        """Get list of location label names."""
        return self._data.get('classification_labels', {}).get('location_labels', [])
    
    @property
    def all_labels(self) -> List[str]:
        """Get combined list of all classification labels."""
        return (
            self.phase_labels + 
            self.discipline_labels + 
            self.work_type_labels + 
            self.location_labels
        )
    
    @property
    def project_main_levels(self) -> Dict[str, Any]:
        """Get project main level execution order."""
        return self._data.get('project_main_levels', {})
    
    @property
    def target_sub_project_structure(self) -> Dict[str, Any]:
        """Get target sub-project level 1 structure for new system."""
        return self._data.get('target_sub_project_structure', {})
    
    @property
    def project_categories(self) -> Dict[str, Any]:
        """Get project category definitions."""
        return self._data.get('project_categories', {})
    
    @property
    def technical_disciplines(self) -> Dict[str, Any]:
        """Get technical discipline cross-reference."""
        return self._data.get('technical_disciplines', {})
    
    def get_code_description(self, code: str) -> Optional[str]:
        """
        Get description for a discipline code.
        
        Args:
            code: Two-letter discipline code (e.g., 'KE', 'QL')
            
        Returns:
            Description string or None if not found
        """
        for phase_data in self.discipline_codes.values():
            codes = phase_data.get('codes', {})
            if code in codes:
                return codes[code]
        return None
    
    def get_phase_for_code(self, code: str) -> Optional[str]:
        """
        Get phase prefix letter for a discipline code.
        
        Args:
            code: Two-letter discipline code
            
        Returns:
            Single-letter phase prefix or None
        """
        if len(code) >= 1:
            prefix = code[0].upper()
            if prefix in self.discipline_codes:
                return prefix
        return None
    
    def get_labels_for_code(self, code: str) -> Dict[str, Any]:
        """
        Get ML labels for a discipline code.
        
        Args:
            code: Two-letter discipline code
            
        Returns:
            Dict with phase, discipline, work_type, and optionally location
        """
        return self.code_to_labels.get(code, {})
    
    def get_phase_from_sub_project_id(self, sub_project_id: str) -> Optional[str]:
        """
        Derive phase from sub-project ID prefix (NEW system structure).
        
        Args:
            sub_project_id: Sub-project ID (e.g., '3000', '7500')
            
        Returns:
            Phase label or None
        """
        try:
            # Extract first digit to determine thousand range
            if sub_project_id and sub_project_id[0].isdigit():
                prefix = int(sub_project_id[0])
                phase_mapping = {
                    1: 'PRELIMINARY',
                    2: 'MANAGEMENT',
                    3: 'ENGINEERING',
                    4: 'PROCUREMENT',
                    5: 'FABRICATION',
                    6: 'CONSTRUCTION_ONSHORE',
                    7: 'INSTALLATION_OFFSHORE',
                    8: 'COMMISSIONING',
                    9: 'QUALITY'  # or CONTINGENCY
                }
                return phase_mapping.get(prefix)
        except (ValueError, IndexError):
            pass
        return None
    
    def get_contract_type_from_sub_project_id(self, sub_project_id: str) -> Optional[str]:
        """
        Derive contract type (RB/LS) from sub-project ID (NEW system).
        
        Convention: xxx0 = RB (Reimbursable), xxx5 = LS (Lump Sum)
        
        Args:
            sub_project_id: Sub-project ID string
            
        Returns:
            'RB', 'LS', or None
        """
        try:
            if sub_project_id and len(sub_project_id) == 4:
                # Check if it's in the 1000-8999 range (cost tracking)
                num = int(sub_project_id)
                if 1000 <= num < 9000:
                    # x000-x499 = RB, x500-x999 = LS
                    hundreds = (num % 1000) // 100
                    return 'LS' if hundreds >= 5 else 'RB'
        except ValueError:
            pass
        return None
    
    def get_training_sql_query(self, sample: bool = False) -> Optional[str]:
        """
        Get recommended SQL query for training data.
        
        Args:
            sample: If True, returns query limited to 10000 rows
            
        Returns:
            SQL query string or None
        """
        sql_queries = self._data.get('sql_queries', {})
        if sample:
            query_config = sql_queries.get('training_data_sample', {})
        else:
            query_config = sql_queries.get('training_data_full', {})
        return query_config.get('query')
    
    def get_sql_query(self, query_name: str) -> Optional[str]:
        """
        Get a named SQL query from configuration.
        
        Args:
            query_name: Name of the query (e.g., 'activity_count', 'discipline_distribution')
            
        Returns:
            SQL query string or None
        """
        sql_queries = self._data.get('sql_queries', {})
        query_config = sql_queries.get(query_name, {})
        return query_config.get('query')
    
    def get_connection_info(self) -> Dict[str, str]:
        """
        Get database connection information.
        
        Returns:
            Dict with connect_to, db_link, and notes
        """
        return self._data.get('sql_queries', {}).get('connection_info', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (returns raw data)."""
        return self._data.copy()
