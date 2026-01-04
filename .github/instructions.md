# Python Development Best Practices & Patterns

This document outlines standardized patterns and best practices for Python development in this project, designed to maximize code quality, maintainability, and AI efficiency.

## Table of Contents
1. [Code Organization & DRY Principles](#code-organization--dry-principles)
2. [Dependency Injection](#dependency-injection)
3. [Project Structure](#project-structure)
4. [Design Patterns](#design-patterns)
5. [Configuration Management](#configuration-management)
6. [Testing Patterns](#testing-patterns)
7. [Code Quality Standards](#code-quality-standards)

---

## Code Organization & DRY Principles

### Refactoring Achievement
The codebase has been refactored following DRY (Don't Repeat Yourself) principles with significant improvements:

- **83% reduction** in main application file (678 → 116 lines)
- **Modular routes**: Organized by feature area for better maintainability
- **Centralized utilities**: Common operations extracted to reusable components
- **Consistent patterns**: Unified approach to error handling and responses

### Key Utilities

#### DataManager
Centralized file operations to eliminate duplication:

```python
from app.utils import DataManager

# Load training data (handles missing files gracefully)
df = DataManager.load_training_data()

# Save annotations
DataManager.save_annotations(annotations_df)

# Standard paths are centralized
DataManager.TRAINING_DATA_PATH  # 'data/training_data.csv'
DataManager.ANNOTATIONS_PATH    # 'data/annotations.csv'
```

#### ResponseBuilder
Consistent API responses throughout the application:

```python
from app.utils import ResponseBuilder

# Success response
return ResponseBuilder.success('Operation completed', {'data': result})

# Error response
return ResponseBuilder.error('Invalid input', status=400)

# From exception
return ResponseBuilder.from_exception(e, "Operation context")
```

#### FileValidator
Common validation logic:

```python
from app.utils import FileValidator

# Validate CSV upload
is_valid, error_msg = FileValidator.validate_csv_upload(file)

# Check file exists
exists, error_msg = FileValidator.check_file_exists(path, "training data")
```

### Modular Route Organization

Routes are organized by feature area in separate modules:

```python
app/routes/
├── __init__.py        # Registers all routes
├── pages.py           # HTML page routes
├── database.py        # Database operations
├── training.py        # Model training
├── prediction.py      # Predictions
├── annotation.py      # Active learning annotation
└── backup.py          # Azure backup management
```

Each route module:
- Has a single responsibility
- Uses utilities for common operations
- Returns consistent responses
- Handles errors gracefully

### DRY Checklist

When adding new features, ensure:
- [ ] Common operations use utilities (DataManager, ResponseBuilder, etc.)
- [ ] File paths use DataManager constants
- [ ] API responses use ResponseBuilder
- [ ] Validation uses FileValidator
- [ ] Route logic is in appropriate route module
- [ ] No duplicate error handling patterns
- [ ] No duplicate data loading/saving code
- [ ] Configuration uses type-safe dataclasses

---

## Dependency Injection

### Philosophy
Python doesn't have a built-in equivalent to C#'s `IServiceCollection`, but we follow similar principles using:
- **Constructor Injection**: Dependencies passed through `__init__`
- **Factory Pattern**: For complex object creation
- **Service Locator Pattern**: For application-wide services
- **Context Managers**: For resource management

### Service Container Pattern

We use a custom service container that mimics C#'s dependency injection:

```python
# services/container.py
class ServiceContainer:
    """
    Service container for dependency injection.
    Similar to C#'s IServiceCollection.
    """
    def __init__(self):
        self._services = {}
        self._singletons = {}
    
    def register_singleton(self, service_type, instance):
        """Register a singleton service (created once)."""
        self._singletons[service_type] = instance
    
    def register_transient(self, service_type, factory):
        """Register a transient service (created each time)."""
        self._services[service_type] = factory
    
    def resolve(self, service_type):
        """Resolve a service instance."""
        if service_type in self._singletons:
            return self._singletons[service_type]
        if service_type in self._services:
            return self._services[service_type]()
        raise ValueError(f"Service {service_type} not registered")
```

### Usage Example

```python
# app/__init__.py
from services.container import ServiceContainer
from app.database import OracleDBConnection
from app.model import ActivityClassifier

def create_app(config=None):
    """Application factory pattern."""
    app = Flask(__name__)
    
    # Configure app
    app.config.from_object(config or Config())
    
    # Create service container
    container = ServiceContainer()
    
    # Register services
    container.register_singleton('db', OracleDBConnection())
    container.register_transient('classifier', lambda: ActivityClassifier())
    
    # Store container in app context
    app.container = container
    
    # Register blueprints
    from app.routes import api_blueprint
    app.register_blueprint(api_blueprint)
    
    return app
```

---

## Project Structure

### Standard Layout

```
project/
├── .github/
│   └── instructions.md          # This file
├── app/
│   ├── __init__.py              # Application factory
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── database.py          # Database routes
│   │   ├── training.py          # Training routes
│   │   └── prediction.py        # Prediction routes
│   ├── services/
│   │   ├── __init__.py
│   │   ├── database_service.py  # Database operations
│   │   ├── training_service.py  # Training operations
│   │   └── prediction_service.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py        # ML model wrapper
│   ├── config.py                # Configuration classes
│   └── static/ & templates/     # Frontend assets
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py              # Pytest fixtures
├── services/
│   ├── __init__.py
│   └── container.py             # DI container
├── requirements.txt
└── run.py
```

---

## Design Patterns

### 1. Repository Pattern
Separate data access from business logic:

```python
class ActivityRepository:
    """Repository for activity data access."""
    
    def __init__(self, db_connection):
        self.db = db_connection
    
    def get_all(self) -> List[Activity]:
        """Get all activities."""
        pass
    
    def get_by_id(self, id: int) -> Optional[Activity]:
        """Get activity by ID."""
        pass
    
    def add(self, activity: Activity) -> None:
        """Add new activity."""
        pass
```

### 2. Service Layer Pattern
Business logic separated from routes:

```python
class TrainingService:
    """Service for model training operations."""
    
    def __init__(self, classifier, repository, config):
        self.classifier = classifier
        self.repository = repository
        self.config = config
    
    def train_model(self, data_params: dict) -> TrainingResult:
        """Train model with given parameters."""
        # Business logic here
        pass
```

### 3. Factory Pattern
Complex object creation:

```python
class ClassifierFactory:
    """Factory for creating classifier instances."""
    
    @staticmethod
    def create(model_type: str, config: dict) -> BaseClassifier:
        if model_type == 'bigbird':
            return BigBirdClassifier(config)
        elif model_type == 'roberta':
            return RoBertaClassifier(config)
        raise ValueError(f"Unknown model type: {model_type}")
```

### 4. Strategy Pattern
Interchangeable algorithms:

```python
class DataLoadStrategy(ABC):
    """Abstract base for data loading strategies."""
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass

class OracleLoadStrategy(DataLoadStrategy):
    def load(self) -> pd.DataFrame:
        # Oracle-specific loading
        pass

class CSVLoadStrategy(DataLoadStrategy):
    def load(self) -> pd.DataFrame:
        # CSV loading
        pass
```

---

## Configuration Management

### Configuration Classes

```python
# app/config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    """Database configuration."""
    user: str
    password: str
    host: str
    port: int = 1521
    service: str = None
    
    @classmethod
    def from_env(cls):
        """Create from environment variables."""
        return cls(
            user=os.getenv('ORACLE_USER'),
            password=os.getenv('ORACLE_PASSWORD'),
            host=os.getenv('ORACLE_HOST'),
            port=int(os.getenv('ORACLE_PORT', '1521')),
            service=os.getenv('ORACLE_SERVICE')
        )

@dataclass
class ModelConfig:
    """Model training configuration."""
    model_name: str = 'google/bigbird-roberta-base'
    max_length: int = 512
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    save_dir: str = './models/saved'
    
    @classmethod
    def from_env(cls):
        """Create from environment variables."""
        return cls(
            model_name=os.getenv('MODEL_NAME', cls.model_name),
            max_length=int(os.getenv('MAX_LENGTH', cls.max_length)),
            batch_size=int(os.getenv('BATCH_SIZE', cls.batch_size)),
            learning_rate=float(os.getenv('LEARNING_RATE', cls.learning_rate)),
            num_epochs=int(os.getenv('NUM_EPOCHS', cls.num_epochs)),
            save_dir=os.getenv('SAVE_DIR', cls.save_dir)
        )

class Config:
    """Application configuration."""
    def __init__(self):
        self.database = DatabaseConfig.from_env()
        self.model = ModelConfig.from_env()
        self.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret')
        self.debug = os.getenv('FLASK_ENV') == 'development'
```

---

## Testing Patterns

### Fixtures and Dependency Injection

```python
# tests/conftest.py
import pytest
from app import create_app
from services.container import ServiceContainer

@pytest.fixture
def app():
    """Create application for testing."""
    app = create_app({'TESTING': True})
    yield app

@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()

@pytest.fixture
def mock_db():
    """Mock database connection."""
    class MockDB:
        def connect(self):
            return True
        def fetch_training_data(self, query=None):
            return pd.DataFrame({'col': [1, 2, 3]})
    return MockDB()

@pytest.fixture
def container(mock_db):
    """Service container with mocks."""
    container = ServiceContainer()
    container.register_singleton('db', mock_db)
    return container
```

### Unit Test Example

```python
# tests/unit/test_training_service.py
def test_train_model(container):
    """Test model training."""
    service = TrainingService(
        classifier=container.resolve('classifier'),
        repository=container.resolve('repository'),
        config=ModelConfig()
    )
    
    result = service.train_model({'test_size': 0.2})
    assert result.success
    assert result.metrics['accuracy'] > 0
```

---

## Code Quality Standards

### 1. Type Hints
Always use type hints for function parameters and return values:

```python
def fetch_data(
    query: Optional[str] = None,
    limit: int = 100
) -> pd.DataFrame:
    """Fetch data with type hints."""
    pass
```

### 2. Docstrings
Use Google-style docstrings:

```python
def process_data(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Process dataframe by applying transformations.
    
    Args:
        data: Input dataframe to process
        column: Column name to transform
    
    Returns:
        Processed dataframe with transformations applied
    
    Raises:
        ValueError: If column doesn't exist in dataframe
    """
    pass
```

### 3. Error Handling
Use custom exceptions and proper error handling:

```python
# app/exceptions.py
class AppException(Exception):
    """Base exception for application."""
    pass

class DatabaseConnectionError(AppException):
    """Database connection failed."""
    pass

class ModelTrainingError(AppException):
    """Model training failed."""
    pass

# Usage
try:
    db.connect()
except ConnectionError as e:
    raise DatabaseConnectionError(f"Failed to connect: {e}")
```

### 4. Logging
Use structured logging:

```python
import logging

logger = logging.getLogger(__name__)

class TrainingService:
    def train_model(self, params):
        logger.info("Starting model training", extra={
            'params': params,
            'timestamp': datetime.now()
        })
        try:
            # Training logic
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error("Training failed", exc_info=True, extra={
                'error': str(e),
                'params': params
            })
            raise
```

### 5. Context Managers
Use for resource management:

```python
class DatabaseContext:
    """Context manager for database connections."""
    
    def __init__(self, config):
        self.config = config
        self.connection = None
    
    def __enter__(self):
        self.connection = oracledb.connect(**self.config)
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self.connection.close()

# Usage
with DatabaseContext(config) as db:
    data = db.fetch_data()
```

---

## DRY Principles

### 1. Extract Common Logic

**Before:**
```python
# Repeated in multiple places
config = {
    'model_name': os.getenv('MODEL_NAME', 'default'),
    'batch_size': int(os.getenv('BATCH_SIZE', '8'))
}
```

**After:**
```python
# Single source of truth
@dataclass
class Config:
    @classmethod
    def from_env(cls):
        return cls(...)
```

### 2. Use Mixins for Shared Behavior

```python
class ConfigurableMixin:
    """Mixin for configurable classes."""
    
    def load_config(self, config_dict: dict) -> None:
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

class ActivityClassifier(ConfigurableMixin):
    def __init__(self):
        self.load_config(ModelConfig.from_env().__dict__)
```

### 3. Decorator for Common Operations

```python
def with_db_connection(func):
    """Decorator to handle database connections."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.db.connection:
            self.db.connect()
        try:
            return func(self, *args, **kwargs)
        finally:
            self.db.disconnect()
    return wrapper

class DataService:
    @with_db_connection
    def fetch_data(self):
        # Database is guaranteed to be connected
        pass
```

---

## AI Efficiency Tips

### 1. Clear Module Responsibilities
- One class = one responsibility
- Clear naming conventions
- Predictable file locations

### 2. Comprehensive Docstrings
- Include examples in docstrings
- Document parameters, returns, and exceptions
- Use type hints consistently

### 3. Test Coverage
- Unit tests for each service
- Integration tests for workflows
- Clear test names: `test_<what>_<when>_<expected>`

### 4. Configuration as Code
- All configuration in typed classes
- No magic strings
- Environment variables with defaults

### 5. Consistent Patterns
- Same pattern for all services
- Same error handling approach
- Same logging format

---

## Refactoring Checklist

When refactoring code, follow this checklist:

- [ ] Replace global variables with dependency injection
- [ ] Extract configuration into typed classes
- [ ] Create service layer for business logic
- [ ] Separate data access into repositories
- [ ] Add type hints to all functions
- [ ] Write/update unit tests
- [ ] Add logging with context
- [ ] Document with docstrings
- [ ] Use context managers for resources
- [ ] Apply DRY principles
- [ ] Create custom exceptions for error handling
- [ ] Add integration tests

---

## Additional Resources

### Python Libraries for Patterns
- **dependency-injector**: Full DI framework
- **injector**: Lightweight dependency injection
- **flask-injector**: Flask integration for DI
- **pydantic**: Data validation and settings management
- **python-dotenv**: Environment variable management

### Recommended Reading
- "Clean Code" by Robert C. Martin
- "Design Patterns" by Gang of Four
- "Architecture Patterns with Python" by Percival & Gregory
- PEP 8: Python Style Guide
- PEP 257: Docstring Conventions

---

*This document should be updated as new patterns emerge in the project.*
