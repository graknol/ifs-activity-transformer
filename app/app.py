"""
Flask web application factory for IFS Activity Transformer.

Following application factory pattern with dependency injection.
Refactored for DRY principles and improved maintainability.
"""
import logging
from flask import Flask
from app.config import Config
from app.database import OracleDBConnection
from app.model import ActivityClassifier
from app.backup_service import AzureBlobBackupService, AutoBackupManager
from app.model_manager import get_model_manager
from services.container import ServiceContainer
from app.routes import register_all_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config: Config = None) -> Flask:
    """
    Application factory function.
    
    Creates and configures the Flask application following the factory pattern,
    which allows for better testability and multiple application instances.
    
    Args:
        config: Application configuration. If None, loads from environment.
        
    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)
    
    # Load configuration
    app_config = config or Config.from_env()
    _configure_app(app, app_config)
    
    # Initialize model manager and preload models
    logger.info("Initializing model manager...")
    model_manager = get_model_manager()
    model_manager.print_system_info()
    
    # Preload configured model
    logger.info(f"Preloading model: {app_config.model.model_name}")
    model_manager.preload_models([app_config.model.model_name])
    
    # Create and configure service container
    container = _create_service_container(app_config, model_manager)
    app.container = container
    
    # Initialize application state
    app.training_status = _create_training_status()
    
    # Register all routes
    register_all_routes(app)
    
    logger.info("Application initialization complete!")
    
    return app


def _configure_app(app: Flask, config: Config) -> None:
    """
    Configure Flask application settings.
    
    Args:
        app: Flask application instance
        config: Application configuration
    """
    app.config['SECRET_KEY'] = config.flask.secret_key
    app.config['DEBUG'] = config.flask.debug
    app.config['TESTING'] = config.flask.testing


def _create_service_container(config: Config, model_manager) -> ServiceContainer:
    """
    Create and configure the service container with all dependencies.
    
    Args:
        config: Application configuration
        model_manager: ModelManager instance for GPU/model management
        
    Returns:
        Configured ServiceContainer instance
    """
    container = ServiceContainer()
    
    # Register configuration
    container.register_singleton('config', config)
    
    # Register model manager
    container.register_singleton('model_manager', model_manager)
    
    # Register database connection
    container.register_singleton('db_connection', OracleDBConnection(config.database))
    
    # Register model (transient for thread safety)
    container.register_transient('classifier', lambda: ActivityClassifier(config.model))
    
    # Register backup services
    backup_service = AzureBlobBackupService(
        connection_string=config.azure.connection_string,
        container_name=config.azure.container_name
    )
    container.register_singleton('backup_service', backup_service)
    container.register_singleton('auto_backup', AutoBackupManager(backup_service))
    
    # Initialize and register backup scheduler
    from app.backup_scheduler import init_backup_scheduler
    backup_scheduler = init_backup_scheduler(backup_service)
    container.register_singleton('backup_scheduler', backup_scheduler)
    logger.info("Backup scheduler initialized")
    
    return container


def _create_training_status() -> dict:
    """
    Create initial training status dictionary.
    
    Returns:
        Dictionary with training status fields
    """
    return {
        'is_training': False,
        'progress': 0,
        'message': 'Not started',
        'metrics': {}
    }


if __name__ == '__main__':
    print("Starting IFS Activity Transformer...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("\nWARNING: Running in debug mode. For production, use Gunicorn or another WSGI server.")
    
    app = create_app()
    config = app.container.resolve('config')
    app.run(
        debug=config.flask.debug,
        host=config.flask.host,
        port=config.flask.port
    )
