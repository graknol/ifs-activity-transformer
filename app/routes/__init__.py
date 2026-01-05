"""
API routes for the IFS Activity Transformer application.

This package organizes routes by functional area to improve maintainability.
"""
from .pages import register_page_routes
from .database import register_database_routes
from .training import register_training_routes
from .prediction import register_prediction_routes
from .annotation import register_annotation_routes
from .backup import register_backup_routes
from .bootstrap import bootstrap_bp


def register_all_routes(app):
    """
    Register all application routes.
    
    Args:
        app: Flask application instance
    """
    register_page_routes(app)
    register_database_routes(app)
    register_training_routes(app)
    register_prediction_routes(app)
    register_annotation_routes(app)
    register_backup_routes(app)
    
    # Register bootstrap and incremental training routes
    app.register_blueprint(bootstrap_bp)
