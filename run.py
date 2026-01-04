#!/usr/bin/env python3
"""
Main entry point for IFS Activity Transformer web application.
Uses application factory pattern for better testability and configuration.
"""
import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.app import create_app

if __name__ == '__main__':
    print("Starting IFS Activity Transformer...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("\nWARNING: Running in debug mode. For production, use Gunicorn or another WSGI server.")
    
    # Create application using factory pattern
    app = create_app()
    
    # Get configuration from app container
    config = app.container.resolve('config')
    
    # Run application
    app.run(
        debug=config.flask.debug,
        host=config.flask.host,
        port=config.flask.port
    )
