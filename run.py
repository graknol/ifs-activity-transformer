#!/usr/bin/env python3
"""
Main entry point for IFS Activity Transformer web application.
"""
import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.app import app

if __name__ == '__main__':
    print("Starting IFS Activity Transformer...")
    print("Open your browser and navigate to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
