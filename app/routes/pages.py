"""Page routes for HTML templates."""
from flask import render_template


def register_page_routes(app):
    """Register page routes."""
    
    @app.route('/')
    def index():
        """Home page."""
        return render_template('index.html')
    
    @app.route('/database')
    def database_page():
        """Database connection and data loading page."""
        return render_template('database.html')
    
    @app.route('/train')
    def train_page():
        """Model training page."""
        return render_template('train.html')
    
    @app.route('/predict')
    def predict_page():
        """Prediction page."""
        return render_template('predict.html')
    
    @app.route('/annotate')
    def annotate_page():
        """Annotation interface for active learning."""
        return render_template('annotate.html')
