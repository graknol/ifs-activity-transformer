"""Prediction API routes."""
import os
from flask import request
from app.utils import ResponseBuilder


def get_service(service_type: str):
    """Helper to get service from container."""
    from flask import current_app
    return current_app.container.resolve(service_type)


def register_prediction_routes(app):
    """Register prediction routes."""
    
    @app.route('/api/predict', methods=['POST'])
    def predict():
        """Make predictions on new activities."""
        try:
            data = request.get_json()
            texts = data.get('texts', [])
            
            if not texts:
                return ResponseBuilder.error('No texts provided', 400)
            
            # Get or load classifier
            classifier = get_service('classifier')
            config = get_service('config')
            
            # Check if model exists
            model_dir = config.model.save_dir
            if classifier.model is None:
                if not os.path.exists(model_dir):
                    return ResponseBuilder.error(
                        'No trained model found. Please train a model first.',
                        404
                    )
                classifier.load_model(model_dir)
            
            # Make predictions
            predictions = classifier.predict(texts)
            
            # Format results
            results = [
                {
                    'text': text,
                    'predicted_activity': label,
                    'confidence': float(confidence)
                }
                for text, (label, confidence) in zip(texts, predictions)
            ]
            
            return ResponseBuilder.success('Predictions completed', {'predictions': results})
            
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Prediction")
