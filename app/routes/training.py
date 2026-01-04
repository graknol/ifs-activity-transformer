"""Training-related API routes."""
from flask import request
import pandas as pd
from app.utils import ResponseBuilder, DataManager, FileValidator


def get_service(service_type: str):
    """Helper to get service from container."""
    from flask import current_app
    return current_app.container.resolve(service_type)


def register_training_routes(app):
    """Register training routes."""
    
    @app.route('/api/train/start', methods=['POST'])
    def start_training():
        """Start model training."""
        try:
            # Check if data exists
            exists, error_msg = FileValidator.check_file_exists(
                DataManager.TRAINING_DATA_PATH,
                'training data'
            )
            if not exists:
                return ResponseBuilder.error(error_msg, 400)
            
            # Get training parameters
            data = request.get_json()
            text_column = data.get('text_column', 'activity_description')
            label_column = data.get('label_column', 'new_activity_path')
            test_size = data.get('test_size', 0.2)
            
            # Update status
            app.training_status.update({
                'is_training': True,
                'progress': 10,
                'message': 'Loading data...'
            })
            
            # Load data
            df = DataManager.load_training_data()
            if df is None:
                return ResponseBuilder.error('Failed to load training data', 500)
            
            # Get classifier from container
            classifier = get_service('classifier')
            
            # Prepare data
            app.training_status.update({'progress': 20, 'message': 'Preparing data...'})
            train_dataset, val_dataset, num_labels = classifier.prepare_data(
                df, text_column, label_column, test_size
            )
            
            # Initialize model
            app.training_status.update({'progress': 40, 'message': 'Initializing model...'})
            classifier.initialize_model(num_labels)
            
            # Train model
            app.training_status.update({'progress': 50, 'message': 'Training model...'})
            metrics = classifier.train(train_dataset, val_dataset)
            
            # Complete
            app.training_status.update({
                'progress': 100,
                'message': 'Training complete!',
                'metrics': metrics,
                'is_training': False
            })
            
            return ResponseBuilder.success('Training completed successfully', {'metrics': metrics})
            
        except Exception as e:
            app.training_status.update({
                'is_training': False,
                'message': f'Error: {str(e)}'
            })
            return ResponseBuilder.from_exception(e, "Training")
    
    @app.route('/api/train/status', methods=['GET'])
    def get_training_status():
        """Get current training status."""
        from flask import jsonify
        return jsonify(app.training_status)
