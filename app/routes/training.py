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
    
    @app.route('/api/training/label-distribution', methods=['GET'])
    def get_label_distribution():
        """Get distribution of labels in the training data."""
        import os
        try:
            # Load annotations
            annotations_path = 'data/annotations.csv'
            annotated_path = 'data/bootstrap_annotated.csv'
            
            df = None
            if os.path.exists(annotations_path):
                df = pd.read_csv(annotations_path)
            elif os.path.exists(annotated_path):
                df = pd.read_csv(annotated_path)
            
            if df is None or df.empty:
                return ResponseBuilder.success('No data', {
                    'distributions': {
                        'phase': {},
                        'discipline': {},
                        'work_type': {},
                        'location': {}
                    }
                })
            
            # Calculate distributions for each label type
            distributions = {
                'phase': df['PHASE_LABEL'].value_counts().to_dict() if 'PHASE_LABEL' in df.columns else {},
                'discipline': df['DISCIPLINE_LABEL'].value_counts().to_dict() if 'DISCIPLINE_LABEL' in df.columns else {},
                'work_type': df['WORK_TYPE_LABEL'].value_counts().to_dict() if 'WORK_TYPE_LABEL' in df.columns else {},
                'location': df['LOCATION_LABEL'].value_counts().to_dict() if 'LOCATION_LABEL' in df.columns else {}
            }
            
            # Clean NaN keys
            for label_type in distributions:
                distributions[label_type] = {
                    k: v for k, v in distributions[label_type].items() 
                    if pd.notna(k) and str(k).strip()
                }
            
            return ResponseBuilder.success('Label distributions', {
                'success': True,
                'distributions': distributions
            })
            
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Label distribution")
