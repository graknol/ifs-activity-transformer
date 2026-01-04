"""
Flask web application factory for IFS Activity Transformer.
Following application factory pattern with dependency injection.
"""
import os
import json
from flask import Flask, render_template, request, jsonify, g
from app.config import Config
from app.database import OracleDBConnection
from app.model import ActivityClassifier
from app.backup_service import AzureBlobBackupService, AutoBackupManager
from services.container import ServiceContainer
import pandas as pd


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
    app.config['SECRET_KEY'] = app_config.flask.secret_key
    app.config['DEBUG'] = app_config.flask.debug
    app.config['TESTING'] = app_config.flask.testing
    
    # Create service container
    container = ServiceContainer()
    
    # Register services
    container.register_singleton('config', app_config)
    container.register_singleton('db_connection', OracleDBConnection(app_config.database))
    container.register_transient('classifier', lambda: ActivityClassifier(app_config.model))
    
    # Register backup service
    backup_service = AzureBlobBackupService()
    container.register_singleton('backup_service', backup_service)
    container.register_singleton('auto_backup', AutoBackupManager(backup_service))
    
    # Store container in app context
    app.container = container
    
    # Training status (using app context instead of global)
    app.training_status = {
        'is_training': False,
        'progress': 0,
        'message': 'Not started',
        'metrics': {}
    }
    
    # Register routes
    register_routes(app)
    
    return app


def get_service(service_type: str):
    """
    Helper function to resolve services from the container.
    
    Args:
        service_type: Service identifier
        
    Returns:
        Service instance from the container
    """
    from flask import current_app
    return current_app.container.resolve(service_type)


def register_routes(app: Flask) -> None:
    """
    Register all application routes.
    
    Args:
        app: Flask application instance
    """
    
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
    
    @app.route('/api/database/connect', methods=['POST'])
    def connect_database():
        """Connect to Oracle database."""
        try:
            db_connection = get_service('db_connection')
            success = db_connection.connect()
            
            if success:
                stats = db_connection.get_activity_statistics()
                return jsonify({
                    'success': True,
                    'message': 'Connected to database successfully',
                    'statistics': stats
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Failed to connect to database'
                }), 500
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            }), 500
    
    @app.route('/api/database/fetch-data', methods=['POST'])
    def fetch_data():
        """Fetch training data from database."""
        try:
            db_connection = get_service('db_connection')
            
            if not db_connection.connection:
                return jsonify({
                    'success': False,
                    'message': 'Not connected to database'
                }), 400
            
            data = request.get_json()
            custom_query = data.get('query', None)
            
            df = db_connection.fetch_training_data(custom_query)
            
            if df.empty:
                return jsonify({
                    'success': False,
                    'message': 'No data retrieved from database'
                }), 404
            
            # Save data to file
            data_path = 'data/training_data.csv'
            os.makedirs('data', exist_ok=True)
            df.to_csv(data_path, index=False)
            
            return jsonify({
                'success': True,
                'message': f'Retrieved {len(df)} records',
                'preview': df.head(10).to_dict('records'),
                'columns': df.columns.tolist(),
                'row_count': len(df)
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            }), 500
    
    @app.route('/api/train/start', methods=['POST'])
    def start_training():
        """Start model training."""
        try:
            # Get training parameters
            data = request.get_json()
            text_column = data.get('text_column', 'activity_description')
            label_column = data.get('label_column', 'new_activity_path')
            test_size = data.get('test_size', 0.2)
            
            # Check if data exists
            data_path = 'data/training_data.csv'
            if not os.path.exists(data_path):
                return jsonify({
                    'success': False,
                    'message': 'No training data found. Please fetch data first.'
                }), 400
            
            # Update status
            app.training_status['is_training'] = True
            app.training_status['progress'] = 10
            app.training_status['message'] = 'Loading data...'
            
            # Load data
            df = pd.read_csv(data_path)
            
            # Get classifier from container
            classifier = get_service('classifier')
            
            app.training_status['progress'] = 20
            app.training_status['message'] = 'Preparing data...'
            
            # Prepare data
            train_dataset, val_dataset, num_labels = classifier.prepare_data(
                df, text_column, label_column, test_size
            )
            
            app.training_status['progress'] = 40
            app.training_status['message'] = 'Initializing model...'
            
            # Initialize model
            classifier.initialize_model(num_labels)
            
            app.training_status['progress'] = 50
            app.training_status['message'] = 'Training model...'
            
            # Train model
            metrics = classifier.train(train_dataset, val_dataset)
            
            app.training_status['progress'] = 100
            app.training_status['message'] = 'Training complete!'
            app.training_status['metrics'] = metrics
            app.training_status['is_training'] = False
            
            return jsonify({
                'success': True,
                'message': 'Training completed successfully',
                'metrics': metrics
            })
        except Exception as e:
            app.training_status['is_training'] = False
            app.training_status['message'] = f'Error: {str(e)}'
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            }), 500
    
    @app.route('/api/train/status', methods=['GET'])
    def get_training_status():
        """Get current training status."""
        return jsonify(app.training_status)
    
    @app.route('/api/predict', methods=['POST'])
    def predict():
        """Make predictions on new activities."""
        try:
            data = request.get_json()
            texts = data.get('texts', [])
            
            if not texts:
                return jsonify({
                    'success': False,
                    'message': 'No texts provided'
                }), 400
            
            # Get or load classifier
            classifier = get_service('classifier')
            
            if classifier.model is None:
                config = get_service('config')
                model_dir = config.model.save_dir
                if not os.path.exists(model_dir):
                    return jsonify({
                        'success': False,
                        'message': 'No trained model found. Please train a model first.'
                    }), 404
                
                classifier.load_model(model_dir)
            
            # Make predictions
            predictions = classifier.predict(texts)
            
            results = []
            for i, (text, (label, confidence)) in enumerate(zip(texts, predictions)):
                results.append({
                    'text': text,
                    'predicted_activity': label,
                    'confidence': float(confidence)
                })
            
            return jsonify({
                'success': True,
                'predictions': results
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            }), 500
    
    @app.route('/api/upload-csv', methods=['POST'])
    def upload_csv():
        """Upload CSV file with training data."""
        try:
            if 'file' not in request.files:
                return jsonify({
                    'success': False,
                    'message': 'No file uploaded'
                }), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'message': 'No file selected'
                }), 400
            
            if not file.filename.endswith('.csv'):
                return jsonify({
                    'success': False,
                    'message': 'File must be a CSV'
                }), 400
            
            # Save file
            data_path = 'data/training_data.csv'
            os.makedirs('data', exist_ok=True)
            file.save(data_path)
            
            # Read and validate
            df = pd.read_csv(data_path)
            
            return jsonify({
                'success': True,
                'message': f'Uploaded {len(df)} records',
                'columns': df.columns.tolist(),
                'preview': df.head(10).to_dict('records'),
                'row_count': len(df)
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            }), 500
    
    # Annotation API routes for active learning
    @app.route('/api/annotate/get-batch', methods=['POST'])
    def get_annotation_batch():
        """
        Get a batch of activities for annotation with uncertainty sampling.
        """
        try:
            from app.active_learning import ActiveLearningService, MultiLabelActiveLearning
            import numpy as np
            
            data = request.get_json()
            batch_size = data.get('batch_size', 20)
            strategy = data.get('sampling_strategy', 'least_confidence')
            
            # Load data
            data_path = 'data/training_data.csv'
            if not os.path.exists(data_path):
                return jsonify({
                    'success': False,
                    'message': 'No training data found'
                }), 404
            
            df = pd.read_csv(data_path)
            
            # Check if we have a trained model for predictions
            config = get_service('config')
            model_dir = config.model.save_dir
            has_model = os.path.exists(model_dir)
            
            if has_model:
                # Get predictions for uncertainty sampling
                classifier = get_service('classifier')
                try:
                    classifier.load_model(model_dir)
                    # Simple prediction - in production, handle multi-label properly
                    texts = df.iloc[:, 1].tolist()  # Assuming second column is text
                    predictions = []
                    # TODO: Implement actual prediction logic
                    predictions = np.random.rand(len(df), 8)  # Mock predictions
                except:
                    # If model loading fails, use random sampling
                    predictions = np.random.rand(len(df), 8)
            else:
                # No model yet, use random sampling
                predictions = np.random.rand(len(df), 8)
            
            # Use active learning to select samples
            al_service = MultiLabelActiveLearning()
            sample_ids = df.iloc[:, 0].astype(str).tolist()  # Assuming first column is ID
            
            selected = al_service.select_multilabel_samples(
                predictions,
                sample_ids,
                n_samples=min(batch_size, len(df)),
                sanity_check_ratio=0.1
            )
            
            # Build activity objects for UI
            activities = []
            for item in selected:
                idx = item['index']
                row = df.iloc[idx]
                
                activity = {
                    'id': str(row.iloc[0]),
                    'name': str(row.iloc[1]) if len(row) > 1 else 'Unnamed',
                    'description': str(row.iloc[2]) if len(row) > 2 else '',
                    'uncertainty_score': item['uncertainty_score'],
                    'max_confidence': 1 - item['uncertainty_score'],
                    'predicted_categories': [],  # TODO: decode from predictions
                    'features': {}  # TODO: extract numerical features
                }
                activities.append(activity)
            
            # Load annotation statistics
            annotations_path = 'data/annotations.csv'
            if os.path.exists(annotations_path):
                annotations_df = pd.read_csv(annotations_path)
                al_service_stats = ActiveLearningService()
                stats = al_service_stats.get_annotation_statistics(annotations_df)
            else:
                stats = {
                    'total_samples': len(df),
                    'labeled_samples': 0,
                    'unlabeled_samples': len(df),
                    'completion_rate': 0
                }
            
            return jsonify({
                'success': True,
                'activities': activities,
                'statistics': stats
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            }), 500
    
    @app.route('/api/annotate/save', methods=['POST'])
    def save_annotation():
        """Save user annotation for an activity."""
        try:
            data = request.get_json()
            activity_id = data.get('activity_id')
            categories = data.get('categories', [])
            uncertainty_score = data.get('uncertainty_score', 0)
            
            # Load or create annotations file
            annotations_path = 'data/annotations.csv'
            
            if os.path.exists(annotations_path):
                annotations_df = pd.read_csv(annotations_path)
            else:
                annotations_df = pd.DataFrame(columns=[
                    'activity_id', 'categories', 'uncertainty_score', 'timestamp'
                ])
            
            # Add or update annotation
            from datetime import datetime
            new_annotation = {
                'activity_id': activity_id,
                'categories': ','.join(categories),
                'uncertainty_score': uncertainty_score,
                'timestamp': datetime.now().isoformat(),
                'label': 1 if categories else None  # For statistics
            }
            
            # Remove existing annotation if any
            annotations_df = annotations_df[annotations_df['activity_id'] != activity_id]
            
            # Add new annotation
            annotations_df = pd.concat([
                annotations_df,
                pd.DataFrame([new_annotation])
            ], ignore_index=True)
            
            # Save
            os.makedirs('data', exist_ok=True)
            annotations_df.to_csv(annotations_path, index=False)
            
            # Auto backup to Azure (if enabled)
            auto_backup = get_service('auto_backup')
            auto_backup.on_annotation_saved(annotations_df)
            
            # Get updated statistics
            from app.active_learning import ActiveLearningService
            al_service = ActiveLearningService()
            stats = al_service.get_annotation_statistics(annotations_df)
            
            return jsonify({
                'success': True,
                'message': 'Annotation saved',
                'statistics': stats
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            }), 500
    
    @app.route('/api/annotate/update-query', methods=['POST'])
    def update_annotation_query():
        """Update the SQL query for fetching annotation data."""
        try:
            data = request.get_json()
            query = data.get('query', '')
            
            if not query:
                return jsonify({
                    'success': False,
                    'message': 'No query provided'
                }), 400
            
            # Store query in a file for later use
            query_path = 'data/annotation_query.sql'
            os.makedirs('data', exist_ok=True)
            
            with open(query_path, 'w') as f:
                f.write(query)
            
            # TODO: Execute query and reload data
            
            return jsonify({
                'success': True,
                'message': 'Query updated successfully'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            }), 500
    
    # Backup management API routes
    @app.route('/api/backup/status', methods=['GET'])
    def get_backup_status():
        """Get backup service status and statistics."""
        try:
            backup_service = get_service('backup_service')
            stats = backup_service.get_backup_statistics()
            
            return jsonify({
                'success': True,
                'backup_enabled': backup_service.is_enabled(),
                'statistics': stats
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            }), 500
    
    @app.route('/api/backup/create', methods=['POST'])
    def create_manual_backup():
        """Create a manual backup of annotations."""
        try:
            data = request.get_json()
            backup_type = data.get('type', 'manual')
            milestone_name = data.get('milestone_name', '')
            description = data.get('description', '')
            
            # Load annotations
            annotations_path = 'data/annotations.csv'
            if not os.path.exists(annotations_path):
                return jsonify({
                    'success': False,
                    'message': 'No annotations to backup'
                }), 404
            
            annotations_df = pd.read_csv(annotations_path)
            backup_service = get_service('backup_service')
            
            if backup_type == 'milestone' and milestone_name:
                blob_name = backup_service.create_milestone_backup(
                    annotations_df,
                    milestone_name,
                    description
                )
            else:
                blob_name = backup_service.create_snapshot(
                    annotations_df,
                    metadata={'created_by': 'user'},
                    snapshot_type='manual'
                )
            
            if blob_name:
                return jsonify({
                    'success': True,
                    'message': 'Backup created successfully',
                    'blob_name': blob_name
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Backup failed or not enabled'
                }), 500
                
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            }), 500
    
    @app.route('/api/backup/list', methods=['GET'])
    def list_backups():
        """List available backups."""
        try:
            backup_type = request.args.get('type', None)
            limit = int(request.args.get('limit', 50))
            
            backup_service = get_service('backup_service')
            backups = backup_service.list_backups(backup_type, limit)
            
            return jsonify({
                'success': True,
                'backups': backups
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            }), 500
    
    @app.route('/api/backup/restore', methods=['POST'])
    def restore_backup():
        """Restore annotations from a backup."""
        try:
            data = request.get_json()
            blob_name = data.get('blob_name')
            
            if not blob_name:
                return jsonify({
                    'success': False,
                    'message': 'No blob name provided'
                }), 400
            
            backup_service = get_service('backup_service')
            restore_path = 'data/annotations_restored.csv'
            
            success = backup_service.restore_snapshot(blob_name, restore_path)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'Backup restored to {restore_path}'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Restore failed'
                }), 500
                
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}'
            }), 500


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
