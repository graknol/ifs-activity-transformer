"""Annotation API routes for active learning."""
import os
import pandas as pd
import numpy as np
from flask import request
from app.utils import ResponseBuilder, DataManager, DataFrameHelper, FileValidator
from app.active_learning import ActiveLearningService, MultiLabelActiveLearning


def get_service(service_type: str):
    """Helper to get service from container."""
    from flask import current_app
    return current_app.container.resolve(service_type)


def register_annotation_routes(app):
    """Register annotation routes."""
    
    @app.route('/api/annotate/get-batch', methods=['POST'])
    def get_annotation_batch():
        """Get a batch of activities for annotation with uncertainty sampling."""
        try:
            data = request.get_json()
            batch_size = data.get('batch_size', 20)
            strategy = data.get('sampling_strategy', 'least_confidence')
            
            # Load data
            df = DataManager.load_training_data()
            if df is None:
                return ResponseBuilder.error('No training data found', 404)
            
            # Check for trained model and get predictions
            config = get_service('config')
            model_dir = config.model.save_dir
            has_model = os.path.exists(model_dir)
            
            if has_model:
                try:
                    classifier = get_service('classifier')
                    classifier.load_model(model_dir)
                    # TODO: Implement actual prediction logic
                    predictions = np.random.rand(len(df), 8)  # Mock predictions
                except:
                    predictions = np.random.rand(len(df), 8)
            else:
                predictions = np.random.rand(len(df), 8)
            
            # Use active learning to select samples
            al_service = MultiLabelActiveLearning()
            sample_ids = df.iloc[:, 0].astype(str).tolist()
            
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
            annotations_df = DataManager.load_annotations()
            al_service_stats = ActiveLearningService()
            stats = al_service_stats.get_annotation_statistics(annotations_df)
            
            return ResponseBuilder.success(
                'Batch retrieved',
                {'activities': activities, 'statistics': stats}
            )
            
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Get annotation batch")
    
    @app.route('/api/annotate/save', methods=['POST'])
    def save_annotation():
        """Save user annotation for an activity."""
        try:
            data = request.get_json()
            activity_id = data.get('activity_id')
            categories = data.get('categories', [])
            uncertainty_score = data.get('uncertainty_score', 0)
            
            # Load annotations
            annotations_df = DataManager.load_annotations()
            
            # Add or update annotation
            annotations_df = DataFrameHelper.merge_annotation(
                annotations_df,
                activity_id,
                categories,
                uncertainty_score
            )
            
            # Save
            if not DataManager.save_annotations(annotations_df):
                return ResponseBuilder.error('Failed to save annotation', 500)
            
            # Auto backup to Azure (if enabled)
            auto_backup = get_service('auto_backup')
            auto_backup.on_annotation_saved(annotations_df)
            
            # Get updated statistics
            al_service = ActiveLearningService()
            stats = al_service.get_annotation_statistics(annotations_df)
            
            return ResponseBuilder.success('Annotation saved', {'statistics': stats})
            
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Save annotation")
    
    @app.route('/api/annotate/update-query', methods=['POST'])
    def update_annotation_query():
        """Update the SQL query for fetching annotation data."""
        try:
            data = request.get_json()
            query = data.get('query', '')
            
            if not query:
                return ResponseBuilder.error('No query provided', 400)
            
            # Store query
            DataManager.ensure_data_dir()
            with open(DataManager.QUERY_PATH, 'w') as f:
                f.write(query)
            
            # TODO: Execute query and reload data
            
            return ResponseBuilder.success('Query updated successfully')
            
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Update query")
