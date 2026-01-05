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
            data = request.get_json() or {}
            batch_size = data.get('batch_size', 20)
            strategy = data.get('sampling_strategy', 'least_confidence')
            
            # Load data from cache
            df = DataManager.load_training_data()
            if df is None or df.empty:
                # Check if cache exists
                cache_info = DataManager.get_cache_info()
                if cache_info and cache_info.get('exists'):
                    return ResponseBuilder.error(
                        'Failed to load cached training data. Try re-fetching from database.',
                        500
                    )
                return ResponseBuilder.error(
                    'No training data found. Please load data from the Database page first.',
                    404,
                    details={'hint': 'Go to Data page and fetch data from Oracle or load from cache'}
                )
            
            # Identify column names (case-insensitive matching)
            columns = {col.upper(): col for col in df.columns}
            
            # Find ID column
            id_col = None
            for name in ['ACTIVITY_SEQ', 'ACTIVITY_ID', 'ID']:
                if name in columns:
                    id_col = columns[name]
                    break
            if not id_col:
                id_col = df.columns[0]  # Fallback to first column
            
            # Find description column
            desc_col = None
            for name in ['ACTIVITY_DESCRIPTION', 'DESCRIPTION', 'SHORT_NAME']:
                if name in columns:
                    desc_col = columns[name]
                    break
            if not desc_col:
                desc_col = df.columns[1] if len(df.columns) > 1 else id_col
            
            # Find name column
            name_col = None
            for name in ['SHORT_NAME', 'ACTIVITY_NO', 'NAME']:
                if name in columns:
                    name_col = columns[name]
                    break
            if not name_col:
                name_col = desc_col
            
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
            sample_ids = df[id_col].astype(str).tolist()
            
            # Get early_start dates if available for recency weighting
            early_start_dates = None
            for name in ['EARLY_START', 'early_start', 'Early_Start']:
                if name in columns:
                    early_start_dates = df[columns[name]].tolist()
                    break
            
            selected = al_service.select_multilabel_samples(
                predictions,
                sample_ids,
                n_samples=min(batch_size, len(df)),
                sanity_check_ratio=0.1,
                early_start_dates=early_start_dates,
                recency_weight=0.3  # 30% weight to recency, 70% to uncertainty
            )
            
            # Build activity objects for UI
            activities = []
            for item in selected:
                idx = item['index']
                row = df.iloc[idx]
                
                # Get additional context fields if available
                project_name = row.get(columns.get('PROJECT_NAME', ''), '') if 'PROJECT_NAME' in columns else ''
                discipline_code = row.get(columns.get('DISCIPLINE_CODE', ''), '') if 'DISCIPLINE_CODE' in columns else ''
                sub_project_desc = row.get(columns.get('SUB_PROJECT_DESCRIPTION', ''), '') if 'SUB_PROJECT_DESCRIPTION' in columns else ''
                
                activity = {
                    'id': str(row[id_col]),
                    'name': str(row[name_col]) if pd.notna(row[name_col]) else 'Unnamed',
                    'description': str(row[desc_col]) if pd.notna(row[desc_col]) else '',
                    'uncertainty_score': item['uncertainty_score'],
                    'max_confidence': 1 - item['uncertainty_score'],
                    'predicted_categories': [],  # TODO: decode from predictions
                    'project_name': str(project_name) if pd.notna(project_name) else '',
                    'discipline_code': str(discipline_code) if pd.notna(discipline_code) else '',
                    'sub_project_description': str(sub_project_desc) if pd.notna(sub_project_desc) else '',
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

    @app.route('/api/annotate/data-status', methods=['GET'])
    def get_data_status():
        """Get status of training data for annotation."""
        try:
            cache_info = DataManager.get_cache_info()
            annotations_df = DataManager.load_annotations()
            
            # Get annotation statistics
            al_service = ActiveLearningService()
            stats = al_service.get_annotation_statistics(annotations_df)
            
            if cache_info and cache_info.get('exists'):
                return ResponseBuilder.success(
                    f'Training data loaded: {cache_info.get("row_count", 0)} activities',
                    {
                        'data_loaded': True,
                        'cache': cache_info,
                        'statistics': stats
                    }
                )
            else:
                return ResponseBuilder.success(
                    'No training data loaded',
                    {
                        'data_loaded': False,
                        'cache': None,
                        'statistics': stats,
                        'hint': 'Go to Data page to load training data from Oracle database or CSV'
                    }
                )
                
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Data status")
