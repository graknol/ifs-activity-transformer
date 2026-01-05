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
    
    @app.route('/api/annotate/activities', methods=['GET'])
    def get_activities_for_annotation():
        """
        Get activities for multi-label annotation.
        Returns activities from bootstrap_sample.csv or bootstrap_annotated.csv.
        """
        try:
            # Try to load bootstrap_annotated.csv first (has existing labels)
            annotated_path = 'data/bootstrap_annotated.csv'
            sample_path = 'data/bootstrap_sample.csv'
            annotations_path = 'data/annotations.csv'
            
            df = None
            existing_annotations = {}
            
            # Priority: annotations.csv > bootstrap_annotated.csv > bootstrap_sample.csv
            if os.path.exists(annotations_path):
                df = pd.read_csv(annotations_path)
            elif os.path.exists(annotated_path):
                df = pd.read_csv(annotated_path)
            elif os.path.exists(sample_path):
                df = pd.read_csv(sample_path)
            
            if df is None or df.empty:
                return ResponseBuilder.success(
                    'No bootstrap sample found',
                    {'activities': [], 'existing_annotations': {}, 'stats': {'total': 0, 'annotated': 0}}
                )
            
            # Replace NaN with None for JSON compatibility
            df = df.where(pd.notna(df), None)
            
            # Convert to list of dicts for frontend
            activities = df.to_dict('records')
            
            # Clean up any remaining NaN/float('nan') values
            import math
            def clean_value(v):
                if v is None:
                    return None
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    return None
                return v
            
            activities = [
                {k: clean_value(v) for k, v in row.items()}
                for row in activities
            ]
            
            # Build existing annotations dict from CSV labels
            for row in activities:
                activity_id = row.get('ACTIVITY_SEQ') or row.get('id')
                if activity_id:
                    ann = {}
                    if row.get('PHASE_LABEL'):
                        ann['phase'] = row['PHASE_LABEL']
                    if row.get('DISCIPLINE_LABEL'):
                        ann['discipline'] = row['DISCIPLINE_LABEL']
                    if row.get('WORK_TYPE_LABEL'):
                        ann['worktype'] = row['WORK_TYPE_LABEL']
                    if row.get('LOCATION_LABEL'):
                        ann['location'] = row['LOCATION_LABEL']
                    if row.get('NOTES'):
                        ann['notes'] = row['NOTES']
                    if ann:
                        existing_annotations[str(activity_id)] = ann
            
            # Calculate stats
            total = len(activities)
            annotated = sum(1 for a in activities 
                          if a.get('PHASE_LABEL') and a.get('DISCIPLINE_LABEL') 
                          and a.get('WORK_TYPE_LABEL') and a.get('LOCATION_LABEL'))
            
            return ResponseBuilder.success(
                f'Loaded {len(activities)} activities',
                {
                    'activities': activities,
                    'existing_annotations': existing_annotations,
                    'stats': {
                        'total': total,
                        'annotated': annotated
                    }
                }
            )
            
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Get activities")
    
    @app.route('/api/annotate/save', methods=['POST'])
    def save_annotation():
        """Save multi-label annotation for an activity."""
        try:
            data = request.get_json()
            activity_seq = data.get('activity_seq')
            phase_label = data.get('phase_label')
            discipline_label = data.get('discipline_label')
            work_type_label = data.get('work_type_label')
            location_label = data.get('location_label')
            notes = data.get('notes', '')
            
            if not activity_seq:
                return ResponseBuilder.error('Missing activity_seq', 400)
            
            # Load the annotations file
            annotations_path = 'data/annotations.csv'
            annotated_path = 'data/bootstrap_annotated.csv'
            
            # Load existing data
            if os.path.exists(annotations_path):
                df = pd.read_csv(annotations_path)
            elif os.path.exists(annotated_path):
                df = pd.read_csv(annotated_path)
            else:
                return ResponseBuilder.error('No annotation file found', 404)
            
            # Find and update the row
            # Handle both string and numeric ACTIVITY_SEQ
            mask = df['ACTIVITY_SEQ'].astype(str) == str(activity_seq)
            
            if not mask.any():
                return ResponseBuilder.error(f'Activity {activity_seq} not found', 404)
            
            # Update labels
            df.loc[mask, 'PHASE_LABEL'] = phase_label
            df.loc[mask, 'DISCIPLINE_LABEL'] = discipline_label
            df.loc[mask, 'WORK_TYPE_LABEL'] = work_type_label
            df.loc[mask, 'LOCATION_LABEL'] = location_label
            df.loc[mask, 'NOTES'] = notes
            
            # Save back
            df.to_csv(annotations_path, index=False)
            
            return ResponseBuilder.success('Annotation saved', {'activity_seq': activity_seq})
            
        except Exception as e:
            return ResponseBuilder.from_exception(e, "Save annotation")
    
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
