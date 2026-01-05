"""
Routes for bootstrap sampling and export.

Provides API endpoints for:
- Generating diverse bootstrap samples
- Exporting samples for annotation (JSON, CSV, Markdown)
- Importing annotated samples back
"""
import os
import json
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file
import pandas as pd

from app.bootstrap_sampler import get_bootstrap_sampler
from app.incremental_training import get_incremental_training_service
from app.utils import DataFrameHelper

logger = logging.getLogger(__name__)

bootstrap_bp = Blueprint('bootstrap', __name__)


@bootstrap_bp.route('/api/bootstrap/status', methods=['GET'])
def get_bootstrap_status():
    """Get status of bootstrap data."""
    try:
        data_dir = 'data'
        
        # Check for various bootstrap files
        bootstrap_path = os.path.join(data_dir, 'bootstrap_sample.csv')
        annotated_path = os.path.join(data_dir, 'bootstrap_annotated.csv')
        training_data_path = os.path.join(data_dir, 'training_data.csv')
        
        status = {
            'has_bootstrap_sample': os.path.exists(bootstrap_path),
            'has_annotated_bootstrap': os.path.exists(annotated_path),
            'has_training_data': os.path.exists(training_data_path)
        }
        
        # Get counts if files exist
        if status['has_bootstrap_sample']:
            df = pd.read_csv(bootstrap_path)
            status['bootstrap_sample_count'] = len(df)
        
        if status['has_annotated_bootstrap']:
            df = pd.read_csv(annotated_path)
            status['annotated_count'] = len(df)
            # Count labeled rows
            labeled = df[df.get('categories', pd.Series()).notna() | 
                        df.get('DISCIPLINE_LABEL', pd.Series()).notna()]
            status['labeled_count'] = len(labeled) if not labeled.empty else 0
        
        if status['has_training_data']:
            df = pd.read_csv(training_data_path)
            status['training_data_count'] = len(df)
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting bootstrap status: {e}")
        return jsonify({'error': str(e)}), 500


@bootstrap_bp.route('/api/bootstrap/generate', methods=['POST'])
def generate_bootstrap_sample():
    """
    Generate a diverse bootstrap sample from loaded data.
    
    Request body:
    {
        "n_samples": 200,          // Number of samples to select
        "text_column": "ACTIVITY_DESCRIPTION",  // Column with text
        "diversity_method": "embeddings"  // or "random"
    }
    """
    try:
        # Get parameters
        data = request.get_json() or {}
        n_samples = data.get('n_samples', 200)
        text_column = data.get('text_column', 'ACTIVITY_DESCRIPTION')
        method = data.get('diversity_method', 'embeddings')
        
        # Load training data
        training_data_path = 'data/training_data.csv'
        if not os.path.exists(training_data_path):
            return jsonify({
                'error': 'No training data loaded. Please load data from database first.'
            }), 400
        
        df = pd.read_csv(training_data_path)
        logger.info(f"Loaded {len(df)} rows for bootstrap sampling")
        
        # Validate text column exists
        text_col_found = None
        for col in [text_column, text_column.lower(), 'description', 'DESCRIPTION']:
            if col in df.columns:
                text_col_found = col
                break
        
        if not text_col_found:
            return jsonify({
                'error': f'Text column not found. Available: {df.columns.tolist()}'
            }), 400
        
        # Get sampler and generate sample
        sampler = get_bootstrap_sampler()
        
        if method == 'random':
            # Simple random sampling
            sample_df = df.sample(n=min(n_samples, len(df)))
        else:
            # Diversity-based sampling
            sample_df = sampler.select_diverse_sample(
                df=df,
                n_samples=n_samples,
                text_column=text_col_found
            )
        
        # Save bootstrap sample
        os.makedirs('data', exist_ok=True)
        output_path = 'data/bootstrap_sample.csv'
        sample_df.to_csv(output_path, index=False)
        
        logger.info(f"Generated bootstrap sample with {len(sample_df)} items")
        
        return jsonify({
            'success': True,
            'n_samples': len(sample_df),
            'method': method,
            'output_path': output_path,
            'columns': sample_df.columns.tolist()
        })
        
    except Exception as e:
        logger.error(f"Error generating bootstrap sample: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@bootstrap_bp.route('/api/bootstrap/export', methods=['POST'])
def export_bootstrap_sample():
    """
    Export bootstrap sample in specified format.
    
    Request body:
    {
        "format": "json",  // "json", "csv", or "markdown"
        "include_context": true  // Include project/subproject info
    }
    """
    try:
        data = request.get_json() or {}
        export_format = data.get('format', 'json')
        include_context = data.get('include_context', True)
        
        # Load bootstrap sample
        bootstrap_path = 'data/bootstrap_sample.csv'
        if not os.path.exists(bootstrap_path):
            return jsonify({
                'error': 'No bootstrap sample generated. Generate one first.'
            }), 400
        
        df = pd.read_csv(bootstrap_path)
        sampler = get_bootstrap_sampler()
        
        # Export based on format
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if export_format == 'json':
            output_path = f'data/bootstrap_export_{timestamp}.json'
            sampler.export_for_annotation(df, output_path, format='json')
            
        elif export_format == 'csv':
            output_path = f'data/bootstrap_export_{timestamp}.csv'
            sampler.export_for_annotation(df, output_path, format='csv')
            
        elif export_format == 'markdown':
            output_path = f'data/bootstrap_export_{timestamp}.md'
            sampler.export_for_annotation(df, output_path, format='markdown')
            
        else:
            return jsonify({'error': f'Unknown format: {export_format}'}), 400
        
        # For JSON format, also return the data directly
        if export_format == 'json':
            with open(output_path, 'r', encoding='utf-8') as f:
                export_data = json.load(f)
            return jsonify({
                'success': True,
                'format': export_format,
                'output_path': output_path,
                'n_samples': len(df),
                'data': export_data
            })
        
        return jsonify({
            'success': True,
            'format': export_format,
            'output_path': output_path,
            'n_samples': len(df)
        })
        
    except Exception as e:
        logger.error(f"Error exporting bootstrap sample: {e}")
        return jsonify({'error': str(e)}), 500


@bootstrap_bp.route('/api/bootstrap/download/<format_type>')
def download_bootstrap(format_type):
    """Download the bootstrap export file."""
    try:
        import glob
        
        pattern = f'data/bootstrap_export_*.{format_type}'
        if format_type == 'markdown':
            pattern = 'data/bootstrap_export_*.md'
        
        files = glob.glob(pattern)
        if not files:
            return jsonify({'error': f'No {format_type} export found'}), 404
        
        # Get most recent
        latest = max(files, key=os.path.getmtime)
        
        return send_file(
            latest,
            as_attachment=True,
            download_name=os.path.basename(latest)
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bootstrap_bp.route('/api/bootstrap/import', methods=['POST'])
def import_annotated_bootstrap():
    """
    Import annotated bootstrap data.
    
    Accepts JSON with annotated samples or CSV file upload.
    """
    try:
        content_type = request.content_type or ''
        
        if 'application/json' in content_type:
            data = request.get_json()
            
            if 'samples' not in data:
                return jsonify({'error': 'Missing "samples" field in JSON'}), 400
            
            samples = data['samples']
            df = pd.DataFrame(samples)
            
        elif 'multipart/form-data' in content_type:
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith('.json'):
                df = pd.read_json(file)
            else:
                return jsonify({'error': 'Unsupported file format'}), 400
        else:
            return jsonify({'error': 'Unsupported content type'}), 400
        
        # Validate required columns
        required = ['ACTIVITY_DESCRIPTION']
        label_cols = ['categories', 'PHASE_LABEL', 'DISCIPLINE_LABEL', 'WORK_TYPE', 'LOCATION']
        
        has_text = any(col in df.columns for col in ['ACTIVITY_DESCRIPTION', 'description', 'text'])
        has_labels = any(col in df.columns for col in label_cols)
        
        if not has_text:
            return jsonify({'error': 'Missing text column (ACTIVITY_DESCRIPTION)'}), 400
        
        if not has_labels:
            return jsonify({
                'warning': 'No label columns found. Expected one of: ' + ', '.join(label_cols),
                'columns_found': df.columns.tolist()
            }), 400
        
        # Save annotated data
        output_path = 'data/bootstrap_annotated.csv'
        df.to_csv(output_path, index=False)
        
        # Count labeled rows
        labeled_count = 0
        for col in label_cols:
            if col in df.columns:
                labeled_count = max(labeled_count, df[col].notna().sum())
        
        # Update training service
        training_service = get_incremental_training_service()
        stats = training_service.get_annotation_stats()
        
        return jsonify({
            'success': True,
            'imported_count': len(df),
            'labeled_count': int(labeled_count),
            'output_path': output_path,
            'ready_for_training': stats['ready_for_initial_train']
        })
        
    except Exception as e:
        logger.error(f"Error importing bootstrap data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@bootstrap_bp.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get status of incremental training system."""
    try:
        service = get_incremental_training_service()
        status = service.get_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        return jsonify({'error': str(e)}), 500


@bootstrap_bp.route('/api/training/trigger', methods=['POST'])
def trigger_training():
    """
    Trigger model training with current annotations.
    
    Request body:
    {
        "epochs": 5,           // Optional, auto-determined if not set
        "early_stopping": true // Optional, default true
    }
    """
    try:
        data = request.get_json() or {}
        epochs = data.get('epochs')
        early_stopping = data.get('early_stopping', True)
        
        service = get_incremental_training_service()
        
        # Check if we have enough data
        stats = service.get_annotation_stats()
        if stats['total_annotations'] < stats['min_samples_needed']:
            return jsonify({
                'error': f"Not enough training data. Have {stats['total_annotations']}, "
                        f"need {stats['min_samples_needed']}."
            }), 400
        
        # Import the classifier
        # This would connect to your actual model
        from app.multilabel_model import MultiLabelClassifier
        
        # Get or create classifier
        classifier = MultiLabelClassifier()
        
        # Trigger training
        session = service.trigger_training(
            classifier=classifier,
            epochs=epochs,
            early_stopping=early_stopping
        )
        
        return jsonify({
            'success': True,
            'session_id': session.session_id,
            'n_samples': session.n_samples,
            'metrics': session.metrics,
            'model_path': session.model_path
        })
        
    except Exception as e:
        logger.error(f"Error triggering training: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@bootstrap_bp.route('/api/training/history', methods=['GET'])
def get_training_history():
    """Get history of all training sessions."""
    try:
        service = get_incremental_training_service()
        history = service.get_training_history()
        return jsonify({
            'sessions': history,
            'total_sessions': len(history)
        })
    except Exception as e:
        logger.error(f"Error getting training history: {e}")
        return jsonify({'error': str(e)}), 500
