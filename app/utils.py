"""
Common utilities and helper functions following DRY principles.

This module centralizes repeated operations across the codebase to eliminate
duplication and improve maintainability.
"""
import os
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from flask import jsonify
import logging

logger = logging.getLogger(__name__)


class DataManager:
    """Centralized data file management to eliminate duplication."""
    
    DATA_DIR = 'data'
    TRAINING_DATA_PATH = os.path.join(DATA_DIR, 'training_data.csv')
    ANNOTATIONS_PATH = os.path.join(DATA_DIR, 'annotations.csv')
    QUERY_PATH = os.path.join(DATA_DIR, 'annotation_query.sql')
    
    @classmethod
    def ensure_data_dir(cls) -> None:
        """Ensure data directory exists."""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
    
    @classmethod
    def has_cached_data(cls) -> bool:
        """Check if cached training data exists."""
        return os.path.exists(cls.TRAINING_DATA_PATH)
    
    @classmethod
    def get_cache_info(cls) -> Optional[Dict[str, Any]]:
        """
        Get information about cached training data.
        
        Returns:
            Dictionary with cache info, or None if no cache exists
        """
        if not cls.has_cached_data():
            return None
        
        try:
            stat = os.stat(cls.TRAINING_DATA_PATH)
            from datetime import datetime
            
            # Get row count without loading entire file
            with open(cls.TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
                row_count = sum(1 for _ in f) - 1  # Subtract header
            
            return {
                'exists': True,
                'path': cls.TRAINING_DATA_PATH,
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'row_count': row_count
            }
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {'exists': True, 'error': str(e)}
    
    @classmethod
    def clear_cache(cls) -> bool:
        """
        Clear cached training data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(cls.TRAINING_DATA_PATH):
                os.remove(cls.TRAINING_DATA_PATH)
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    @classmethod
    def load_training_data(cls) -> Optional[pd.DataFrame]:
        """
        Load training data from standard location.
        
        Returns:
            DataFrame if successful, None if file doesn't exist
        """
        if not os.path.exists(cls.TRAINING_DATA_PATH):
            return None
        try:
            return pd.read_csv(cls.TRAINING_DATA_PATH)
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return None
    
    @classmethod
    def save_training_data(cls, df: pd.DataFrame) -> bool:
        """
        Save training data to standard location.
        
        Args:
            df: DataFrame to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cls.ensure_data_dir()
            df.to_csv(cls.TRAINING_DATA_PATH, index=False)
            return True
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
            return False
    
    @classmethod
    def load_annotations(cls) -> pd.DataFrame:
        """
        Load annotations from standard location.
        
        Returns:
            DataFrame with annotations, or empty DataFrame with correct columns
        """
        if os.path.exists(cls.ANNOTATIONS_PATH):
            try:
                return pd.read_csv(cls.ANNOTATIONS_PATH)
            except Exception as e:
                logger.error(f"Error loading annotations: {e}")
        
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=[
            'activity_id', 'categories', 'uncertainty_score', 'timestamp', 'label'
        ])
    
    @classmethod
    def save_annotations(cls, df: pd.DataFrame) -> bool:
        """
        Save annotations to standard location.
        
        Args:
            df: DataFrame with annotations
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cls.ensure_data_dir()
            df.to_csv(cls.ANNOTATIONS_PATH, index=False)
            return True
        except Exception as e:
            logger.error(f"Error saving annotations: {e}")
            return False


class ResponseBuilder:
    """Centralized JSON response building to eliminate duplication."""
    
    @staticmethod
    def success(message: str, data: Optional[Dict[str, Any]] = None, status: int = 200) -> Tuple[Any, int]:
        """
        Build a success response.
        
        Args:
            message: Success message
            data: Optional additional data to include
            status: HTTP status code
            
        Returns:
            Tuple of (response, status_code)
        """
        response = {'success': True, 'message': message}
        if data:
            response.update(data)
        
        # Try to use Flask's jsonify if in app context, otherwise return dict
        try:
            from flask import current_app
            if current_app:
                return jsonify(response), status
        except (ImportError, RuntimeError):
            pass
        
        return response, status
    
    @staticmethod
    def error(message: str, status: int = 500, details: Optional[Dict[str, Any]] = None) -> Tuple[Any, int]:
        """
        Build an error response.
        
        Args:
            message: Error message
            status: HTTP status code
            details: Optional additional error details
            
        Returns:
            Tuple of (response, status_code)
        """
        response = {'success': False, 'message': message}
        if details:
            response['details'] = details
        
        # Try to use Flask's jsonify if in app context, otherwise return dict
        try:
            from flask import current_app
            if current_app:
                return jsonify(response), status
        except (ImportError, RuntimeError):
            pass
        
        return response, status
    
    @staticmethod
    def from_exception(e: Exception, context: str = "", status: int = 500) -> Tuple[Any, int]:
        """
        Build an error response from an exception.
        
        Args:
            e: Exception that occurred
            context: Optional context about what was being done
            status: HTTP status code
            
        Returns:
            Tuple of (response, status_code)
        """
        prefix = f"{context}: " if context else ""
        message = f"{prefix}{str(e)}"
        logger.error(message, exc_info=True)
        return ResponseBuilder.error(message, status)


class FileValidator:
    """Common file validation operations."""
    
    @staticmethod
    def validate_csv_upload(file) -> Tuple[bool, str]:
        """
        Validate uploaded CSV file.
        
        Args:
            file: Flask file upload object
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file or file.filename == '':
            return False, 'No file selected'
        
        if not file.filename.endswith('.csv'):
            return False, 'File must be a CSV'
        
        return True, ''
    
    @staticmethod
    def check_file_exists(path: str, description: str = "file") -> Tuple[bool, str]:
        """
        Check if a file exists and return user-friendly message.
        
        Args:
            path: Path to check
            description: Description of the file for error message (e.g., "training data", "annotations")
            
        Returns:
            Tuple of (exists, error_message)
        """
        if not os.path.exists(path):
            return False, f'No {description} found'
        return True, ''


class DataFrameHelper:
    """Helper functions for DataFrame operations."""
    
    @staticmethod
    def get_preview(df: pd.DataFrame, n_rows: int = 10) -> Dict[str, Any]:
        """
        Get a preview of DataFrame for JSON response.
        
        Args:
            df: DataFrame to preview
            n_rows: Number of rows to include
            
        Returns:
            Dictionary with preview data
        """
        # Replace NaN/NaT with None for valid JSON serialization
        preview_df = df.head(n_rows).copy()
        preview_df = preview_df.where(pd.notnull(preview_df), None)
        
        # Convert to records, handling any remaining edge cases
        records = preview_df.to_dict('records')
        
        # Ensure no NaN values remain (double-check for edge cases)
        import math
        for record in records:
            for key, value in record.items():
                if isinstance(value, float) and math.isnan(value):
                    record[key] = None
        
        return {
            'preview': records,
            'columns': df.columns.tolist(),
            'row_count': len(df)
        }
    
    @staticmethod
    def merge_annotation(
        annotations_df: pd.DataFrame,
        activity_id: str,
        categories: list,
        uncertainty_score: float
    ) -> pd.DataFrame:
        """
        Add or update an annotation in the DataFrame.
        
        Args:
            annotations_df: Existing annotations DataFrame
            activity_id: ID of activity to annotate
            categories: List of category labels
            uncertainty_score: Uncertainty score from model
            
        Returns:
            Updated DataFrame
        """
        from datetime import datetime
        
        # Remove existing annotation if any
        annotations_df = annotations_df[annotations_df['activity_id'] != activity_id]
        
        # Create new annotation
        new_annotation = {
            'activity_id': activity_id,
            'categories': ','.join(categories),
            'uncertainty_score': uncertainty_score,
            'timestamp': datetime.now().isoformat(),
            'label': 1 if categories else None
        }
        
        # Add to DataFrame
        return pd.concat([
            annotations_df,
            pd.DataFrame([new_annotation])
        ], ignore_index=True)
