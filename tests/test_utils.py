"""
Unit tests for utility functions (app/utils.py).

Tests the centralized utility classes following DRY principles.
"""
import pytest
import pandas as pd
import os
from unittest.mock import patch, mock_open
from app.utils import DataManager, ResponseBuilder, FileValidator, DataFrameHelper


class TestDataManager:
    """Test DataManager class."""
    
    def test_ensure_data_dir(self, temp_dir):
        """Test ensuring data directory exists."""
        with patch.object(DataManager, 'DATA_DIR', temp_dir):
            DataManager.ensure_data_dir()
            assert os.path.exists(temp_dir)
    
    def test_save_and_load_training_data(self, temp_dir, sample_dataframe):
        """Test saving and loading training data."""
        training_path = os.path.join(temp_dir, 'training_data.csv')
        
        with patch.object(DataManager, 'DATA_DIR', temp_dir):
            with patch.object(DataManager, 'TRAINING_DATA_PATH', training_path):
                # Save data
                success = DataManager.save_training_data(sample_dataframe)
                assert success is True
                assert os.path.exists(training_path)
                
                # Load data
                loaded_df = DataManager.load_training_data()
                assert loaded_df is not None
                assert len(loaded_df) == len(sample_dataframe)
                assert list(loaded_df.columns) == list(sample_dataframe.columns)
    
    def test_load_training_data_nonexistent(self, temp_dir):
        """Test loading training data when file doesn't exist."""
        training_path = os.path.join(temp_dir, 'nonexistent.csv')
        
        with patch.object(DataManager, 'TRAINING_DATA_PATH', training_path):
            loaded_df = DataManager.load_training_data()
            assert loaded_df is None
    
    def test_load_annotations_empty(self, temp_dir):
        """Test loading annotations when file doesn't exist returns empty DataFrame."""
        annotations_path = os.path.join(temp_dir, 'annotations.csv')
        
        with patch.object(DataManager, 'ANNOTATIONS_PATH', annotations_path):
            loaded_df = DataManager.load_annotations()
            assert len(loaded_df) == 0
            assert 'activity_id' in loaded_df.columns
            assert 'categories' in loaded_df.columns
    
    def test_save_and_load_annotations(self, temp_dir, sample_annotations):
        """Test saving and loading annotations."""
        annotations_path = os.path.join(temp_dir, 'annotations.csv')
        
        with patch.object(DataManager, 'DATA_DIR', temp_dir):
            with patch.object(DataManager, 'ANNOTATIONS_PATH', annotations_path):
                # Save annotations
                success = DataManager.save_annotations(sample_annotations)
                assert success is True
                assert os.path.exists(annotations_path)
                
                # Load annotations
                loaded_df = DataManager.load_annotations()
                assert len(loaded_df) == len(sample_annotations)
                assert 'activity_id' in loaded_df.columns


class TestResponseBuilder:
    """Test ResponseBuilder class."""
    
    def test_success_response_basic(self):
        """Test creating a basic success response."""
        response, status = ResponseBuilder.success("Operation successful")
        
        assert status == 200
        response_json = response if isinstance(response, dict) else response.get_json()
        assert response_json['success'] is True
        assert response_json['message'] == "Operation successful"
    
    def test_success_response_with_data(self):
        """Test creating success response with additional data."""
        data = {'count': 10, 'items': ['a', 'b', 'c']}
        response, status = ResponseBuilder.success("Data retrieved", data=data)
        
        assert status == 200
        response_json = response if isinstance(response, dict) else response.get_json()
        assert response_json['success'] is True
        assert response_json['count'] == 10
        assert response_json['items'] == ['a', 'b', 'c']
    
    def test_success_response_custom_status(self):
        """Test creating success response with custom status code."""
        response, status = ResponseBuilder.success("Created", status=201)
        
        assert status == 201
        response_json = response if isinstance(response, dict) else response.get_json()
        assert response_json['success'] is True
    
    def test_error_response_basic(self):
        """Test creating a basic error response."""
        response, status = ResponseBuilder.error("An error occurred")
        
        assert status == 500
        response_json = response if isinstance(response, dict) else response.get_json()
        assert response_json['success'] is False
        assert response_json['message'] == "An error occurred"
    
    def test_error_response_custom_status(self):
        """Test creating error response with custom status code."""
        response, status = ResponseBuilder.error("Not found", status=404)
        
        assert status == 404
        response_json = response if isinstance(response, dict) else response.get_json()
        assert response_json['success'] is False
    
    def test_error_response_with_details(self):
        """Test creating error response with details."""
        details = {'field': 'email', 'error': 'invalid format'}
        response, status = ResponseBuilder.error("Validation failed", status=400, details=details)
        
        assert status == 400
        response_json = response if isinstance(response, dict) else response.get_json()
        assert response_json['success'] is False
        assert response_json['details']['field'] == 'email'
    
    def test_from_exception(self):
        """Test creating error response from exception."""
        exception = ValueError("Invalid input value")
        response, status = ResponseBuilder.from_exception(exception, context="Processing data")
        
        assert status == 500
        response_json = response if isinstance(response, dict) else response.get_json()
        assert response_json['success'] is False
        assert "Processing data" in response_json['message']
        assert "Invalid input value" in response_json['message']
    
    def test_from_exception_without_context(self):
        """Test creating error response from exception without context."""
        exception = RuntimeError("Something went wrong")
        response, status = ResponseBuilder.from_exception(exception)
        
        assert status == 500
        response_json = response if isinstance(response, dict) else response.get_json()
        assert "Something went wrong" in response_json['message']


class TestFileValidator:
    """Test FileValidator class."""
    
    def test_validate_csv_upload_success(self):
        """Test validating a valid CSV upload."""
        from unittest.mock import MagicMock
        mock_file = MagicMock()
        mock_file.filename = 'data.csv'
        
        is_valid, error_msg = FileValidator.validate_csv_upload(mock_file)
        assert is_valid is True
        assert error_msg == ''
    
    def test_validate_csv_upload_no_file(self):
        """Test validating when no file is provided."""
        is_valid, error_msg = FileValidator.validate_csv_upload(None)
        assert is_valid is False
        assert 'No file selected' in error_msg
    
    def test_validate_csv_upload_empty_filename(self):
        """Test validating when filename is empty."""
        from unittest.mock import MagicMock
        mock_file = MagicMock()
        mock_file.filename = ''
        
        is_valid, error_msg = FileValidator.validate_csv_upload(mock_file)
        assert is_valid is False
        assert 'No file selected' in error_msg
    
    def test_validate_csv_upload_wrong_extension(self):
        """Test validating when file is not CSV."""
        from unittest.mock import MagicMock
        mock_file = MagicMock()
        mock_file.filename = 'data.xlsx'
        
        is_valid, error_msg = FileValidator.validate_csv_upload(mock_file)
        assert is_valid is False
        assert 'must be a CSV' in error_msg
    
    def test_check_file_exists_true(self, temp_dir):
        """Test checking if a file exists."""
        test_file = os.path.join(temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        
        exists, error_msg = FileValidator.check_file_exists(test_file, "test file")
        assert exists is True
        assert error_msg == ''
    
    def test_check_file_exists_false(self):
        """Test checking if a nonexistent file exists."""
        exists, error_msg = FileValidator.check_file_exists('/nonexistent/file.txt', "data file")
        assert exists is False
        assert 'No data file found' in error_msg


class TestDataFrameHelper:
    """Test DataFrameHelper class."""
    
    def test_get_preview(self, sample_dataframe):
        """Test getting DataFrame preview."""
        preview = DataFrameHelper.get_preview(sample_dataframe, n_rows=2)
        
        assert 'preview' in preview
        assert 'columns' in preview
        assert 'row_count' in preview
        assert len(preview['preview']) == 2
        assert preview['row_count'] == 3
        assert 'activity_id' in preview['columns']
    
    def test_get_preview_all_rows(self, sample_dataframe):
        """Test getting preview with more rows than available."""
        preview = DataFrameHelper.get_preview(sample_dataframe, n_rows=100)
        
        assert len(preview['preview']) == 3  # Only 3 rows available
        assert preview['row_count'] == 3
    
    def test_merge_annotation_new(self):
        """Test adding a new annotation."""
        annotations_df = pd.DataFrame(columns=[
            'activity_id', 'categories', 'uncertainty_score', 'timestamp', 'label'
        ])
        
        result_df = DataFrameHelper.merge_annotation(
            annotations_df,
            activity_id='ACT001',
            categories=['engineering', 'procurement'],
            uncertainty_score=0.25
        )
        
        assert len(result_df) == 1
        assert result_df.iloc[0]['activity_id'] == 'ACT001'
        assert result_df.iloc[0]['categories'] == 'engineering,procurement'
        assert result_df.iloc[0]['uncertainty_score'] == 0.25
    
    def test_merge_annotation_update(self, sample_annotations):
        """Test updating an existing annotation."""
        result_df = DataFrameHelper.merge_annotation(
            sample_annotations,
            activity_id='ACT001',
            categories=['construction'],
            uncertainty_score=0.10
        )
        
        # Should still have 2 rows (one updated, not added)
        assert len(result_df) == 2
        
        # Find the updated annotation
        updated = result_df[result_df['activity_id'] == 'ACT001'].iloc[0]
        assert updated['categories'] == 'construction'
        assert updated['uncertainty_score'] == 0.10
    
    def test_merge_annotation_empty_categories(self):
        """Test merging annotation with empty categories."""
        annotations_df = pd.DataFrame(columns=[
            'activity_id', 'categories', 'uncertainty_score', 'timestamp', 'label'
        ])
        
        result_df = DataFrameHelper.merge_annotation(
            annotations_df,
            activity_id='ACT999',
            categories=[],
            uncertainty_score=0.50
        )
        
        assert len(result_df) == 1
        assert result_df.iloc[0]['categories'] == ''
        assert pd.isna(result_df.iloc[0]['label'])
