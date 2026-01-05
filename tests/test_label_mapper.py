"""
Tests for the LabelMapper and DisciplineConfig classes.
"""
import pytest
import numpy as np
import pandas as pd
from app.config import DisciplineConfig
from app.label_mapper import LabelMapper, get_label_mapper


class TestDisciplineConfig:
    """Tests for DisciplineConfig class."""
    
    def test_load_default_config(self):
        """Test loading the default discipline_codes.json file."""
        config = DisciplineConfig.load()
        assert config.is_loaded is True
        
    def test_phase_labels(self):
        """Test that phase labels are loaded correctly."""
        config = DisciplineConfig.load()
        labels = config.phase_labels
        assert len(labels) == 8
        assert 'ENGINEERING' in labels
        assert 'INSTALLATION_OFFSHORE' in labels
        
    def test_discipline_labels(self):
        """Test that discipline labels are loaded correctly."""
        config = DisciplineConfig.load()
        labels = config.discipline_labels
        assert 'ELECTRICAL' in labels
        assert 'PIPING' in labels
        assert 'STRUCTURAL' in labels
        
    def test_get_code_description(self):
        """Test getting description for a discipline code."""
        config = DisciplineConfig.load()
        desc = config.get_code_description('KE')
        assert desc is not None
        assert 'Electrical' in desc
        
    def test_get_code_description_invalid(self):
        """Test getting description for invalid code returns None."""
        config = DisciplineConfig.load()
        assert config.get_code_description('XX') is None
        
    def test_get_phase_for_code(self):
        """Test getting phase prefix for a discipline code."""
        config = DisciplineConfig.load()
        assert config.get_phase_for_code('KE') == 'K'
        assert config.get_phase_for_code('QL') == 'Q'
        assert config.get_phase_for_code('LN') == 'L'
        
    def test_get_labels_for_code(self):
        """Test getting ML labels for a discipline code."""
        config = DisciplineConfig.load()
        labels = config.get_labels_for_code('KE')
        assert labels['phase'] == 'ENGINEERING'
        assert labels['discipline'] == 'ELECTRICAL'
        assert labels['work_type'] == 'DIRECT'
        
    def test_get_phase_from_sub_project_id(self):
        """Test deriving phase from sub-project ID."""
        config = DisciplineConfig.load()
        assert config.get_phase_from_sub_project_id('3000') == 'ENGINEERING'
        assert config.get_phase_from_sub_project_id('7500') == 'INSTALLATION_OFFSHORE'
        assert config.get_phase_from_sub_project_id('5000') == 'FABRICATION'
        
    def test_get_contract_type_from_sub_project_id(self):
        """Test deriving contract type from sub-project ID."""
        config = DisciplineConfig.load()
        assert config.get_contract_type_from_sub_project_id('3000') == 'RB'
        assert config.get_contract_type_from_sub_project_id('3500') == 'LS'
        assert config.get_contract_type_from_sub_project_id('7000') == 'RB'
        assert config.get_contract_type_from_sub_project_id('7500') == 'LS'


class TestLabelMapper:
    """Tests for LabelMapper class."""
    
    def test_initialization(self):
        """Test LabelMapper initialization."""
        mapper = LabelMapper()
        assert mapper.num_labels > 0
        assert len(mapper.label_names) == mapper.num_labels
        
    def test_get_label_mapper_factory(self):
        """Test the factory function."""
        mapper = get_label_mapper()
        assert isinstance(mapper, LabelMapper)
        
    def test_extract_discipline_code(self):
        """Test extracting discipline codes from sub_project_id."""
        mapper = LabelMapper()
        assert mapper.extract_discipline_code('KE001') == 'KE'
        assert mapper.extract_discipline_code('QL050') == 'QL'
        assert mapper.extract_discipline_code('LN100') == 'LN'
        
    def test_extract_discipline_code_invalid(self):
        """Test extracting from invalid sub_project_id."""
        mapper = LabelMapper()
        assert mapper.extract_discipline_code('XX999') is None
        assert mapper.extract_discipline_code('') is None
        assert mapper.extract_discipline_code(None) is None
        
    def test_encode_labels_binary(self):
        """Test binary encoding of labels."""
        mapper = LabelMapper()
        labels = {
            'phase': 'ENGINEERING',
            'discipline': 'ELECTRICAL',
            'work_type': 'DIRECT',
            'location': None
        }
        binary = mapper.encode_labels_binary(labels)
        assert binary.shape == (mapper.num_labels,)
        assert binary.sum() == 3  # Three labels set
        
    def test_encode_from_discipline_code(self):
        """Test encoding directly from discipline code."""
        mapper = LabelMapper()
        binary = mapper.encode_from_discipline_code('QL')
        assert binary.shape == (mapper.num_labels,)
        # QL = Installation Offshore, Piping, Direct, Offshore
        assert binary.sum() == 4
        
    def test_decode_binary_labels(self):
        """Test decoding binary labels back to names."""
        mapper = LabelMapper()
        binary = mapper.encode_from_discipline_code('QL')
        decoded = mapper.decode_binary_labels(binary)
        
        assert 'INSTALLATION_OFFSHORE' in decoded['phase']
        assert 'PIPING' in decoded['discipline']
        assert 'DIRECT' in decoded['work_type']
        assert 'OFFSHORE' in decoded['location']
        
    def test_prepare_training_dataframe(self):
        """Test preparing a DataFrame for training."""
        mapper = LabelMapper()
        df = pd.DataFrame({
            'sub_project_id': ['KE001', 'QL050', 'LN100'],
            'activity_description': ['Test 1', 'Test 2', 'Test 3']
        })
        
        prepared = mapper.prepare_training_dataframe(df)
        
        assert 'discipline_code' in prepared.columns
        assert 'label_phase' in prepared.columns
        assert 'label_discipline' in prepared.columns
        assert 'binary_labels' in prepared.columns
        
        # Check extracted codes
        assert prepared['discipline_code'].tolist() == ['KE', 'QL', 'LN']
        
    def test_create_binary_label_matrix(self):
        """Test creating binary label matrix for a DataFrame."""
        mapper = LabelMapper()
        df = pd.DataFrame({
            'discipline_code': ['KE', 'QL', 'LN']
        })
        
        matrix = mapper.create_binary_label_matrix(df)
        assert matrix.shape == (3, mapper.num_labels)
        assert matrix.dtype == np.float32
        
    def test_validate_dataset(self):
        """Test dataset validation."""
        mapper = LabelMapper()
        df = pd.DataFrame({
            'sub_project_id': ['KE001', 'QL050', 'XX999', None]
        })
        
        stats = mapper.validate_dataset(df)
        assert stats['total_records'] == 4
        assert stats['valid_codes'] == 2
        assert stats['invalid_codes'] == 1
        assert stats['missing_codes'] == 1
        assert stats['coverage_pct'] == 50.0
        
    def test_label_counts(self):
        """Test that label counts are correct."""
        mapper = LabelMapper()
        assert mapper.num_phase_labels == 8
        assert mapper.num_discipline_labels == 17
        assert mapper.num_work_type_labels == 5
        assert mapper.num_location_labels == 2
        assert mapper.num_labels == 32


class TestLabelMapperIntegration:
    """Integration tests for the full labeling workflow."""
    
    def test_full_workflow(self):
        """Test the complete label mapping workflow."""
        # Simulate IFS activity data
        df = pd.DataFrame({
            'activity_seq': [1, 2, 3, 4, 5],
            'sub_project_id': ['KE001', 'QL050', 'LN100', 'HA010', 'NE200'],
            'activity_description': [
                'Design electrical systems',
                'Install piping module B',
                'Fabricate structural steel',
                'Project management report',
                'Install electrical panels'
            ]
        })
        
        mapper = LabelMapper()
        
        # Prepare data
        prepared = mapper.prepare_training_dataframe(df)
        
        # Verify all records have labels
        assert prepared['discipline_code'].notna().all()
        
        # Verify label correctness
        assert prepared.loc[0, 'label_phase'] == 'ENGINEERING'
        assert prepared.loc[1, 'label_phase'] == 'INSTALLATION_OFFSHORE'
        assert prepared.loc[2, 'label_phase'] == 'FABRICATION'
        assert prepared.loc[3, 'label_phase'] == 'MANAGEMENT'
        assert prepared.loc[4, 'label_phase'] == 'CONSTRUCTION_ONSHORE'
        
        # Verify binary labels can be decoded
        for idx in range(len(prepared)):
            binary = prepared.loc[idx, 'binary_labels']
            decoded = mapper.decode_binary_labels(binary)
            assert len(decoded['phase']) > 0
