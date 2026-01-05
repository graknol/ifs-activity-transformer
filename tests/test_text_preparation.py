"""
Tests for text preparation module.

Validates that activity data is optimally shaped for RoBERTa input.
"""
import pytest
import pandas as pd
import numpy as np
from app.text_preparation import (
    ActivityTextPreparer,
    TextPrepConfig,
    TemplateBasedPreparer,
    create_training_text,
    compare_text_strategies
)


@pytest.fixture
def sample_activity_data():
    """Sample activity data matching SQL query output."""
    return pd.DataFrame({
        'activity_seq': [1, 2, 3],
        'project_id': ['P001', 'P001', 'P002'],
        'sub_project_id': ['NE-1000-EL', 'MF-2000-PI', 'QO-3000-ST'],
        'activity_no': ['ACT-001', 'ACT-002', 'ACT-003'],
        'short_name': ['Cable installation E-deck', 'Pipe spool fab', 'Steel erection'],
        'activity_description': [
            'Install electrical cable trays and cables in module B deck 3',
            'Fabricate piping spool for process system P-101',
            'Erect structural steel for platform deck'
        ],
        'activity_status': ['Completed', 'Completed', 'Released'],
        'plan_hrs': [450.0, 200.0, 800.0],
        'used_hrs': [380.0, 210.0, 0.0],
        'early_start': ['2024-01-15', '2024-02-01', '2024-03-01'],
        'early_finish': ['2024-02-15', '2024-02-28', '2024-04-30'],
        'discipline_code': ['NE', 'MF', 'QO'],
        'phase_prefix': ['N', 'M', 'Q'],
        'sub_project_description': [
            'Electrical Installation - Onshore',
            'Piping Fabrication - Yard',
            'Structural Offshore Installation'
        ],
        'parent_sub_project_id': ['NE-1000', 'MF-2000', 'QO-3000'],
        'project_name': ['Johan Sverdrup P2', 'Johan Sverdrup P2', 'Aker BP Topside'],
        'project_description': [
            'Topside electrical systems',
            'Process piping systems',
            'Platform structural work'
        ],
        'customer_id': ['CUST-01', 'CUST-01', 'CUST-02'],
        'project_category_id': ['OIL_GAS', 'OIL_GAS', 'OIL_GAS']
    })


class TestTextPrepConfig:
    """Tests for TextPrepConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TextPrepConfig()
        
        assert config.include_project_context is True
        assert config.include_sub_project_context is True
        assert config.include_hierarchy_context is True
        assert config.include_metadata is True
        assert config.use_structured_template is True
        assert config.use_separator_tokens is True
        assert config.normalize_whitespace is True
        assert config.lowercase is False
        assert config.max_description_length == 300
        assert config.max_context_length == 200
    
    def test_minimal_config(self):
        """Test minimal configuration (description only)."""
        config = TextPrepConfig(
            include_project_context=False,
            include_sub_project_context=False,
            include_hierarchy_context=False,
            include_metadata=False
        )
        
        assert config.include_project_context is False
        assert config.include_metadata is False


class TestActivityTextPreparer:
    """Tests for ActivityTextPreparer."""
    
    def test_prepare_single_full_context(self):
        """Test preparing single activity with full context."""
        preparer = ActivityTextPreparer()
        
        result = preparer.prepare_single(
            activity_description="Install cable trays in module B",
            short_name="E-CABLE-001",
            sub_project_description="Electrical Installation - Onshore",
            project_name="Johan Sverdrup Phase 2",
            plan_hrs=450,
            used_hrs=380,
            progress=85
        )
        
        # Should contain activity description
        assert "Activity: Install cable trays in module B" in result
        
        # Should contain task code
        assert "Task: E-CABLE-001" in result
        
        # Should contain sub-project context
        assert "Sub-project: Electrical Installation - Onshore" in result
        
        # Should contain project context
        assert "Project: Johan Sverdrup Phase 2" in result
        
        # Should contain status with hours
        assert "450 planned hours" in result
        assert "380 used" in result
        
        # Should use [SEP] tokens
        assert "[SEP]" in result
    
    def test_prepare_single_minimal(self):
        """Test minimal preparation (description only)."""
        preparer = ActivityTextPreparer()
        
        result = preparer.prepare_minimal("Install cable trays in module B deck 3")
        
        assert result == "Install cable trays in module B deck 3"
        assert "[SEP]" not in result
    
    def test_prepare_single_no_metadata(self):
        """Test preparation without metadata."""
        config = TextPrepConfig(include_metadata=False)
        preparer = ActivityTextPreparer(config)
        
        result = preparer.prepare_single(
            activity_description="Install cable trays",
            short_name="E-001",
            plan_hrs=450,
            used_hrs=380
        )
        
        # Should NOT contain hours
        assert "planned hours" not in result
        
        # Should still have activity
        assert "Activity: Install cable trays" in result
    
    def test_prepare_single_no_separator_tokens(self):
        """Test preparation without [SEP] tokens."""
        config = TextPrepConfig(use_separator_tokens=False)
        preparer = ActivityTextPreparer(config)
        
        result = preparer.prepare_single(
            activity_description="Install cable trays",
            sub_project_description="Electrical Installation"
        )
        
        assert "[SEP]" not in result
        assert "Activity: Install cable trays" in result
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        preparer = ActivityTextPreparer()
        
        # Test whitespace normalization
        result = preparer._clean_text("  Multiple   spaces  and\nnewlines\r\n  ", 100)
        assert "  " not in result
        assert "\n" not in result
        assert "\r" not in result
    
    def test_text_truncation(self):
        """Test text truncation for long descriptions."""
        preparer = ActivityTextPreparer()
        
        long_text = "A" * 500
        result = preparer._clean_text(long_text, max_length=100)
        
        assert len(result) <= 103  # 100 + "..."
        assert result.endswith("...")
    
    def test_handle_missing_fields(self):
        """Test that missing fields are handled gracefully."""
        preparer = ActivityTextPreparer()
        
        result = preparer.prepare_single(
            activity_description="Install cables",
            short_name=None,
            sub_project_description=None,
            project_name=None,
            plan_hrs=None,
            used_hrs=None
        )
        
        assert "Activity: Install cables" in result
        # Should not crash, should just omit missing fields


class TestActivityTextPreparerDataFrame:
    """Tests for DataFrame preparation."""
    
    def test_prepare_dataframe(self, sample_activity_data):
        """Test DataFrame preparation."""
        preparer = ActivityTextPreparer()
        
        result = preparer.prepare_dataframe(sample_activity_data)
        
        # Should add prepared_text column
        assert 'prepared_text' in result.columns
        
        # Should add text_minimal column
        assert 'text_minimal' in result.columns
        
        # Should have same number of rows
        assert len(result) == len(sample_activity_data)
        
        # Check first record
        first_text = result.iloc[0]['prepared_text']
        assert "Install electrical cable trays" in first_text
        assert "Electrical Installation - Onshore" in first_text
        assert "Johan Sverdrup P2" in first_text
    
    def test_column_detection(self, sample_activity_data):
        """Test automatic column detection."""
        preparer = ActivityTextPreparer()
        
        col_map = preparer._detect_columns(sample_activity_data)
        
        assert col_map.get('activity_description') == 'activity_description'
        assert col_map.get('short_name') == 'short_name'
        assert col_map.get('sub_project_description') == 'sub_project_description'
        assert col_map.get('project_name') == 'project_name'
        assert col_map.get('plan_hrs') == 'plan_hrs'
        assert col_map.get('used_hrs') == 'used_hrs'
    
    def test_handles_alternative_column_names(self):
        """Test handling of alternative column names."""
        df = pd.DataFrame({
            'description': ['Test activity'],  # Alternative name
            'shortname': ['T-001'],  # Alternative name
            'planned_hours': [100.0],  # Alternative name
            'actual_hours': [50.0]  # Alternative name
        })
        
        preparer = ActivityTextPreparer()
        col_map = preparer._detect_columns(df)
        
        assert col_map.get('activity_description') == 'description'
        assert col_map.get('short_name') == 'shortname'
        assert col_map.get('plan_hrs') == 'planned_hours'
        assert col_map.get('used_hrs') == 'actual_hours'


class TestTemplateBasedPreparer:
    """Tests for TemplateBasedPreparer."""
    
    def test_general_template(self):
        """Test general template."""
        preparer = TemplateBasedPreparer('general')
        
        result = preparer.prepare_single(
            activity_description="Install cable trays",
            short_name="E-001",
            sub_project_description="Electrical Installation",
            project_name="Project Alpha"
        )
        
        assert "Classify this oil & gas activity" in result
        assert "Install cable trays" in result
        assert "Task code: E-001" in result
        assert "Work package: Electrical Installation" in result
        assert "Project: Project Alpha" in result
    
    def test_phase_focused_template(self):
        """Test phase-focused template."""
        preparer = TemplateBasedPreparer('phase_focused')
        
        result = preparer.prepare_single(
            activity_description="Weld pipe spool",
            sub_project_description="Piping Fabrication",
            project_name="Offshore Platform"
        )
        
        assert "Determine the project phase" in result
        assert "Weld pipe spool" in result
    
    def test_discipline_focused_template(self):
        """Test discipline-focused template."""
        preparer = TemplateBasedPreparer('discipline_focused')
        
        result = preparer.prepare_single(
            activity_description="Install electrical panels",
            short_name="E-PANEL-001",
            sub_project_description="Electrical Systems"
        )
        
        assert "Identify the engineering discipline" in result
        assert "Install electrical panels" in result
    
    def test_location_focused_template(self):
        """Test location-focused template."""
        preparer = TemplateBasedPreparer('location_focused')
        
        result = preparer.prepare_single(
            activity_description="Hook-up cables at platform",
            sub_project_description="Offshore Installation",
            project_name="North Sea Platform"
        )
        
        assert "onshore or offshore" in result
        assert "Hook-up cables at platform" in result


class TestCreateTrainingText:
    """Tests for create_training_text convenience function."""
    
    def test_rich_strategy(self, sample_activity_data):
        """Test rich strategy."""
        result = create_training_text(sample_activity_data, strategy='rich')
        
        assert 'prepared_text' in result.columns
        
        # Should have full context
        first_text = result.iloc[0]['prepared_text']
        assert "[SEP]" in first_text
        assert "Sub-project:" in first_text
    
    def test_minimal_strategy(self, sample_activity_data):
        """Test minimal strategy."""
        result = create_training_text(sample_activity_data, strategy='minimal')
        
        assert 'prepared_text' in result.columns
        
        # Should be minimal (just description)
        first_text = result.iloc[0]['prepared_text']
        # Minimal includes just the Activity:
        assert "Activity:" in first_text
        assert "Sub-project:" not in first_text
        assert "Project:" not in first_text
    
    def test_template_strategy(self, sample_activity_data):
        """Test template strategy."""
        result = create_training_text(sample_activity_data, strategy='template')
        
        assert 'prepared_text' in result.columns
        
        # Should use template format
        first_text = result.iloc[0]['prepared_text']
        assert "Classify this oil & gas activity" in first_text


class TestRealWorldScenarios:
    """Tests for real-world scenarios."""
    
    def test_norwegian_english_mixed_content(self):
        """Test handling of mixed Norwegian/English content."""
        preparer = ActivityTextPreparer()
        
        result = preparer.prepare_single(
            activity_description="Installere elektriske kabelbroer på dekk 3",  # Norwegian
            short_name="E-KABEL-001",
            sub_project_description="Elektrisk installasjon - Landanlegg",
            project_name="Johan Sverdrup Fase 2"
        )
        
        # Should handle Norwegian text without issues
        assert "Installere elektriske kabelbroer" in result
        assert "Elektrisk installasjon" in result
    
    def test_special_characters(self):
        """Test handling of special characters."""
        preparer = ActivityTextPreparer()
        
        result = preparer.prepare_single(
            activity_description="Install 6\" pipe (SS304L) @ location B-12/A",
            short_name="P-6\"-SS304L-001",
            sub_project_description="Piping - 6\" SS Lines"
        )
        
        # Should handle special chars
        assert "6\"" in result or "6" in result
        assert "SS304L" in result
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        preparer = ActivityTextPreparer()
        
        empty_df = pd.DataFrame(columns=[
            'activity_description', 'short_name', 'sub_project_description'
        ])
        
        result = preparer.prepare_dataframe(empty_df)
        
        assert len(result) == 0
        assert 'prepared_text' in result.columns
    
    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrame."""
        preparer = ActivityTextPreparer()
        
        single_df = pd.DataFrame({
            'activity_description': ['Single activity test'],
            'short_name': ['TEST-001']
        })
        
        result = preparer.prepare_dataframe(single_df)
        
        assert len(result) == 1
        assert 'prepared_text' in result.columns
        assert "Single activity test" in result.iloc[0]['prepared_text']


class TestTextQualityForRoBERTa:
    """Tests ensuring text is optimally formatted for RoBERTa."""
    
    def test_separator_tokens_are_valid(self):
        """Test that separator tokens are RoBERTa-compatible."""
        preparer = ActivityTextPreparer()
        
        result = preparer.prepare_single(
            activity_description="Test activity",
            sub_project_description="Test sub-project"
        )
        
        # [SEP] is a valid token for RoBERTa
        assert " [SEP] " in result
    
    def test_text_length_reasonable(self, sample_activity_data):
        """Test that prepared text has reasonable length."""
        preparer = ActivityTextPreparer()
        result = preparer.prepare_dataframe(sample_activity_data)
        
        for text in result['prepared_text']:
            # Should be under typical RoBERTa max length (512 tokens ≈ 2000 chars)
            assert len(text) < 2000
            
            # Should have meaningful content
            assert len(text) > 20
    
    def test_no_excessive_special_chars(self):
        """Test that special characters don't dominate."""
        preparer = ActivityTextPreparer()
        
        result = preparer.prepare_single(
            activity_description="Normal activity description here",
            short_name="CODE-123",
            sub_project_description="Normal sub-project"
        )
        
        # Mostly alphanumeric + spaces + some punctuation
        alpha_count = sum(1 for c in result if c.isalnum() or c.isspace())
        total_count = len(result)
        
        # At least 70% should be alphanumeric/space
        assert alpha_count / total_count > 0.7
    
    def test_whitespace_normalized(self):
        """Test that whitespace is properly normalized."""
        preparer = ActivityTextPreparer()
        
        result = preparer.prepare_single(
            activity_description="Text  with   multiple    spaces",
        )
        
        # No consecutive spaces (except around [SEP])
        assert "Text with multiple spaces" in result


class TestIntegrationWithLabelMapper:
    """Tests for integration with label mapper pipeline."""
    
    def test_prepared_text_compatible_with_training(self, sample_activity_data):
        """Test that prepared text works in training pipeline."""
        # Prepare text
        prepared = create_training_text(sample_activity_data, strategy='rich')
        
        # Verify columns exist for training
        assert 'prepared_text' in prepared.columns
        
        # Each prepared text should be a non-empty string
        for text in prepared['prepared_text']:
            assert isinstance(text, str)
            assert len(text) > 0
