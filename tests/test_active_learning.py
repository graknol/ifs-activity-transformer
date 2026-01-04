"""
Unit tests for active learning module (app/active_learning.py).

Tests the active learning service with uncertainty sampling.
"""
import pytest
import numpy as np
from app.active_learning import ActiveLearningService, SamplingStrategy, MultiLabelActiveLearning


class TestActiveLearningService:
    """Test ActiveLearningService class."""
    
    @pytest.fixture
    def al_service(self):
        """Create an active learning service for testing."""
        return ActiveLearningService(confidence_threshold=0.7, sanity_check_ratio=0.1)
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample prediction probabilities."""
        return np.array([
            [0.9, 0.05, 0.05],  # High confidence
            [0.4, 0.3, 0.3],    # Uncertain (low max)
            [0.5, 0.48, 0.02],  # Uncertain (small margin)
            [0.7, 0.2, 0.1],    # Moderate confidence
            [0.33, 0.34, 0.33], # Very uncertain (high entropy)
            [0.95, 0.03, 0.02]  # Very high confidence
        ])
    
    @pytest.fixture
    def sample_ids(self):
        """Create sample IDs."""
        return ['ACT001', 'ACT002', 'ACT003', 'ACT004', 'ACT005', 'ACT006']
    
    def test_least_confidence_strategy(self, al_service, sample_predictions, sample_ids):
        """Test least confidence sampling strategy."""
        results = al_service.select_samples_for_annotation(
            sample_predictions,
            sample_ids,
            n_samples=4,
            strategy=SamplingStrategy.LEAST_CONFIDENCE
        )
        
        assert len(results) == 4
        assert all('uncertainty_score' in r for r in results)
        assert all('sample_id' in r for r in results)
        assert all('max_confidence' in r for r in results)
    
    def test_margin_sampling_strategy(self, al_service, sample_predictions, sample_ids):
        """Test margin sampling strategy."""
        results = al_service.select_samples_for_annotation(
            sample_predictions,
            sample_ids,
            n_samples=4,
            strategy=SamplingStrategy.MARGIN_SAMPLING
        )
        
        assert len(results) == 4
        # Check that results have required fields
        for result in results:
            assert 'uncertainty_score' in result
            assert 'sample_id' in result
    
    def test_entropy_strategy(self, al_service, sample_predictions, sample_ids):
        """Test entropy sampling strategy."""
        results = al_service.select_samples_for_annotation(
            sample_predictions,
            sample_ids,
            n_samples=4,
            strategy=SamplingStrategy.ENTROPY
        )
        
        assert len(results) == 4
        # Entropy should be higher for more uncertain predictions
        for result in results:
            assert result['uncertainty_score'] >= 0
    
    def test_random_strategy(self, al_service, sample_predictions, sample_ids):
        """Test random sampling strategy."""
        results = al_service.select_samples_for_annotation(
            sample_predictions,
            sample_ids,
            n_samples=4,
            strategy=SamplingStrategy.RANDOM
        )
        
        assert len(results) == 4
    
    def test_sanity_check_ratio(self, al_service, sample_predictions, sample_ids):
        """Test that sanity check ratio is respected."""
        al_service.sanity_check_ratio = 0.2  # 20% confident samples
        
        results = al_service.select_samples_for_annotation(
            sample_predictions,
            sample_ids,
            n_samples=10,
            strategy=SamplingStrategy.LEAST_CONFIDENCE
        )
        
        # Should select 8 uncertain + 2 confident = 10 total
        # (exact breakdown depends on shuffling, but total should be 10)
        assert len(results) == min(10, len(sample_ids))
    
    def test_prediction_probabilities(self, al_service, sample_predictions, sample_ids):
        """Test that prediction probabilities are included in results."""
        results = al_service.select_samples_for_annotation(
            sample_predictions,
            sample_ids,
            n_samples=2,
            strategy=SamplingStrategy.LEAST_CONFIDENCE
        )
        
        for result in results:
            assert 'probabilities' in result
            assert len(result['probabilities']) == 3  # 3 classes
            assert sum(result['probabilities']) == pytest.approx(1.0, abs=0.01)
    
    def test_predicted_class(self, al_service, sample_predictions, sample_ids):
        """Test that predicted class is correct."""
        results = al_service.select_samples_for_annotation(
            sample_predictions,
            sample_ids,
            n_samples=2,
            strategy=SamplingStrategy.LEAST_CONFIDENCE
        )
        
        for result in results:
            assert 'predicted_class' in result
            idx = result['index']
            expected_class = np.argmax(sample_predictions[idx])
            assert result['predicted_class'] == expected_class
    
    def test_uncertainty_flag(self, al_service, sample_predictions, sample_ids):
        """Test that is_uncertain flag is set."""
        results = al_service.select_samples_for_annotation(
            sample_predictions,
            sample_ids,
            n_samples=6,
            strategy=SamplingStrategy.LEAST_CONFIDENCE
        )
        
        for result in results:
            assert 'is_uncertain' in result
            assert isinstance(result['is_uncertain'], bool)


class TestMultiLabelActiveLearning:
    """Test MultiLabelActiveLearning class."""
    
    @pytest.fixture
    def ml_al_service(self):
        """Create a multi-label active learning service."""
        return MultiLabelActiveLearning(confidence_threshold=0.5, sanity_check_ratio=0.1)
    
    @pytest.fixture
    def multilabel_predictions(self):
        """Create sample multi-label predictions."""
        return np.array([
            [0.9, 0.8, 0.1, 0.05],  # High confidence, multiple labels
            [0.45, 0.52, 0.48, 0.49],  # Very uncertain
            [0.7, 0.2, 0.15, 0.1],  # Moderate confidence
            [0.51, 0.49, 0.48, 0.52]   # All near threshold
        ])
    
    @pytest.fixture
    def multilabel_ids(self):
        """Create sample IDs for multi-label."""
        return ['ML001', 'ML002', 'ML003', 'ML004']
    
    def test_multilabel_uncertainty_calculation(self, ml_al_service, multilabel_predictions):
        """Test uncertainty calculation for multi-label predictions."""
        uncertainty = ml_al_service._calculate_multilabel_uncertainty(multilabel_predictions)
        
        assert len(uncertainty) == len(multilabel_predictions)
        assert all(u >= 0 for u in uncertainty)
        # Most uncertain should be predictions near 0.5
        assert uncertainty[1] > uncertainty[0]  # All near 0.5 vs clear predictions
    
    def test_select_multilabel_samples(self, ml_al_service, multilabel_predictions, multilabel_ids):
        """Test selecting multi-label samples for annotation."""
        results = ml_al_service.select_multilabel_samples(
            multilabel_predictions,
            multilabel_ids,
            n_samples=3
        )
        
        assert len(results) == 3
        for result in results:
            assert 'sample_id' in result
            assert 'uncertainty_score' in result
            assert 'predicted_labels' in result
            assert 'label_confidences' in result
    
    def test_predicted_labels_above_threshold(self, ml_al_service, multilabel_predictions, multilabel_ids):
        """Test that predicted labels are above threshold."""
        ml_al_service.confidence_threshold = 0.6
        
        results = ml_al_service.select_multilabel_samples(
            multilabel_predictions,
            multilabel_ids,
            n_samples=2
        )
        
        for result in results:
            predicted_labels = result['predicted_labels']
            confidences = result['label_confidences']
            # All predicted labels should have confidence > 0.6
            for label_idx in predicted_labels:
                assert confidences[label_idx] > 0.6
    
    def test_label_confidences(self, ml_al_service, multilabel_predictions, multilabel_ids):
        """Test that label confidences are correctly returned."""
        results = ml_al_service.select_multilabel_samples(
            multilabel_predictions,
            multilabel_ids,
            n_samples=2
        )
        
        for result in results:
            confidences = result['label_confidences']
            assert len(confidences) == 4  # 4 labels
            assert all(0 <= c <= 1 for c in confidences)
    
    def test_multilabel_sanity_check_ratio(self, ml_al_service, multilabel_predictions, multilabel_ids):
        """Test sanity check ratio in multi-label sampling."""
        ml_al_service.sanity_check_ratio = 0.25  # 25% confident samples
        
        results = ml_al_service.select_multilabel_samples(
            multilabel_predictions,
            multilabel_ids,
            n_samples=4
        )
        
        # Should return all 4 samples (3 uncertain + 1 confident)
        assert len(results) == 4
    
    def test_empty_predictions(self, ml_al_service):
        """Test handling empty predictions."""
        empty_predictions = np.array([]).reshape(0, 4)
        empty_ids = []
        
        results = ml_al_service.select_multilabel_samples(
            empty_predictions,
            empty_ids,
            n_samples=5
        )
        
        assert len(results) == 0
    
    def test_fewer_samples_than_requested(self, ml_al_service, multilabel_predictions, multilabel_ids):
        """Test requesting more samples than available."""
        results = ml_al_service.select_multilabel_samples(
            multilabel_predictions,
            multilabel_ids,
            n_samples=100  # More than available
        )
        
        # Should return all available samples
        assert len(results) == len(multilabel_ids)
