"""
Active learning service for uncertainty-based sample selection.

This module implements active learning strategies to prioritize samples
that will most improve model performance when labeled.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum
from datetime import datetime
import random


class SamplingStrategy(Enum):
    """Strategies for selecting samples for annotation."""
    LEAST_CONFIDENCE = "least_confidence"
    MARGIN_SAMPLING = "margin_sampling"
    ENTROPY = "entropy"
    RANDOM = "random"


class ActiveLearningService:
    """
    Service for active learning sample selection.
    
    Implements uncertainty sampling strategies to identify the most valuable
    samples for human annotation, maximizing model improvement per labeled sample.
    """
    
    def __init__(self, confidence_threshold: float = 0.7, sanity_check_ratio: float = 0.1):
        """
        Initialize active learning service.
        
        Args:
            confidence_threshold: Threshold below which samples are considered uncertain
            sanity_check_ratio: Ratio of high-confidence samples to include for validation
        """
        self.confidence_threshold = confidence_threshold
        self.sanity_check_ratio = sanity_check_ratio
    
    def select_samples_for_annotation(
        self,
        predictions: np.ndarray,
        sample_ids: List[str],
        n_samples: int = 20,
        strategy: SamplingStrategy = SamplingStrategy.LEAST_CONFIDENCE,
        early_start_dates: Optional[List] = None,
        recency_weight: float = 0.3
    ) -> List[Dict]:
        """
        Select samples for human annotation based on model uncertainty.
        
        Combines uncertainty sampling with recency preference to prioritize
        newer activities (based on early_start date).
        
        Args:
            predictions: Model prediction probabilities (n_samples, n_classes)
            sample_ids: Identifiers for each sample
            n_samples: Number of samples to select
            strategy: Sampling strategy to use
            early_start_dates: Optional list of early_start dates for recency weighting
            recency_weight: Weight for recency in combined score (0-1, default 0.3)
                           0 = pure uncertainty, 1 = pure recency
            
        Returns:
            List of dictionaries with sample info and uncertainty scores
        """
        # Calculate uncertainty scores based on strategy
        if strategy == SamplingStrategy.LEAST_CONFIDENCE:
            uncertainty_scores = self._least_confidence(predictions)
        elif strategy == SamplingStrategy.MARGIN_SAMPLING:
            uncertainty_scores = self._margin_sampling(predictions)
        elif strategy == SamplingStrategy.ENTROPY:
            uncertainty_scores = self._entropy_sampling(predictions)
        else:
            uncertainty_scores = np.random.random(len(predictions))
        
        # Calculate combined scores with recency
        combined_scores = self._combine_with_recency(
            uncertainty_scores, early_start_dates, recency_weight
        )
        
        # Determine how many uncertain vs confident samples to show
        # Cap to available samples
        n_samples = min(n_samples, len(predictions))
        n_uncertain = int(n_samples * (1 - self.sanity_check_ratio))
        n_confident = n_samples - n_uncertain
        
        # Get indices sorted by combined score (high to low)
        sorted_indices = np.argsort(combined_scores)[::-1]
        
        # Select most uncertain samples (cap to available)
        uncertain_indices = sorted_indices[:min(n_uncertain, len(predictions))]
        
        # Select some confident samples for sanity checks (low uncertainty)
        remaining = len(predictions) - len(uncertain_indices)
        n_confident = min(n_confident, remaining)
        confident_indices = sorted_indices[-n_confident:] if n_confident > 0 else []
        
        # Combine and shuffle so user doesn't know which is which
        selected_indices = np.concatenate([uncertain_indices, confident_indices])
        np.random.shuffle(selected_indices)
        
        # Build result list
        results = []
        for idx in selected_indices:
            max_conf = np.max(predictions[idx])
            pred_class = np.argmax(predictions[idx])
            
            result = {
                'sample_id': sample_ids[idx],
                'index': int(idx),
                'uncertainty_score': float(uncertainty_scores[idx]),
                'combined_score': float(combined_scores[idx]),
                'max_confidence': float(max_conf),
                'predicted_class': int(pred_class),
                'is_uncertain': bool(uncertainty_scores[idx] > np.median(uncertainty_scores)),
                'probabilities': predictions[idx].tolist()
            }
            
            # Include early_start if available
            if early_start_dates is not None and idx < len(early_start_dates):
                result['early_start'] = early_start_dates[idx]
            
            results.append(result)
        
        return results
    
    def _least_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """
        Least confidence sampling: select samples with lowest max probability.
        
        Args:
            predictions: Prediction probabilities
            
        Returns:
            Uncertainty scores (higher = more uncertain)
        """
        max_probs = np.max(predictions, axis=1)
        return 1 - max_probs
    
    def _margin_sampling(self, predictions: np.ndarray) -> np.ndarray:
        """
        Margin sampling: select samples with smallest margin between top 2 predictions.
        
        Args:
            predictions: Prediction probabilities
            
        Returns:
            Uncertainty scores (higher = more uncertain)
        """
        # Sort predictions for each sample
        sorted_preds = np.sort(predictions, axis=1)
        # Margin is difference between top 2
        margins = sorted_preds[:, -1] - sorted_preds[:, -2]
        # Smaller margin = more uncertain
        return 1 - margins
    
    def _entropy_sampling(self, predictions: np.ndarray) -> np.ndarray:
        """
        Entropy sampling: select samples with highest prediction entropy.
        
        Args:
            predictions: Prediction probabilities
            
        Returns:
            Uncertainty scores (higher = more uncertain)
        """
        # Calculate entropy: -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -np.sum(predictions * np.log(predictions + epsilon), axis=1)
        # Normalize by max possible entropy
        max_entropy = np.log(predictions.shape[1])
        return entropy / max_entropy
    
    def _combine_with_recency(
        self,
        uncertainty_scores: np.ndarray,
        early_start_dates: Optional[List],
        recency_weight: float = 0.3
    ) -> np.ndarray:
        """
        Combine uncertainty scores with recency preference.
        
        Newer activities (higher early_start dates) get a boost to their score.
        
        Args:
            uncertainty_scores: Base uncertainty scores
            early_start_dates: List of early_start dates (can be datetime, Timestamp, or string)
            recency_weight: Weight for recency (0-1)
            
        Returns:
            Combined scores (higher = more likely to be selected)
        """
        if early_start_dates is None or len(early_start_dates) == 0:
            return uncertainty_scores
        
        # Convert dates to timestamps for comparison
        timestamps = []
        for date in early_start_dates:
            if date is None or (isinstance(date, float) and np.isnan(date)):
                timestamps.append(None)
            elif isinstance(date, (datetime, pd.Timestamp)):
                timestamps.append(date.timestamp())
            elif isinstance(date, str):
                try:
                    parsed = pd.to_datetime(date)
                    timestamps.append(parsed.timestamp())
                except:
                    timestamps.append(None)
            else:
                timestamps.append(None)
        
        # Handle missing dates - use median for missing values
        valid_timestamps = [t for t in timestamps if t is not None]
        if not valid_timestamps:
            return uncertainty_scores
        
        median_ts = np.median(valid_timestamps)
        timestamps = [t if t is not None else median_ts for t in timestamps]
        timestamps = np.array(timestamps)
        
        # Normalize timestamps to [0, 1] range
        min_ts = timestamps.min()
        max_ts = timestamps.max()
        
        if max_ts > min_ts:
            recency_scores = (timestamps - min_ts) / (max_ts - min_ts)
        else:
            recency_scores = np.ones_like(timestamps) * 0.5
        
        # Combine: weighted average of uncertainty and recency
        combined = (1 - recency_weight) * uncertainty_scores + recency_weight * recency_scores
        
        return combined
    
    def get_annotation_statistics(self, annotations: pd.DataFrame) -> Dict:
        """
        Calculate statistics about annotation progress.
        
        Args:
            annotations: DataFrame with annotation data
            
        Returns:
            Dictionary with statistics
        """
        total = len(annotations)
        labeled = annotations['label'].notna().sum() if 'label' in annotations else 0
        
        stats = {
            'total_samples': total,
            'labeled_samples': labeled,
            'unlabeled_samples': total - labeled,
            'completion_rate': (labeled / total * 100) if total > 0 else 0
        }
        
        if 'uncertainty_score' in annotations and labeled > 0:
            labeled_data = annotations[annotations['label'].notna()]
            stats['avg_uncertainty_labeled'] = float(labeled_data['uncertainty_score'].mean())
            stats['avg_confidence_labeled'] = float(1 - stats['avg_uncertainty_labeled'])
        
        return stats
    
    def prioritize_annotation_queue(
        self,
        unlabeled_data: pd.DataFrame,
        predictions: np.ndarray,
        batch_size: int = 50
    ) -> pd.DataFrame:
        """
        Create a prioritized queue of samples for annotation.
        
        Args:
            unlabeled_data: DataFrame of unlabeled samples
            predictions: Model predictions for these samples
            batch_size: Size of annotation batch
            
        Returns:
            Prioritized DataFrame with top samples to annotate
        """
        # Calculate uncertainty scores
        uncertainty_scores = self._least_confidence(predictions)
        
        # Add uncertainty to dataframe
        unlabeled_data = unlabeled_data.copy()
        unlabeled_data['uncertainty_score'] = uncertainty_scores
        unlabeled_data['max_confidence'] = np.max(predictions, axis=1)
        
        # Sort by uncertainty (descending)
        prioritized = unlabeled_data.sort_values('uncertainty_score', ascending=False)
        
        return prioritized.head(batch_size)


class MultiLabelActiveLearning:
    """
    Active learning for multi-label classification tasks.
    
    Handles uncertainty sampling when each sample can have multiple labels.
    """
    
    def __init__(self, confidence_threshold: float = 0.5, sanity_check_ratio: float = 0.1):
        """
        Initialize multi-label active learning.
        
        Args:
            confidence_threshold: Threshold for binary classification per label
            sanity_check_ratio: Ratio of confident samples to include for validation
        """
        self.confidence_threshold = confidence_threshold
        self.sanity_check_ratio = sanity_check_ratio
    
    def calculate_multilabel_uncertainty(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Calculate uncertainty for multi-label predictions.
        
        For multi-label, uncertainty is based on how many labels are
        near the decision boundary (close to 0.5).
        
        Args:
            predictions: Binary prediction probabilities (n_samples, n_labels)
            
        Returns:
            Uncertainty scores per sample
        """
        # Distance from decision boundary (0.5) for each label
        distances = np.abs(predictions - 0.5)
        
        # Average distance across all labels (lower = more uncertain)
        avg_distance = np.mean(distances, axis=1)
        
        # Convert to uncertainty score (higher = more uncertain)
        uncertainty = 1 - (avg_distance * 2)  # Scale to [0, 1]
        
        return uncertainty
    
    def select_multilabel_samples(
        self,
        predictions: np.ndarray,
        sample_ids: List[str],
        n_samples: int = 20,
        sanity_check_ratio: float = 0.1,
        early_start_dates: Optional[List] = None,
        recency_weight: float = 0.3
    ) -> List[Dict]:
        """
        Select samples for multi-label annotation.
        
        Combines uncertainty sampling with recency preference to prioritize
        newer activities (based on early_start date).
        
        Args:
            predictions: Multi-label predictions (n_samples, n_labels)
            sample_ids: Sample identifiers
            n_samples: Number of samples to select
            sanity_check_ratio: Ratio of confident samples
            early_start_dates: Optional list of early_start dates for recency weighting
            recency_weight: Weight for recency in combined score (0-1, default 0.3)
                           0 = pure uncertainty, 1 = pure recency
            
        Returns:
            List of selected samples with metadata
        """
        # Ensure we don't request more samples than available
        n_samples = min(n_samples, len(predictions))
        
        uncertainty_scores = self.calculate_multilabel_uncertainty(predictions)
        
        # Calculate combined scores with recency
        combined_scores = self._combine_with_recency(
            uncertainty_scores, early_start_dates, recency_weight
        )
        
        # Determine split
        n_uncertain = int(n_samples * (1 - sanity_check_ratio))
        n_confident = n_samples - n_uncertain
        
        # Sort by combined score (uncertainty + recency)
        sorted_indices = np.argsort(combined_scores)[::-1]
        
        # Select samples
        uncertain_indices = sorted_indices[:n_uncertain]
        confident_indices = sorted_indices[-n_confident:] if n_confident > 0 else []
        
        selected_indices = np.concatenate([uncertain_indices, confident_indices])
        np.random.shuffle(selected_indices)
        
        # Build results
        results = []
        median_uncertainty = np.median(uncertainty_scores)
        
        for idx in selected_indices:
            probs = predictions[idx].tolist()
            result = {
                'sample_id': sample_ids[idx],
                'index': int(idx),
                'uncertainty_score': float(uncertainty_scores[idx]),
                'combined_score': float(combined_scores[idx]),
                'predictions': probs,
                'label_confidences': probs,
                'predicted_labels': np.where(predictions[idx] >= self.confidence_threshold)[0].tolist(),
                'is_uncertain': bool(uncertainty_scores[idx] > median_uncertainty)
            }
            
            # Include early_start if available
            if early_start_dates is not None and idx < len(early_start_dates):
                result['early_start'] = early_start_dates[idx]
            
            results.append(result)
        
        return results
    
    def _combine_with_recency(
        self,
        uncertainty_scores: np.ndarray,
        early_start_dates: Optional[List],
        recency_weight: float = 0.3
    ) -> np.ndarray:
        """
        Combine uncertainty scores with recency preference.
        
        Newer activities (higher early_start dates) get a boost to their score.
        
        Args:
            uncertainty_scores: Base uncertainty scores
            early_start_dates: List of early_start dates (can be datetime, Timestamp, or string)
            recency_weight: Weight for recency (0-1)
            
        Returns:
            Combined scores (higher = more likely to be selected)
        """
        if early_start_dates is None or len(early_start_dates) == 0:
            return uncertainty_scores
        
        # Convert dates to timestamps for comparison
        timestamps = []
        for date in early_start_dates:
            if date is None or (isinstance(date, float) and np.isnan(date)):
                timestamps.append(None)
            elif isinstance(date, (datetime, pd.Timestamp)):
                timestamps.append(date.timestamp())
            elif isinstance(date, str):
                try:
                    parsed = pd.to_datetime(date)
                    timestamps.append(parsed.timestamp())
                except:
                    timestamps.append(None)
            else:
                timestamps.append(None)
        
        # Handle missing dates - use median for missing values
        valid_timestamps = [t for t in timestamps if t is not None]
        if not valid_timestamps:
            return uncertainty_scores
        
        median_ts = np.median(valid_timestamps)
        timestamps = [t if t is not None else median_ts for t in timestamps]
        timestamps = np.array(timestamps)
        
        # Normalize timestamps to [0, 1] range
        min_ts = timestamps.min()
        max_ts = timestamps.max()
        
        if max_ts > min_ts:
            recency_scores = (timestamps - min_ts) / (max_ts - min_ts)
        else:
            recency_scores = np.ones_like(timestamps) * 0.5
        
        # Combine: weighted average of uncertainty and recency
        combined = (1 - recency_weight) * uncertainty_scores + recency_weight * recency_scores
        
        return combined
