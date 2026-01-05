"""
Incremental training service for active learning workflow.

This module manages the train-annotate-retrain loop:
1. Train on initial bootstrap data
2. User annotates uncertain samples
3. Retrain model with new annotations
4. Repeat - model improves with each iteration
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingSession:
    """Represents a single training session."""
    session_id: str
    timestamp: str
    n_samples: int
    n_new_samples: int
    epochs: int
    metrics: Dict
    model_path: str
    

class IncrementalTrainingService:
    """
    Manages incremental training for active learning.
    
    Key features:
    - Tracks training history
    - Manages annotation batches
    - Triggers retraining after N annotations
    - Provides training/annotation statistics
    """
    
    def __init__(
        self,
        data_dir: str = 'data',
        models_dir: str = 'models',
        retrain_threshold: int = 20,
        min_samples_for_training: int = 50
    ):
        """
        Initialize the incremental training service.
        
        Args:
            data_dir: Directory for data files
            models_dir: Directory for model checkpoints
            retrain_threshold: Number of new annotations before auto-retrain
            min_samples_for_training: Minimum samples needed to start training
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.retrain_threshold = retrain_threshold
        self.min_samples_for_training = min_samples_for_training
        
        # Paths
        self.annotations_path = os.path.join(data_dir, 'annotations.csv')
        self.training_history_path = os.path.join(data_dir, 'training_history.json')
        self.pending_annotations_path = os.path.join(data_dir, 'pending_annotations.json')
        
        # State
        self._pending_count = 0
        self._load_state()
    
    def _load_state(self):
        """Load persistent state."""
        if os.path.exists(self.pending_annotations_path):
            with open(self.pending_annotations_path, 'r') as f:
                state = json.load(f)
                self._pending_count = state.get('pending_count', 0)
    
    def _save_state(self):
        """Save persistent state."""
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.pending_annotations_path, 'w') as f:
            json.dump({
                'pending_count': self._pending_count,
                'last_updated': datetime.now().isoformat()
            }, f)
    
    def get_training_history(self) -> List[Dict]:
        """Get history of all training sessions."""
        if not os.path.exists(self.training_history_path):
            return []
        
        with open(self.training_history_path, 'r') as f:
            return json.load(f)
    
    def _save_training_session(self, session: TrainingSession):
        """Save a training session to history."""
        history = self.get_training_history()
        history.append(asdict(session))
        
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.training_history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def get_annotation_stats(self) -> Dict:
        """Get statistics about annotations."""
        if not os.path.exists(self.annotations_path):
            return {
                'total_annotations': 0,
                'pending_for_training': self._pending_count,
                'retrain_threshold': self.retrain_threshold,
                'ready_for_retrain': False,
                'ready_for_initial_train': False
            }
        
        df = pd.read_csv(self.annotations_path)
        total = len(df)
        
        return {
            'total_annotations': total,
            'pending_for_training': self._pending_count,
            'retrain_threshold': self.retrain_threshold,
            'ready_for_retrain': self._pending_count >= self.retrain_threshold,
            'ready_for_initial_train': total >= self.min_samples_for_training,
            'min_samples_needed': self.min_samples_for_training
        }
    
    def record_annotation(self, annotation: Dict) -> Dict:
        """
        Record a new annotation and check if retraining is needed.
        
        Args:
            annotation: Dict with activity_id, labels, etc.
            
        Returns:
            Status dict including whether retrain is triggered
        """
        self._pending_count += 1
        self._save_state()
        
        stats = self.get_annotation_stats()
        
        result = {
            'recorded': True,
            'pending_count': self._pending_count,
            'should_retrain': stats['ready_for_retrain'],
            'message': f"Annotation recorded. {self._pending_count}/{self.retrain_threshold} until next retrain."
        }
        
        if stats['ready_for_retrain']:
            result['message'] = f"Retrain threshold reached! {self._pending_count} new annotations ready for training."
        
        return result
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare training data from annotations.
        
        Combines:
        - Bootstrap annotations (if any)
        - User annotations from active learning
        
        Returns:
            Tuple of (training DataFrame, metadata)
        """
        training_samples = []
        sources = {}
        
        # Load bootstrap data if exists
        bootstrap_path = os.path.join(self.data_dir, 'bootstrap_annotated.csv')
        if os.path.exists(bootstrap_path):
            bootstrap_df = pd.read_csv(bootstrap_path)
            # Filter to rows with labels
            labeled = bootstrap_df[
                (bootstrap_df.get('PHASE_LABEL', '').notna()) | 
                (bootstrap_df.get('DISCIPLINE_LABEL', '').notna()) |
                (bootstrap_df.get('categories', '').notna())
            ]
            if len(labeled) > 0:
                training_samples.append(labeled)
                sources['bootstrap'] = len(labeled)
        
        # Load user annotations
        if os.path.exists(self.annotations_path):
            annotations_df = pd.read_csv(self.annotations_path)
            if len(annotations_df) > 0:
                training_samples.append(annotations_df)
                sources['user_annotations'] = len(annotations_df)
        
        if not training_samples:
            return pd.DataFrame(), {'error': 'No training data available'}
        
        # Combine all sources
        combined_df = pd.concat(training_samples, ignore_index=True)
        
        metadata = {
            'total_samples': len(combined_df),
            'sources': sources,
            'timestamp': datetime.now().isoformat()
        }
        
        return combined_df, metadata
    
    def trigger_training(
        self,
        classifier,
        epochs: int = None,
        early_stopping: bool = True
    ) -> TrainingSession:
        """
        Trigger a training run with current annotations.
        
        Args:
            classifier: The model classifier to train
            epochs: Number of epochs (auto-determined if None)
            early_stopping: Whether to use early stopping
            
        Returns:
            TrainingSession with results
        """
        # Prepare data
        train_df, data_meta = self.prepare_training_data()
        
        if train_df.empty:
            raise ValueError("No training data available")
        
        n_samples = len(train_df)
        
        # Determine optimal epochs based on data size
        if epochs is None:
            if n_samples < 100:
                epochs = 10
            elif n_samples < 500:
                epochs = 8
            elif n_samples < 2000:
                epochs = 5
            else:
                epochs = 3
        
        logger.info(f"Starting training with {n_samples} samples for {epochs} epochs")
        
        # Generate session ID
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.models_dir, f'model_{session_id}')
        
        # Train the model
        # This would call your actual training logic
        # For now, we'll structure the interface
        try:
            # Prepare texts and labels
            text_col = None
            for col in ['ACTIVITY_DESCRIPTION', 'description', 'text']:
                if col in train_df.columns:
                    text_col = col
                    break
            
            label_col = None
            for col in ['categories', 'DISCIPLINE_LABEL', 'label']:
                if col in train_df.columns:
                    label_col = col
                    break
            
            if text_col is None or label_col is None:
                raise ValueError(f"Missing required columns. Have: {train_df.columns.tolist()}")
            
            texts = train_df[text_col].tolist()
            labels = train_df[label_col].tolist()
            
            # Call actual training
            metrics = classifier.train(
                texts=texts,
                labels=labels,
                epochs=epochs,
                early_stopping=early_stopping,
                save_path=model_path
            )
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            metrics = {'error': str(e), 'status': 'failed'}
        
        # Create session record
        history = self.get_training_history()
        n_previous = sum(s.get('n_samples', 0) for s in history)
        
        session = TrainingSession(
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            n_samples=n_samples,
            n_new_samples=self._pending_count,
            epochs=epochs,
            metrics=metrics,
            model_path=model_path
        )
        
        # Save session and reset pending count
        self._save_training_session(session)
        self._pending_count = 0
        self._save_state()
        
        logger.info(f"Training complete. Session: {session_id}, Metrics: {metrics}")
        
        return session
    
    def get_status(self) -> Dict:
        """Get overall status of the incremental training system."""
        stats = self.get_annotation_stats()
        history = self.get_training_history()
        
        last_training = history[-1] if history else None
        
        return {
            'annotations': stats,
            'training_sessions': len(history),
            'last_training': last_training,
            'model_available': last_training is not None,
            'recommendations': self._get_recommendations(stats, history)
        }
    
    def _get_recommendations(self, stats: Dict, history: List) -> List[str]:
        """Generate recommendations for the user."""
        recs = []
        
        if stats['total_annotations'] == 0:
            recs.append("Start by annotating the bootstrap sample or loading pre-annotated data.")
        elif not stats['ready_for_initial_train'] and len(history) == 0:
            remaining = stats['min_samples_needed'] - stats['total_annotations']
            recs.append(f"Annotate {remaining} more samples to enable initial training.")
        elif stats['ready_for_retrain']:
            recs.append(f"You have {stats['pending_for_training']} new annotations. Consider retraining the model.")
        elif stats['pending_for_training'] > 0:
            remaining = stats['retrain_threshold'] - stats['pending_for_training']
            recs.append(f"Annotate {remaining} more samples to trigger automatic retraining.")
        
        if len(history) > 0:
            last = history[-1]
            if last.get('metrics', {}).get('accuracy', 0) < 0.8:
                recs.append("Model accuracy is below 80%. More diverse annotations may help.")
        
        return recs


# Singleton instance for app-wide use
_service_instance = None

def get_incremental_training_service() -> IncrementalTrainingService:
    """Get or create the singleton service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = IncrementalTrainingService()
    return _service_instance
