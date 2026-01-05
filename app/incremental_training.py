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
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

logger = logging.getLogger(__name__)


class TextClassificationDataset(Dataset):
    """PyTorch Dataset for text classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


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
                'ready_for_initial_train': False,
                'min_samples_needed': self.min_samples_for_training
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
            # Filter to rows with labels - check if columns exist first
            label_mask = pd.Series([False] * len(bootstrap_df))
            for col in ['PHASE_LABEL', 'DISCIPLINE_LABEL', 'categories']:
                if col in bootstrap_df.columns:
                    label_mask = label_mask | bootstrap_df[col].notna()
            
            labeled = bootstrap_df[label_mask]
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
        early_stopping: bool = True,
        hyperparameters: Dict = None
    ) -> TrainingSession:
        """
        Trigger a training run with current annotations.
        
        Args:
            classifier: The model classifier to train (ActivityClassificationPipeline or compatible)
            epochs: Number of epochs (auto-determined if None)
            early_stopping: Whether to use early stopping
            hyperparameters: Optional dict with training hyperparameters:
                - model_name: str (default 'google/bigbird-roberta-base')
                - learning_rate: float (default 2e-5)
                - batch_size: int (default 8)
                - dropout: float (default 0.1)
                - weight_decay: float (default 0.01)
                - scheduler: str (default 'linear')
                - warmup_ratio: float (default 0.06)
                - max_length: int (default 512)
                - hidden_dim: int (default 256)
                - patience: int (default 3)
                - min_delta: float (default 0.001)
            
        Returns:
            TrainingSession with results
        """
        # Default hyperparameters
        default_hyperparams = {
            'model_name': 'google/bigbird-roberta-base',
            'learning_rate': 2e-5,
            'batch_size': 8,
            'dropout': 0.1,
            'weight_decay': 0.01,
            'scheduler': 'linear',
            'warmup_ratio': 0.06,
            'max_length': 512,
            'hidden_dim': 256,
            'patience': 3,
            'min_delta': 0.001
        }
        
        # Merge with provided hyperparameters
        hp = {**default_hyperparams, **(hyperparameters or {})}
        
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
        os.makedirs(model_path, exist_ok=True)
        
        # Train the model
        try:
            # Identify text column
            text_col = None
            for col in ['ACTIVITY_DESCRIPTION', 'activity_description', 'description', 'text', 'prepared_text']:
                if col in train_df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                raise ValueError(f"No text column found. Have: {train_df.columns.tolist()}")
            
            # Identify label columns  
            label_cols = []
            for col in ['PHASE_LABEL', 'DISCIPLINE_LABEL', 'WORK_TYPE_LABEL', 'LOCATION_LABEL']:
                if col in train_df.columns:
                    label_cols.append(col)
            
            if not label_cols:
                # Fall back to single label column
                for col in ['categories', 'label', 'target']:
                    if col in train_df.columns:
                        label_cols = [col]
                        break
            
            if not label_cols:
                raise ValueError(f"No label columns found. Have: {train_df.columns.tolist()}")
            
            logger.info(f"Using text column: {text_col}, label columns: {label_cols}")
            
            # Prepare training data - create a simple text-label format
            # For multi-label, we'll concatenate labels
            if len(label_cols) > 1:
                # Multi-label case: combine labels into one target
                train_df['_combined_label'] = train_df[label_cols].apply(
                    lambda row: ' | '.join([str(v) for v in row.values if pd.notna(v) and str(v).strip()]),
                    axis=1
                )
                target_col = '_combined_label'
            else:
                target_col = label_cols[0]
            
            # Filter out rows with empty labels
            train_df = train_df[train_df[target_col].notna() & (train_df[target_col].str.strip() != '')]
            
            if len(train_df) < 10:
                raise ValueError(f"Not enough labeled samples after filtering. Only {len(train_df)} remain.")
            
            # Create train/val split
            train_indices = train_df.sample(frac=0.8, random_state=42).index
            val_indices = train_df.drop(train_indices).index
            train_split = train_df.loc[train_indices]
            val_split = train_df.loc[val_indices]
            
            # Get unique labels for encoding
            all_labels = sorted(train_df[target_col].unique().tolist())
            label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
            idx_to_label = {idx: label for label, idx in label_to_idx.items()}
            num_labels = len(all_labels)
            
            logger.info(f"Found {num_labels} unique labels from columns: {label_cols}")
            
            # Encode labels
            train_labels = [label_to_idx[l] for l in train_split[target_col].tolist()]
            val_labels = [label_to_idx[l] for l in val_split[target_col].tolist()]
            
            # Get texts
            train_texts = train_split[text_col].tolist()
            val_texts = val_split[text_col].tolist()
            
            # Initialize tokenizer - handle BigBird compatibility
            model_name = hp.get('model_name', 'google/bigbird-roberta-base')
            logger.info(f"Loading tokenizer for {model_name}")
            
            if 'bigbird' in model_name.lower():
                from transformers import BigBirdTokenizer
                tokenizer = BigBirdTokenizer.from_pretrained(model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Create datasets
            max_length = hp.get('max_length', 512)
            train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
            val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
            
            logger.info(f"Created datasets: {len(train_dataset)} train, {len(val_dataset)} val")
            
            # Initialize model
            logger.info(f"Loading model {model_name} with {num_labels} labels")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                problem_type="single_label_classification"
            )
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=model_path,
                num_train_epochs=epochs,
                per_device_train_batch_size=hp.get('batch_size', 8),
                per_device_eval_batch_size=hp.get('batch_size', 8),
                learning_rate=hp.get('learning_rate', 2e-5),
                weight_decay=hp.get('weight_decay', 0.01),
                warmup_ratio=hp.get('warmup_ratio', 0.06),
                lr_scheduler_type=hp.get('scheduler', 'linear'),
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                logging_dir=os.path.join(model_path, 'logs'),
                logging_steps=10,
                save_total_limit=2,
                fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
                dataloader_num_workers=0,  # Windows compatibility
                report_to="none",  # Disable wandb etc
            )
            
            # Setup callbacks
            callbacks = []
            if early_stopping:
                callbacks.append(EarlyStoppingCallback(
                    early_stopping_patience=hp.get('patience', 3),
                    early_stopping_threshold=hp.get('min_delta', 0.001)
                ))
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks=callbacks
            )
            
            # Train!
            logger.info("Starting transformer training...")
            train_result = trainer.train()
            
            # Evaluate
            eval_result = trainer.evaluate()
            
            # Save model
            trainer.save_model(model_path)
            tokenizer.save_pretrained(model_path)
            
            # Save label mapping
            labels_path = os.path.join(model_path, 'labels.json')
            with open(labels_path, 'w') as f:
                json.dump({
                    'labels': all_labels,
                    'label_to_idx': label_to_idx,
                    'idx_to_label': idx_to_label
                }, f, indent=2)
            
            # Compute metrics
            metrics = {
                'status': 'completed',
                'n_samples': len(train_split),
                'n_val_samples': len(val_split),
                'n_labels': num_labels,
                'label_columns': label_cols,
                'text_column': text_col,
                'epochs': epochs,
                'epochs_trained': train_result.global_step // len(train_dataset) * hp.get('batch_size', 8),
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result.get('eval_loss'),
                'labels': all_labels[:20],  # First 20 labels for reference
                'hyperparameters': hp,
                'model_saved': True
            }
            
            # Save training config with hyperparameters
            config_path = os.path.join(model_path, 'training_config.json')
            with open(config_path, 'w') as f:
                json.dump({
                    'session_id': session_id,
                    'text_column': text_col,
                    'label_columns': label_cols,
                    'target_column': target_col,
                    'n_labels': num_labels,
                    'label_mapping': label_to_idx,
                    'epochs': epochs,
                    'early_stopping': early_stopping,
                    'hyperparameters': hp,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Training complete - Loss: {train_result.training_loss:.4f}, Eval Loss: {eval_result.get('eval_loss', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            metrics = {'error': str(e), 'status': 'failed'}
        
        # Create session record
        history = self.get_training_history()
        
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
