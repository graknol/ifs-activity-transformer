"""
Multi-label classifier for activity categorization.

This module implements a hybrid model combining RoBERTa text encoding
with tabular features for multi-label classification.

The model uses discipline codes configuration from discipline_codes.json
to derive multi-label targets for:
- Phase (ENGINEERING, FABRICATION, INSTALLATION_OFFSHORE, etc.)
- Discipline (ELECTRICAL, PIPING, STRUCTURAL, etc.)
- Work Type (DIRECT, INDIRECT, VENDORS_3RD_PARTY, etc.)
- Location (ONSHORE, OFFSHORE)

Key optimization for RoBERTa:
- Uses rich text preparation combining activity description with contextual information
- Includes sub-project description, project name, and hierarchical context
- Converts numerical features to natural language for transformer understanding
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from app.config import ModelConfig, DisciplineConfig
from app.label_mapper import LabelMapper, get_label_mapper
from app.text_preparation import (
    ActivityTextPreparer, 
    TextPrepConfig, 
    create_training_text,
    TemplateBasedPreparer
)


class MultiLabelActivityClassifier(nn.Module):
    """
    Multi-label classifier combining text and tabular features.
    
    Architecture:
    - RoBERTa encoder for activity descriptions
    - Feed-forward network for numerical features
    - Combined classifier head with sigmoid activation
    """
    
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_tabular_features: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize multi-label classifier.
        
        Args:
            model_name: HuggingFace model name (e.g., roberta-base)
            num_labels: Number of label categories
            num_tabular_features: Number of additional numerical features
            hidden_dim: Hidden dimension for fusion layer
            dropout: Dropout rate
        """
        super().__init__()
        
        # Load pre-trained transformer (will use cached version if available)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.transformer_dim = self.transformer.config.hidden_size
        
        # Tabular feature encoder (if we have numerical features)
        self.has_tabular = num_tabular_features > 0
        if self.has_tabular:
            self.tabular_encoder = nn.Sequential(
                nn.Linear(num_tabular_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            fusion_dim = self.transformer_dim + hidden_dim
        else:
            fusion_dim = self.transformer_dim
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )
        
        self.num_labels = num_labels
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tabular_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Tokenized text input
            attention_mask: Attention mask
            tabular_features: Optional numerical features
            
        Returns:
            Logits for multi-label classification
        """
        # Encode text with transformer
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        text_features = transformer_output.last_hidden_state[:, 0, :]
        
        # Combine with tabular features if available
        if self.has_tabular and tabular_features is not None:
            tabular_encoded = self.tabular_encoder(tabular_features)
            combined = torch.cat([text_features, tabular_encoded], dim=1)
        else:
            combined = text_features
        
        # Classify
        logits = self.classifier(combined)
        
        return logits


class MultiLabelTrainer:
    """
    Trainer for multi-label activity classification.
    
    Handles training, validation, and inference for the hybrid model.
    """
    
    def __init__(
        self,
        model: MultiLabelActivityClassifier,
        tokenizer: AutoTokenizer,
        config: ModelConfig,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Multi-label classifier model
            tokenizer: Tokenizer for text encoding
            config: Model configuration
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # BCE loss for multi-label classification
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
    
    def train_epoch(
        self,
        train_data: pd.DataFrame,
        text_column: str,
        label_columns: List[str],
        tabular_columns: Optional[List[str]] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_data: Training DataFrame
            text_column: Column with text descriptions
            label_columns: Columns with binary labels
            tabular_columns: Optional numerical feature columns
            batch_size: Batch size (uses config if None)
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        batch_size = batch_size or self.config.batch_size
        
        total_loss = 0
        num_batches = 0
        
        # Simple batching (in production, use DataLoader)
        for i in range(0, len(train_data), batch_size):
            batch = train_data.iloc[i:i+batch_size]
            
            # Tokenize text
            texts = batch[text_column].tolist()
            encoding = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Get labels
            labels = torch.tensor(
                batch[label_columns].values,
                dtype=torch.float32
            ).to(self.device)
            
            # Get tabular features if available
            tabular_features = None
            if tabular_columns:
                tabular_features = torch.tensor(
                    batch[tabular_columns].values,
                    dtype=torch.float32
                ).to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask, tabular_features)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return {
            'train_loss': total_loss / num_batches if num_batches > 0 else 0
        }
    
    def predict(
        self,
        data: pd.DataFrame,
        text_column: str,
        tabular_columns: Optional[List[str]] = None,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Make predictions on data.
        
        Args:
            data: DataFrame with samples
            text_column: Column with text
            tabular_columns: Optional numerical columns
            batch_size: Batch size
            
        Returns:
            Prediction probabilities (n_samples, n_labels)
        """
        self.model.eval()
        batch_size = batch_size or self.config.batch_size
        
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i+batch_size]
                
                # Tokenize
                texts = batch[text_column].tolist()
                encoding = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Tabular features
                tabular_features = None
                if tabular_columns:
                    tabular_features = torch.tensor(
                        batch[tabular_columns].values,
                        dtype=torch.float32
                    ).to(self.device)
                
                # Predict
                logits = self.model(input_ids, attention_mask, tabular_features)
                probs = torch.sigmoid(logits)
                
                all_predictions.append(probs.cpu().numpy())
        
        return np.vstack(all_predictions)
    
    def evaluate(
        self,
        eval_data: pd.DataFrame,
        text_column: str,
        label_columns: List[str],
        tabular_columns: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on validation data.
        
        Args:
            eval_data: Evaluation DataFrame
            text_column: Text column name
            label_columns: Label column names
            tabular_columns: Optional numerical columns
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(eval_data, text_column, tabular_columns)
        true_labels = eval_data[label_columns].values
        
        # Calculate metrics
        pred_binary = (predictions > 0.5).astype(int)
        
        # Accuracy per label
        accuracy_per_label = (pred_binary == true_labels).mean(axis=0)
        
        # Overall accuracy (exact match)
        exact_match = (pred_binary == true_labels).all(axis=1).mean()
        
        # Hamming accuracy (average per-label accuracy)
        hamming_acc = (pred_binary == true_labels).mean()
        
        return {
            'exact_match_accuracy': float(exact_match),
            'hamming_accuracy': float(hamming_acc),
            'mean_label_accuracy': float(accuracy_per_label.mean()),
            'label_accuracies': accuracy_per_label.tolist()
        }


class ActivityClassificationPipeline:
    """
    End-to-end pipeline for activity classification using discipline codes.
    
    This pipeline integrates:
    - DisciplineConfig for label definitions
    - LabelMapper for feature engineering and label derivation
    - MultiLabelActivityClassifier for predictions
    
    Example usage:
        >>> pipeline = ActivityClassificationPipeline()
        >>> prepared_df = pipeline.prepare_data(raw_df)
        >>> pipeline.train(prepared_df)
        >>> predictions = pipeline.predict(new_df)
    """
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        discipline_config: Optional[DisciplineConfig] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the classification pipeline.
        
        Args:
            model_config: Model training configuration
            discipline_config: Discipline codes configuration
            device: Device for training ('cuda' or 'cpu', auto-detected if None)
        """
        self.model_config = model_config or ModelConfig.from_env()
        self.discipline_config = discipline_config or DisciplineConfig.load()
        self.label_mapper = LabelMapper(self.discipline_config)
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        
        # Model and trainer will be initialized during training
        self.model: Optional[MultiLabelActivityClassifier] = None
        self.trainer: Optional[MultiLabelTrainer] = None
        
        # Store label information
        self._label_names = self.label_mapper.label_names
        self._num_labels = self.label_mapper.num_labels
    
    @property
    def num_labels(self) -> int:
        """Number of classification labels."""
        return self._num_labels
    
    @property
    def label_names(self) -> List[str]:
        """Names of all classification labels."""
        return self._label_names.copy()
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        sub_project_id_column: str = 'sub_project_id',
        text_column: str = 'activity_description',
        text_strategy: str = 'rich',
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Prepare training data with derived labels and optimized text.
        
        This method does two critical things:
        1. Derives multi-label targets from discipline codes
        2. Creates rich text representations for optimal RoBERTa input
        
        Text strategies:
        - 'rich': Full context (description + sub-project + project + metadata)
        - 'minimal': Description only (baseline for comparison)
        - 'template': Structured template-based format
        
        Args:
            df: Raw DataFrame with activity data
            sub_project_id_column: Column containing sub-project IDs
            text_column: Column containing activity descriptions
            text_strategy: Text preparation strategy ('rich', 'minimal', 'template')
            validate: Whether to print validation statistics
            
        Returns:
            DataFrame with:
            - discipline_code: Extracted discipline code
            - binary_labels: Multi-label encoding
            - prepared_text: Optimized text for RoBERTa
            - text_minimal: Original description (for comparison)
        """
        # Use label mapper to derive labels
        prepared_df = self.label_mapper.prepare_training_dataframe(
            df,
            sub_project_id_column=sub_project_id_column,
            text_column=text_column
        )
        
        # Prepare optimized text for RoBERTa
        print(f"  Preparing text with '{text_strategy}' strategy...")
        prepared_df = create_training_text(prepared_df, strategy=text_strategy)
        
        if validate:
            stats = self.label_mapper.validate_dataset(df, sub_project_id_column)
            print(f"\nDataset validation:")
            print(f"  Total records: {stats['total_records']}")
            print(f"  Valid codes: {stats['valid_codes']} ({stats['coverage_pct']:.1f}%)")
            print(f"  Invalid codes: {stats['invalid_codes']}")
            print(f"  Missing codes: {stats['missing_codes']}")
            if stats['unmapped_codes']:
                print(f"  Unmapped codes: {stats['unmapped_codes']}")
            print(f"  Phase distribution: {stats['phase_distribution']}")
            
            # Show text preparation example
            if len(prepared_df) > 0:
                print(f"\n  Sample prepared text (first record):")
                sample_text = prepared_df.iloc[0].get('prepared_text', '')
                print(f"  {sample_text[:200]}..." if len(sample_text) > 200 else f"  {sample_text}")
        
        return prepared_df
    
    def build_model(
        self,
        num_tabular_features: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ) -> MultiLabelActivityClassifier:
        """
        Build the multi-label classifier model.
        
        Args:
            num_tabular_features: Number of additional numerical features
            hidden_dim: Hidden dimension for fusion layer
            dropout: Dropout rate
            
        Returns:
            Initialized model
        """
        self.model = MultiLabelActivityClassifier(
            model_name=self.model_config.model_name,
            num_labels=self._num_labels,
            num_tabular_features=num_tabular_features,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.trainer = MultiLabelTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.model_config,
            device=self.device
        )
        
        return self.model
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        text_column: str = 'prepared_text',  # Use prepared text by default
        tabular_columns: Optional[List[str]] = None,
        num_epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the classifier.
        
        IMPORTANT: Uses 'prepared_text' column by default for optimal RoBERTa input.
        This column contains the rich text representation prepared by prepare_data().
        
        Args:
            train_df: Prepared training DataFrame (must have binary_labels column)
            val_df: Optional validation DataFrame
            text_column: Column with text (default: 'prepared_text' for rich context)
            tabular_columns: Optional numerical feature columns
            num_epochs: Number of training epochs (uses config if None)
            
        Returns:
            Training history with metrics
        """
        if self.model is None:
            num_tabular = len(tabular_columns) if tabular_columns else 0
            self.build_model(num_tabular_features=num_tabular)
        
        num_epochs = num_epochs or self.model_config.num_epochs
        history = {'train_loss': [], 'val_metrics': []}
        
        # Verify text column exists
        if text_column not in train_df.columns:
            available = [c for c in train_df.columns if 'text' in c.lower() or 'description' in c.lower()]
            raise ValueError(
                f"Column '{text_column}' not found. "
                f"Available text columns: {available}. "
                f"Did you run prepare_data() first?"
            )
        
        # Create binary label columns from binary_labels array column
        label_columns = self.label_names
        for i, label in enumerate(label_columns):
            train_df[label] = train_df['binary_labels'].apply(lambda x: x[i])
            if val_df is not None:
                val_df[label] = val_df['binary_labels'].apply(lambda x: x[i])
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.trainer.train_epoch(
                train_df,
                text_column=text_column,
                label_columns=label_columns,
                tabular_columns=tabular_columns
            )
            history['train_loss'].append(train_metrics['train_loss'])
            
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_metrics['train_loss']:.4f}")
            
            # Validate
            if val_df is not None:
                val_metrics = self.trainer.evaluate(
                    val_df,
                    text_column=text_column,
                    label_columns=label_columns,
                    tabular_columns=tabular_columns
                )
                history['val_metrics'].append(val_metrics)
                print(f"  Validation - Hamming Acc: {val_metrics['hamming_accuracy']:.4f}, "
                      f"Exact Match: {val_metrics['exact_match_accuracy']:.4f}")
        
        return history
    
    def predict(
        self,
        df: pd.DataFrame,
        text_column: str = 'prepared_text',  # Use prepared text by default
        tabular_columns: Optional[List[str]] = None,
        return_labels: bool = True,
        prepare_text: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        For inference, if the text isn't already prepared, set prepare_text=True
        to automatically create the rich text representation.
        
        Args:
            df: DataFrame with activity data
            text_column: Column containing text (default: 'prepared_text')
            tabular_columns: Optional numerical feature columns
            return_labels: Whether to decode predictions to label names
            prepare_text: If True and 'prepared_text' not in df, prepare it
            
        Returns:
            DataFrame with predictions
        """
        if self.trainer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        result = df.copy()
        
        # Auto-prepare text if needed for inference
        if prepare_text and text_column == 'prepared_text' and 'prepared_text' not in result.columns:
            preparer = ActivityTextPreparer()
            result = preparer.prepare_dataframe(result)
        
        # Get probability predictions
        probs = self.trainer.predict(
            result,
            text_column=text_column,
            tabular_columns=tabular_columns
        )
        
        result = df.copy()
        result['prediction_probs'] = list(probs)
        
        if return_labels:
            # Decode to label names
            decoded = [
                self.label_mapper.decode_binary_labels(p)
                for p in probs
            ]
            result['predicted_phase'] = [d['phase'] for d in decoded]
            result['predicted_discipline'] = [d['discipline'] for d in decoded]
            result['predicted_work_type'] = [d['work_type'] for d in decoded]
            result['predicted_location'] = [d['location'] for d in decoded]
        
        return result
    
    def get_recommended_sql(self) -> Optional[str]:
        """Get recommended SQL query for fetching training data."""
        return self.discipline_config.get_training_sql_query()
    
    def get_label_info(self) -> Dict[str, List[str]]:
        """Get information about label categories."""
        return {
            'phase_labels': self.discipline_config.phase_labels,
            'discipline_labels': self.discipline_config.discipline_labels,
            'work_type_labels': self.discipline_config.work_type_labels,
            'location_labels': self.discipline_config.location_labels,
            'total_labels': self._num_labels
        }

