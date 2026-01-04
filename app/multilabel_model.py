"""
Multi-label classifier for activity categorization.

This module implements a hybrid model combining RoBERTa text encoding
with tabular features for multi-label classification.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from app.config import ModelConfig


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
