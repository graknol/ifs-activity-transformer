"""
Model training utilities for RoBERTa BigBird classifier.
"""
import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from dotenv import load_dotenv

load_dotenv()


class ActivityClassifier:
    """Handles training and inference for activity classification."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the classifier.
        
        Args:
            model_name: HuggingFace model name, defaults to BigBird RoBERTa
        """
        self.model_name = model_name or os.getenv('MODEL_NAME', 'google/bigbird-roberta-base')
        self.max_length = int(os.getenv('MAX_LENGTH', '512'))
        self.batch_size = int(os.getenv('BATCH_SIZE', '8'))
        self.learning_rate = float(os.getenv('LEARNING_RATE', '2e-5'))
        self.num_epochs = int(os.getenv('NUM_EPOCHS', '3'))
        self.save_dir = os.getenv('SAVE_DIR', './models/saved')
        
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.trainer = None
        
    def prepare_data(self, df: pd.DataFrame, 
                    text_column: str = 'activity_description',
                    label_column: str = 'new_activity_path',
                    test_size: float = 0.2) -> Tuple[Dataset, Dataset, int]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with training data
            text_column: Name of column containing text
            label_column: Name of column containing labels
            test_size: Fraction of data to use for validation
        
        Returns:
            Tuple of (train_dataset, val_dataset, num_labels)
        """
        # Encode labels
        labels = self.label_encoder.fit_transform(df[label_column])
        df['labels'] = labels
        
        # Split data
        train_df, val_df = train_test_split(
            df[[text_column, 'labels']], 
            test_size=test_size,
            stratify=labels,
            random_state=42
        )
        
        # Create datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Tokenize datasets
        train_dataset = train_dataset.map(
            lambda x: self._tokenize_function(x, text_column),
            batched=True
        )
        val_dataset = val_dataset.map(
            lambda x: self._tokenize_function(x, text_column),
            batched=True
        )
        
        num_labels = len(self.label_encoder.classes_)
        
        return train_dataset, val_dataset, num_labels
    
    def _tokenize_function(self, examples: Dict, text_column: str) -> Dict:
        """Tokenize text examples."""
        return self.tokenizer(
            examples[text_column],
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
    
    def initialize_model(self, num_labels: int):
        """
        Initialize the model for training.
        
        Args:
            num_labels: Number of unique labels/classes
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset, 
              output_dir: Optional[str] = None) -> Dict:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save model checkpoints
        
        Returns:
            Dictionary with training metrics
        """
        if output_dir is None:
            output_dir = self.save_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            report_to="none"
        )
        
        # Create data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics
        )
        
        # Train model
        train_result = self.trainer.train()
        
        # Save model
        self.save_model(output_dir)
        
        # Get metrics
        metrics = train_result.metrics
        eval_metrics = self.trainer.evaluate()
        
        return {
            'train_metrics': metrics,
            'eval_metrics': eval_metrics
        }
    
    def _compute_metrics(self, eval_pred):
        """Compute accuracy metrics."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {'accuracy': accuracy}
    
    def save_model(self, output_dir: str):
        """
        Save model and label encoder.
        
        Args:
            output_dir: Directory to save model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label encoder
        label_mapping = {
            str(i): label for i, label in enumerate(self.label_encoder.classes_)
        }
        with open(os.path.join(output_dir, 'label_mapping.json'), 'w') as f:
            json.dump(label_mapping, f, indent=2)
    
    def load_model(self, model_dir: str):
        """
        Load trained model.
        
        Args:
            model_dir: Directory containing saved model
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load label mapping
        with open(os.path.join(model_dir, 'label_mapping.json'), 'r') as f:
            label_mapping = json.load(f)
        
        self.label_encoder.classes_ = np.array([label_mapping[str(i)] 
                                                for i in range(len(label_mapping))])
    
    def predict(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of text strings to classify
        
        Returns:
            List of tuples (predicted_label, confidence)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            confidences = torch.max(probs, dim=-1).values
        
        # Decode predictions
        results = []
        for pred, conf in zip(predictions, confidences):
            label = self.label_encoder.classes_[pred.item()]
            results.append((label, conf.item()))
        
        return results
