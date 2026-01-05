"""
IFS Activity Transformer - A web application for training activity classifiers.

This package provides:
- Multi-label activity classification using transformer models
- Discipline code configuration and label mapping
- Optimized text preparation for RoBERTa
- Active learning pipeline for annotation
- Oracle database integration
- Azure backup services
"""
__version__ = '1.0.0'

# Core configuration
from app.config import (
    Config,
    DatabaseConfig,
    ModelConfig,
    FlaskConfig,
    AzureConfig,
    DisciplineConfig
)

# Label mapping utilities
from app.label_mapper import LabelMapper, get_label_mapper

# Text preparation for optimal RoBERTa input
from app.text_preparation import (
    ActivityTextPreparer,
    TextPrepConfig,
    TemplateBasedPreparer,
    create_training_text,
    compare_text_strategies
)

# Models
from app.multilabel_model import (
    MultiLabelActivityClassifier,
    MultiLabelTrainer,
    ActivityClassificationPipeline
)

__all__ = [
    # Version
    '__version__',
    # Configuration
    'Config',
    'DatabaseConfig', 
    'ModelConfig',
    'FlaskConfig',
    'AzureConfig',
    'DisciplineConfig',
    # Label mapping
    'LabelMapper',
    'get_label_mapper',
    # Text preparation
    'ActivityTextPreparer',
    'TextPrepConfig',
    'TemplateBasedPreparer',
    'create_training_text',
    'compare_text_strategies',
    # Models
    'MultiLabelActivityClassifier',
    'MultiLabelTrainer',
    'ActivityClassificationPipeline',
]
