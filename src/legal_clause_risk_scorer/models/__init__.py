"""Model architectures for legal clause risk assessment.

This package contains the core neural network models including multi-task
transformers for classification and regression, and specialized embedding models.
"""

from .model import (
    LegalClauseRiskModel,
    AttentionPooling,
    LegalEmbeddingModel,
    create_model,
    load_pretrained_model,
    save_model,
    count_parameters
)

__all__ = [
    'LegalClauseRiskModel',
    'AttentionPooling',
    'LegalEmbeddingModel',
    'create_model',
    'load_pretrained_model',
    'save_model',
    'count_parameters'
]