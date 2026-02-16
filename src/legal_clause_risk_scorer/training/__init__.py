"""Training system for legal clause risk assessment models.

This package contains the training infrastructure including trainers,
dataset classes, and training utilities with MLflow integration.
"""

from .trainer import (
    LegalRiskDataset,
    LegalRiskTrainer,
    create_trainer
)

__all__ = [
    'LegalRiskDataset',
    'LegalRiskTrainer',
    'create_trainer'
]