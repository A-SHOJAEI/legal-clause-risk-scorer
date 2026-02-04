"""Data loading and preprocessing modules.

This package contains modules for loading legal datasets (CUAD, LEDGAR),
preprocessing legal text, and preparing data for model training.
"""

from .loader import (
    LegalDataLoader,
    create_data_loader
)

from .preprocessing import (
    LegalTextPreprocessor,
    create_preprocessor,
    preprocess_dataset_batch
)

__all__ = [
    'LegalDataLoader',
    'create_data_loader',
    'LegalTextPreprocessor',
    'create_preprocessor',
    'preprocess_dataset_batch'
]