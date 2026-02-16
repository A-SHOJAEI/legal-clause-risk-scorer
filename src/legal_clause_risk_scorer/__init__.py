"""Legal Clause Risk Scorer - A contract clause risk assessment system.

This package identifies potentially unfavorable legal terms in employment contracts
and NDAs, scoring them from 1-10 based on employee-friendliness.

Key Features:
- Multi-task learning for classification and regression
- Legal domain-specific preprocessing
- Production-ready training and evaluation pipeline
- MLflow experiment tracking
- Comprehensive evaluation metrics

Example Usage:
    >>> from legal_clause_risk_scorer.utils.config import load_config
    >>> from legal_clause_risk_scorer.models.model import LegalClauseRiskModel
    >>> from legal_clause_risk_scorer.data.loader import LegalDataLoader
    >>>
    >>> config = load_config()
    >>> model = LegalClauseRiskModel(config)
    >>> data_loader = LegalDataLoader(config)
"""

__version__ = "1.0.0"
__author__ = "Legal AI Research Team"
__email__ = "team@legalai.example.com"

# Import key classes and functions for easy access
from .utils.config import Config, load_config
from .models.model import LegalClauseRiskModel, LegalEmbeddingModel
from .data.loader import LegalDataLoader
from .data.preprocessing import LegalTextPreprocessor
from .training.trainer import LegalRiskTrainer
from .evaluation.metrics import LegalRiskMetrics

# Define what gets imported with "from legal_clause_risk_scorer import *"
__all__ = [
    # Core classes
    'Config',
    'LegalClauseRiskModel',
    'LegalEmbeddingModel',
    'LegalDataLoader',
    'LegalTextPreprocessor',
    'LegalRiskTrainer',
    'LegalRiskMetrics',

    # Utility functions
    'load_config',

    # Package metadata
    '__version__',
    '__author__',
    '__email__'
]