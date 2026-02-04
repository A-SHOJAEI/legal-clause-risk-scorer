"""Evaluation metrics and utilities for legal clause risk assessment.

This package contains comprehensive evaluation metrics, report generation,
and visualization tools for both classification and regression tasks.
"""

from .metrics import (
    LegalRiskMetrics,
    create_metrics_evaluator
)

__all__ = [
    'LegalRiskMetrics',
    'create_metrics_evaluator'
]