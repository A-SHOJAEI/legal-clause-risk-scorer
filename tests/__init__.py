"""Test suite for legal clause risk scorer.

This package contains comprehensive unit tests, integration tests, and test utilities
for all components of the legal clause risk assessment system.

Test Structure:
- test_data.py: Tests for data loading and preprocessing
- test_model.py: Tests for model architecture and functionality
- test_training.py: Tests for training system and evaluation metrics
- conftest.py: Shared fixtures and test utilities

Run tests with:
    pytest tests/
    pytest tests/ -v  # verbose output
    pytest tests/ -m "not slow"  # skip slow tests
    pytest tests/ --cov=src/legal_clause_risk_scorer  # with coverage
"""

# Test package metadata
__test_version__ = "1.0.0"