"""Shared test fixtures and configuration for legal clause risk scorer tests.

This module provides common test fixtures, sample data, and configuration
that can be reused across all test modules.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
import torch
import numpy as np
import pandas as pd

# Add src to path for imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from legal_clause_risk_scorer.utils.config import Config
from legal_clause_risk_scorer.data.loader import LegalDataLoader
from legal_clause_risk_scorer.data.preprocessing import LegalTextPreprocessor
from legal_clause_risk_scorer.models.model import LegalClauseRiskModel
from legal_clause_risk_scorer.evaluation.metrics import LegalRiskMetrics


@pytest.fixture(scope="session")
def sample_config():
    """Create a sample configuration for testing."""
    config_data = {
        'data': {
            'max_sequence_length': 128,
            'truncation': True,
            'padding': 'max_length',
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
            'risk_categories': {
                'termination_clauses': {
                    'weight': 0.25,
                    'keywords': ['terminate', 'dismissal', 'at-will']
                },
                'non_compete': {
                    'weight': 0.20,
                    'keywords': ['non-compete', 'restraint', 'solicitation']
                }
            }
        },
        'model': {
            'base_model': 'distilbert-base-uncased',  # Smaller model for testing
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'num_labels': 3,
            'dropout': 0.1,
            'hidden_size': 768
        },
        'training': {
            'batch_size': 4,
            'learning_rate': 2e-5,
            'num_epochs': 2,
            'warmup_steps': 10,
            'weight_decay': 0.01,
            'optimizer': 'AdamW',
            'scheduler': 'linear',
            'early_stopping_patience': 2,
            'save_steps': 50,
            'eval_steps': 25,
            'logging_steps': 10,
            'max_grad_norm': 1.0,
            'use_cuda': False,  # Disable CUDA for testing
            'dataloader_num_workers': 0
        },
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'mae'],
            'targets': {
                'clause_detection_f1': 0.82,
                'risk_score_mae': 1.2,
                'unfavorable_term_recall': 0.88
            }
        },
        'mlflow': {
            'experiment_name': 'test_legal_clause_risk_scorer',
            'tracking_uri': 'memory:',  # In-memory tracking for tests
            'artifact_location': 'test_artifacts'
        },
        'logging': {
            'level': 'WARNING',  # Reduce logging during tests
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'test_logs/test.log'
        },
        'random_seed': 42,
        'torch_seed': 42,
        'numpy_seed': 42,
        'transformers_seed': 42,
        'paths': {
            'data_dir': 'test_data',
            'models_dir': 'test_models',
            'logs_dir': 'test_logs',
            'cache_dir': 'test_cache',
            'outputs_dir': 'test_outputs'
        }
    }

    # Create a temporary config instance
    config = Config.__new__(Config)
    config._config = config_data
    config.config_path = Path("test_config.yaml")
    return config


@pytest.fixture(scope="session")
def sample_legal_texts():
    """Sample legal clause texts for testing."""
    return [
        {
            'text': "Employee may terminate this agreement with 30 days written notice.",
            'clause_type': 'termination',
            'risk_score': 3.2,
            'risk_category': 'low_risk'
        },
        {
            'text': "Company may terminate employee at any time without cause or notice.",
            'clause_type': 'termination',
            'risk_score': 8.7,
            'risk_category': 'high_risk'
        },
        {
            'text': "Employee agrees not to compete for 2 years in any capacity worldwide.",
            'clause_type': 'non_compete',
            'risk_score': 9.1,
            'risk_category': 'high_risk'
        },
        {
            'text': "All intellectual property developed shall belong to the Company.",
            'clause_type': 'intellectual_property',
            'risk_score': 6.2,
            'risk_category': 'medium_risk'
        },
        {
            'text': "Employee will receive fair compensation with annual reviews.",
            'clause_type': 'compensation',
            'risk_score': 2.8,
            'risk_category': 'low_risk'
        }
    ]


@pytest.fixture(scope="session")
def sample_dataset(sample_legal_texts):
    """Create a sample dataset for testing."""
    import pandas as pd
    from datasets import Dataset

    df = pd.DataFrame(sample_legal_texts)
    dataset = Dataset.from_pandas(df)
    return dataset


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="legal_risk_test_")
    yield Path(temp_dir)
    # Cleanup after tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_tokenized_data():
    """Mock tokenized data for testing."""
    batch_size = 4
    seq_length = 128

    return {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
        'attention_mask': torch.ones(batch_size, seq_length),
        'labels': torch.randint(0, 3, (batch_size,)),
        'risk_scores': torch.rand(batch_size) * 9 + 1  # Random scores between 1-10
    }


@pytest.fixture
def mock_evaluation_data():
    """Mock evaluation data for testing metrics."""
    n_samples = 50
    np.random.seed(42)

    # Generate synthetic ground truth and predictions
    y_true_class = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.35, 0.25])
    y_pred_class = y_true_class + np.random.choice([-1, 0, 1], size=n_samples, p=[0.1, 0.8, 0.1])
    y_pred_class = np.clip(y_pred_class, 0, 2)

    y_true_score = np.random.normal(5.0, 2.0, n_samples)
    y_true_score = np.clip(y_true_score, 1, 10)
    y_pred_score = y_true_score + np.random.normal(0, 0.8, n_samples)
    y_pred_score = np.clip(y_pred_score, 1, 10)

    # Generate prediction probabilities
    y_prob = np.random.dirichlet([2, 2, 2], size=n_samples)

    return {
        'y_true_class': y_true_class,
        'y_pred_class': y_pred_class,
        'y_true_score': y_true_score,
        'y_pred_score': y_pred_score,
        'y_prob': y_prob
    }


@pytest.fixture
def sample_csv_file(temp_dir, sample_legal_texts):
    """Create a sample CSV file for testing data loading."""
    df = pd.DataFrame(sample_legal_texts)
    csv_path = temp_dir / "sample_clauses.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


class MockDataset:
    """Mock dataset class for testing."""

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def to_pandas(self):
        return pd.DataFrame(self.data)


class MockTokenizer:
    """Mock tokenizer class for testing."""

    def __init__(self):
        self.pad_token = "[PAD]"
        self.pad_token_id = 0

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]

        # Simple mock tokenization
        max_length = kwargs.get('max_length', 128)
        batch_size = len(texts)

        # Create mock tokens
        input_ids = torch.randint(1, 1000, (batch_size, max_length))
        attention_mask = torch.ones(batch_size, max_length)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def encode(self, text, **kwargs):
        return [1, 2, 3, 4, 5]  # Mock token ids

    def decode(self, token_ids, **kwargs):
        return "mock decoded text"


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer fixture."""
    return MockTokenizer()


@pytest.fixture
def sample_config_file(temp_dir):
    """Create a sample configuration file for testing."""
    config_content = """
data:
  max_sequence_length: 128
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

model:
  base_model: "distilbert-base-uncased"
  num_labels: 3
  dropout: 0.1

training:
  batch_size: 4
  learning_rate: 0.0001
  num_epochs: 2
  use_cuda: false

evaluation:
  targets:
    clause_detection_f1: 0.82
    risk_score_mae: 1.2

logging:
  level: "WARNING"

random_seed: 42
"""

    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)

    return config_path


# Test data constants
SAMPLE_LEGAL_CLAUSES = [
    "The employee may be terminated at will without notice.",
    "Employee agrees to confidentiality terms during employment.",
    "Intellectual property rights belong to the company.",
    "Non-compete restrictions apply for 1 year post-employment.",
    "Salary reviews will be conducted annually."
]

SAMPLE_RISK_SCORES = [8.5, 4.2, 6.1, 7.8, 3.0]
SAMPLE_RISK_CATEGORIES = ['high_risk', 'medium_risk', 'medium_risk', 'high_risk', 'low_risk']


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if "slow" in item.name or "training" in item.name:
            item.add_marker(pytest.mark.slow)

        # Mark unit tests (default)
        if not any(marker.name in ["integration", "slow"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)