"""Test suite for training system and evaluation metrics.

This module tests the training pipeline, evaluation metrics, and related utilities.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import modules under test
from legal_clause_risk_scorer.training.trainer import (
    LegalRiskDataset,
    LegalRiskTrainer,
    create_trainer
)
from legal_clause_risk_scorer.evaluation.metrics import (
    LegalRiskMetrics,
    create_metrics_evaluator
)


class TestLegalRiskDataset:
    """Test suite for LegalRiskDataset class."""

    def test_dataset_initialization(self, sample_dataset, mock_tokenizer):
        """Test dataset initialization."""
        dataset = LegalRiskDataset(sample_dataset, mock_tokenizer)

        assert dataset.dataset == sample_dataset
        assert dataset.tokenizer == mock_tokenizer

    def test_dataset_length(self, sample_dataset, mock_tokenizer):
        """Test dataset length."""
        dataset = LegalRiskDataset(sample_dataset, mock_tokenizer)
        assert len(dataset) == len(sample_dataset)

    def test_dataset_getitem(self, sample_dataset, mock_tokenizer):
        """Test dataset item access."""
        # Mock the dataset to have the required fields
        mock_sample_dataset = Mock()
        mock_sample_dataset.__len__.return_value = 2
        mock_item = {
            'input_ids': [1, 2, 3, 0, 0],
            'attention_mask': [1, 1, 1, 0, 0],
            'labels': 1,
            'risk_scores': 5.5
        }
        mock_sample_dataset.__getitem__.return_value = mock_item

        dataset = LegalRiskDataset(mock_sample_dataset, mock_tokenizer)
        item = dataset[0]

        assert isinstance(item, dict)
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item
        assert 'risk_scores' in item

        # Check tensor types
        assert isinstance(item['input_ids'], torch.Tensor)
        assert isinstance(item['attention_mask'], torch.Tensor)
        assert isinstance(item['labels'], torch.Tensor)
        assert isinstance(item['risk_scores'], torch.Tensor)


class TestLegalRiskTrainer:
    """Test suite for LegalRiskTrainer class."""

    @pytest.fixture
    def mock_model(self, sample_config):
        """Create a mock model for testing."""
        model = Mock()
        model.config = sample_config
        model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
        model.to.return_value = model
        model.train.return_value = None
        model.eval.return_value = None
        return model

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset for testing."""
        dataset = Mock()
        dataset.__len__.return_value = 10
        dataset.__getitem__.return_value = {
            'input_ids': torch.randint(0, 1000, (128,)),
            'attention_mask': torch.ones(128),
            'labels': torch.tensor(1),
            'risk_scores': torch.tensor(5.5)
        }
        return dataset

    def test_trainer_initialization(self, sample_config, mock_model, mock_dataset):
        """Test trainer initialization."""
        with patch('legal_clause_risk_scorer.training.trainer.LegalRiskMetrics'), \
             patch('torch.cuda.is_available', return_value=False):

            trainer = LegalRiskTrainer(
                config=sample_config,
                model=mock_model,
                train_dataset=mock_dataset,
                val_dataset=mock_dataset
            )

            assert trainer.config == sample_config
            assert trainer.model == mock_model
            assert trainer.device.type == 'cpu'
            assert hasattr(trainer, 'optimizer')
            assert hasattr(trainer, 'metrics_evaluator')

    def test_device_setup(self, sample_config, mock_model, mock_dataset):
        """Test device setup logic."""
        with patch('legal_clause_risk_scorer.training.trainer.LegalRiskMetrics'):

            # Test CPU fallback
            with patch('torch.cuda.is_available', return_value=False):
                trainer = LegalRiskTrainer(
                    config=sample_config,
                    model=mock_model,
                    train_dataset=mock_dataset,
                    val_dataset=mock_dataset
                )
                assert trainer.device.type == 'cpu'

            # Test CUDA when disabled in config
            sample_config.set('training.use_cuda', False)
            with patch('torch.cuda.is_available', return_value=True):
                trainer = LegalRiskTrainer(
                    config=sample_config,
                    model=mock_model,
                    train_dataset=mock_dataset,
                    val_dataset=mock_dataset
                )
                assert trainer.device.type == 'cpu'

    def test_optimizer_creation(self, sample_config, mock_model, mock_dataset):
        """Test optimizer creation."""
        with patch('legal_clause_risk_scorer.training.trainer.LegalRiskMetrics'), \
             patch('torch.cuda.is_available', return_value=False):

            # Test AdamW optimizer
            sample_config.set('training.optimizer', 'AdamW')
            trainer = LegalRiskTrainer(
                config=sample_config,
                model=mock_model,
                train_dataset=mock_dataset,
                val_dataset=mock_dataset
            )

            from transformers import AdamW
            assert isinstance(trainer.optimizer, AdamW)

            # Test Adam optimizer
            sample_config.set('training.optimizer', 'Adam')
            trainer = LegalRiskTrainer(
                config=sample_config,
                model=mock_model,
                train_dataset=mock_dataset,
                val_dataset=mock_dataset
            )

            from torch.optim import Adam
            assert isinstance(trainer.optimizer, Adam)

            # Test invalid optimizer
            sample_config.set('training.optimizer', 'InvalidOptimizer')
            with pytest.raises(ValueError):
                LegalRiskTrainer(
                    config=sample_config,
                    model=mock_model,
                    train_dataset=mock_dataset,
                    val_dataset=mock_dataset
                )

    def test_scheduler_creation(self, sample_config, mock_model, mock_dataset):
        """Test learning rate scheduler creation."""
        with patch('legal_clause_risk_scorer.training.trainer.LegalRiskMetrics'), \
             patch('torch.cuda.is_available', return_value=False), \
             patch('legal_clause_risk_scorer.training.trainer.get_linear_schedule_with_warmup') as mock_scheduler:

            mock_scheduler_instance = Mock()
            mock_scheduler.return_value = mock_scheduler_instance

            # Test linear scheduler
            sample_config.set('training.scheduler', 'linear')
            trainer = LegalRiskTrainer(
                config=sample_config,
                model=mock_model,
                train_dataset=mock_dataset,
                val_dataset=mock_dataset
            )

            mock_scheduler.assert_called_once()
            assert trainer.scheduler == mock_scheduler_instance

            # Test no scheduler
            sample_config.set('training.scheduler', 'none')
            trainer = LegalRiskTrainer(
                config=sample_config,
                model=mock_model,
                train_dataset=mock_dataset,
                val_dataset=mock_dataset
            )

            assert trainer.scheduler is None

    def test_dataloader_creation(self, sample_config, mock_model, mock_dataset):
        """Test data loader creation."""
        with patch('legal_clause_risk_scorer.training.trainer.LegalRiskMetrics'), \
             patch('torch.cuda.is_available', return_value=False):

            trainer = LegalRiskTrainer(
                config=sample_config,
                model=mock_model,
                train_dataset=mock_dataset,
                val_dataset=mock_dataset
            )

            assert hasattr(trainer, 'train_loader')
            assert hasattr(trainer, 'val_loader')

            # Check batch size
            expected_batch_size = sample_config.get('training.batch_size')
            assert trainer.train_loader.batch_size == expected_batch_size

    @patch('legal_clause_risk_scorer.training.trainer.mlflow')
    def test_mlflow_setup(self, mock_mlflow, sample_config, mock_model, mock_dataset):
        """Test MLflow tracking setup."""
        with patch('legal_clause_risk_scorer.training.trainer.LegalRiskMetrics'), \
             patch('torch.cuda.is_available', return_value=False):

            trainer = LegalRiskTrainer(
                config=sample_config,
                model=mock_model,
                train_dataset=mock_dataset,
                val_dataset=mock_dataset
            )

            # Check MLflow setup calls
            mock_mlflow.set_tracking_uri.assert_called()
            mock_mlflow.set_experiment.assert_called()

    def test_checkpoint_saving(self, sample_config, mock_model, mock_dataset, temp_dir):
        """Test model checkpoint saving."""
        with patch('legal_clause_risk_scorer.training.trainer.LegalRiskMetrics'), \
             patch('torch.cuda.is_available', return_value=False):

            sample_config.set('paths.models_dir', str(temp_dir))

            trainer = LegalRiskTrainer(
                config=sample_config,
                model=mock_model,
                train_dataset=mock_dataset,
                val_dataset=mock_dataset
            )

            # Mock model state dict
            mock_model.state_dict.return_value = {'test': torch.tensor([1, 2, 3])}

            checkpoint_path = temp_dir / "test_checkpoint.pth"
            trainer._save_checkpoint(checkpoint_path, is_best=False)

            assert checkpoint_path.exists()

    def test_checkpoint_loading(self, sample_config, mock_model, mock_dataset, temp_dir):
        """Test model checkpoint loading."""
        with patch('legal_clause_risk_scorer.training.trainer.LegalRiskMetrics'), \
             patch('torch.cuda.is_available', return_value=False):

            sample_config.set('paths.models_dir', str(temp_dir))

            trainer = LegalRiskTrainer(
                config=sample_config,
                model=mock_model,
                train_dataset=mock_dataset,
                val_dataset=mock_dataset
            )

            # Create a dummy checkpoint
            checkpoint = {
                'epoch': 5,
                'global_step': 100,
                'model_state_dict': {'test': torch.tensor([1, 2, 3])},
                'optimizer_state_dict': {},
                'best_score': 0.85,
                'config': sample_config.to_dict()
            }

            checkpoint_path = temp_dir / "test_checkpoint.pth"
            torch.save(checkpoint, checkpoint_path)

            # Test loading
            trainer.load_checkpoint(checkpoint_path)

            assert trainer.epoch == 5
            assert trainer.global_step == 100
            assert trainer.best_score == 0.85

            # Test loading non-existent checkpoint
            with pytest.raises(FileNotFoundError):
                trainer.load_checkpoint(temp_dir / "non_existent.pth")

    @patch('legal_clause_risk_scorer.training.trainer.mlflow')
    def test_training_step(self, mock_mlflow, sample_config, mock_model, mock_dataset):
        """Test single training step."""
        with patch('legal_clause_risk_scorer.training.trainer.LegalRiskMetrics'), \
             patch('torch.cuda.is_available', return_value=False):

            # Mock model forward pass
            mock_outputs = {
                'loss': torch.tensor(0.5, requires_grad=True),
                'classification_loss': torch.tensor(0.3),
                'regression_loss': torch.tensor(0.2)
            }
            mock_model.return_value = mock_outputs
            mock_model.__call__.return_value = mock_outputs

            trainer = LegalRiskTrainer(
                config=sample_config,
                model=mock_model,
                train_dataset=mock_dataset,
                val_dataset=mock_dataset
            )

            # Mock the training loop components
            trainer.global_step = 0

            # This would typically be called within _train_epoch
            # We test the core training logic here
            batch = {
                'input_ids': torch.randint(0, 1000, (2, 128)),
                'attention_mask': torch.ones(2, 128),
                'labels': torch.tensor([0, 1]),
                'risk_scores': torch.tensor([3.0, 7.0])
            }

            # Simulate forward pass
            outputs = mock_model(**batch)
            loss = outputs['loss']

            assert loss is not None
            assert loss.requires_grad

    def test_create_trainer_factory(self, sample_config, mock_model, mock_dataset):
        """Test trainer factory function."""
        with patch('legal_clause_risk_scorer.training.trainer.LegalRiskMetrics'), \
             patch('torch.cuda.is_available', return_value=False):

            trainer = create_trainer(
                config=sample_config,
                model=mock_model,
                train_dataset=mock_dataset,
                val_dataset=mock_dataset
            )

            assert isinstance(trainer, LegalRiskTrainer)
            assert trainer.config == sample_config


class TestLegalRiskMetrics:
    """Test suite for LegalRiskMetrics class."""

    def test_metrics_initialization(self, sample_config):
        """Test metrics evaluator initialization."""
        metrics_evaluator = LegalRiskMetrics(sample_config)

        assert metrics_evaluator.config == sample_config
        assert hasattr(metrics_evaluator, 'target_metrics')
        assert hasattr(metrics_evaluator, 'class_names')
        assert len(metrics_evaluator.class_names) == 3

    def test_evaluate_classification(self, sample_config, mock_evaluation_data):
        """Test classification evaluation."""
        metrics_evaluator = LegalRiskMetrics(sample_config)

        metrics = metrics_evaluator.evaluate_classification(
            y_true=mock_evaluation_data['y_true_class'],
            y_pred=mock_evaluation_data['y_pred_class'],
            y_prob=mock_evaluation_data['y_prob']
        )

        # Check required metrics
        required_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert 0 <= metrics[metric] <= 1

        # Check per-class metrics
        for class_name in metrics_evaluator.class_names:
            assert f'precision_{class_name}' in metrics
            assert f'recall_{class_name}' in metrics
            assert f'f1_{class_name}' in metrics

    def test_evaluate_regression(self, sample_config, mock_evaluation_data):
        """Test regression evaluation."""
        metrics_evaluator = LegalRiskMetrics(sample_config)

        metrics = metrics_evaluator.evaluate_regression(
            y_true=mock_evaluation_data['y_true_score'],
            y_pred=mock_evaluation_data['y_pred_score']
        )

        # Check required metrics
        required_metrics = ['mae', 'mse', 'rmse', 'r2']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

        # Check score statistics
        assert 'pred_mean' in metrics
        assert 'pred_std' in metrics
        assert 'true_mean' in metrics
        assert 'true_std' in metrics

    def test_evaluate_multi_task(self, sample_config, mock_evaluation_data):
        """Test multi-task evaluation."""
        metrics_evaluator = LegalRiskMetrics(sample_config)

        metrics = metrics_evaluator.evaluate_multi_task(
            classification_true=mock_evaluation_data['y_true_class'],
            classification_pred=mock_evaluation_data['y_pred_class'],
            regression_true=mock_evaluation_data['y_true_score'],
            regression_pred=mock_evaluation_data['y_pred_score'],
            classification_prob=mock_evaluation_data['y_prob']
        )

        # Check structure
        assert 'classification' in metrics
        assert 'regression' in metrics
        assert 'consistency' in metrics
        assert 'overall_score' in metrics

        # Check overall score is reasonable
        assert isinstance(metrics['overall_score'], (int, float))
        assert 0 <= metrics['overall_score'] <= 1

    def test_scores_to_categories(self, sample_config):
        """Test score to category conversion."""
        metrics_evaluator = LegalRiskMetrics(sample_config)

        scores = np.array([2.0, 5.0, 8.5, 3.5, 7.0, 9.0])
        categories = metrics_evaluator._scores_to_categories(scores)

        expected = np.array([0, 1, 2, 0, 1, 2])  # low, medium, high, low, medium, high
        np.testing.assert_array_equal(categories, expected)

    def test_to_numpy_conversion(self, sample_config):
        """Test data conversion to numpy."""
        metrics_evaluator = LegalRiskMetrics(sample_config)

        # Test torch tensor conversion
        torch_tensor = torch.tensor([1, 2, 3])
        numpy_result = metrics_evaluator._to_numpy(torch_tensor)
        assert isinstance(numpy_result, np.ndarray)
        np.testing.assert_array_equal(numpy_result, [1, 2, 3])

        # Test list conversion
        list_data = [4, 5, 6]
        numpy_result = metrics_evaluator._to_numpy(list_data)
        assert isinstance(numpy_result, np.ndarray)
        np.testing.assert_array_equal(numpy_result, [4, 5, 6])

        # Test numpy array passthrough
        numpy_data = np.array([7, 8, 9])
        numpy_result = metrics_evaluator._to_numpy(numpy_data)
        assert isinstance(numpy_result, np.ndarray)
        np.testing.assert_array_equal(numpy_result, [7, 8, 9])

        # Test invalid type
        with pytest.raises(ValueError):
            metrics_evaluator._to_numpy("invalid")

    def test_generate_confusion_matrix(self, sample_config, mock_evaluation_data):
        """Test confusion matrix generation."""
        metrics_evaluator = LegalRiskMetrics(sample_config)

        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig'):

            cm = metrics_evaluator.generate_confusion_matrix(
                y_true=mock_evaluation_data['y_true_class'],
                y_pred=mock_evaluation_data['y_pred_class']
            )

            assert isinstance(cm, np.ndarray)
            assert cm.shape == (3, 3)  # 3 classes
            assert cm.sum() == len(mock_evaluation_data['y_true_class'])

    def test_generate_metrics_report(self, sample_config, mock_evaluation_data):
        """Test metrics report generation."""
        metrics_evaluator = LegalRiskMetrics(sample_config)

        # Create sample metrics
        classification_metrics = metrics_evaluator.evaluate_classification(
            y_true=mock_evaluation_data['y_true_class'],
            y_pred=mock_evaluation_data['y_pred_class']
        )

        regression_metrics = metrics_evaluator.evaluate_regression(
            y_true=mock_evaluation_data['y_true_score'],
            y_pred=mock_evaluation_data['y_pred_score']
        )

        metrics = {
            'classification': classification_metrics,
            'regression': regression_metrics,
            'overall_score': 0.85
        }

        report = metrics_evaluator.generate_metrics_report(metrics)

        assert isinstance(report, str)
        assert 'LEGAL CLAUSE RISK ASSESSMENT' in report
        assert 'Classification Performance' in report
        assert 'Regression Performance' in report
        assert 'Overall Performance Score' in report

    def test_legal_specific_metrics(self, sample_config, mock_evaluation_data):
        """Test legal domain-specific metrics."""
        metrics_evaluator = LegalRiskMetrics(sample_config)

        # Test high-risk detection metrics
        y_true = np.array([0, 1, 2, 2, 1, 0, 2])  # Mix of classes with high-risk (2)
        y_pred = np.array([0, 1, 2, 1, 1, 0, 2])  # Some predictions

        legal_metrics = metrics_evaluator._calculate_legal_classification_metrics(y_true, y_pred)

        assert 'high_risk_precision' in legal_metrics
        assert 'high_risk_recall' in legal_metrics
        assert 'high_risk_f1' in legal_metrics
        assert 'conservative_bias' in legal_metrics
        assert 'over_prediction_rate' in legal_metrics
        assert 'under_prediction_rate' in legal_metrics

    def test_target_compliance_checking(self, sample_config):
        """Test target compliance checking."""
        metrics_evaluator = LegalRiskMetrics(sample_config)

        # Test classification targets
        good_metrics = {'f1_macro': 0.85, 'recall_high_risk': 0.90}
        targets = metrics_evaluator._check_classification_targets(good_metrics)
        assert targets['meets_f1_target'] is True
        assert targets['meets_recall_target'] is True

        poor_metrics = {'f1_macro': 0.70, 'recall_high_risk': 0.80}
        targets = metrics_evaluator._check_classification_targets(poor_metrics)
        assert targets['meets_f1_target'] is False
        assert targets['meets_recall_target'] is False

        # Test regression targets
        good_regr_metrics = {'mae': 1.0}
        targets = metrics_evaluator._check_regression_targets(good_regr_metrics)
        assert targets['meets_mae_target'] is True

        poor_regr_metrics = {'mae': 2.0}
        targets = metrics_evaluator._check_regression_targets(poor_regr_metrics)
        assert targets['meets_mae_target'] is False

    def test_create_metrics_evaluator_factory(self, sample_config):
        """Test metrics evaluator factory function."""
        evaluator = create_metrics_evaluator(sample_config)
        assert isinstance(evaluator, LegalRiskMetrics)
        assert evaluator.config == sample_config


class TestTrainingIntegration:
    """Integration tests for training components."""

    @pytest.mark.integration
    @patch('legal_clause_risk_scorer.training.trainer.mlflow')
    def test_training_evaluation_integration(
        self,
        mock_mlflow,
        sample_config,
        mock_evaluation_data
    ):
        """Test integration between trainer and metrics evaluator."""
        with patch('legal_clause_risk_scorer.training.trainer.LegalRiskMetrics') as mock_metrics_class, \
             patch('torch.cuda.is_available', return_value=False):

            # Mock metrics evaluator
            mock_metrics = Mock()
            mock_metrics.evaluate_multi_task.return_value = {
                'classification': {'f1_macro': 0.85},
                'regression': {'mae': 1.1},
                'overall_score': 0.82
            }
            mock_metrics_class.return_value = mock_metrics

            # Create mock model and dataset
            mock_model = Mock()
            mock_model.config = sample_config
            mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
            mock_model.to.return_value = mock_model

            mock_dataset = Mock()
            mock_dataset.__len__.return_value = 5

            trainer = LegalRiskTrainer(
                config=sample_config,
                model=mock_model,
                train_dataset=mock_dataset,
                val_dataset=mock_dataset
            )

            # Verify metrics evaluator was created
            assert trainer.metrics_evaluator == mock_metrics
            mock_metrics_class.assert_called_once_with(sample_config)

    @pytest.mark.slow
    @patch('legal_clause_risk_scorer.training.trainer.mlflow')
    def test_full_training_loop_simulation(
        self,
        mock_mlflow,
        sample_config,
        temp_dir
    ):
        """Test simulation of full training loop."""
        with patch('legal_clause_risk_scorer.training.trainer.LegalRiskMetrics'), \
             patch('torch.cuda.is_available', return_value=False):

            # Configure for quick training
            sample_config.set('training.num_epochs', 1)
            sample_config.set('training.batch_size', 2)
            sample_config.set('training.early_stopping_patience', 1)
            sample_config.set('paths.models_dir', str(temp_dir))

            # Create mock components
            mock_model = Mock()
            mock_model.config = sample_config
            mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
            mock_model.to.return_value = mock_model
            mock_model.state_dict.return_value = {'test': torch.tensor([1, 2, 3])}

            # Mock forward pass
            mock_outputs = {
                'loss': torch.tensor(0.5, requires_grad=True),
                'classification_loss': torch.tensor(0.3),
                'regression_loss': torch.tensor(0.2),
                'logits': torch.randn(2, 3),
                'risk_scores': torch.randn(2, 1)
            }
            mock_model.return_value = mock_outputs
            mock_model.__call__.return_value = mock_outputs

            # Create small mock dataset
            mock_dataset = []
            for i in range(4):  # 4 samples total
                mock_dataset.append({
                    'input_ids': torch.randint(0, 1000, (128,)),
                    'attention_mask': torch.ones(128),
                    'labels': torch.tensor(i % 3),
                    'risk_scores': torch.tensor(float(i + 1))
                })

            class SimpleMockDataset:
                def __init__(self, data):
                    self.data = data

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    return self.data[idx]

            train_dataset = SimpleMockDataset(mock_dataset)
            val_dataset = SimpleMockDataset(mock_dataset[:2])

            trainer = LegalRiskTrainer(
                config=sample_config,
                model=mock_model,
                train_dataset=train_dataset,
                val_dataset=val_dataset
            )

            # This would run training in practice
            # Here we just verify the setup worked
            assert trainer is not None
            assert len(trainer.train_loader) > 0
            assert len(trainer.val_loader) > 0