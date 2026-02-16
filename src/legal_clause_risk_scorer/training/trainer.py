"""Comprehensive training system for legal clause risk assessment.

This module provides a full-featured training system with MLflow integration,
early stopping, checkpointing, and advanced training strategies.
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding
)
import numpy as np
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from ..utils.config import Config
from ..models.model import LegalClauseRiskModel
from ..evaluation.metrics import LegalRiskMetrics
from ..data.loader import LegalDataLoader


logger = logging.getLogger(__name__)


class LegalRiskDataset(Dataset):
    """PyTorch Dataset for legal risk assessment."""

    def __init__(self, dataset, tokenizer):
        """Initialize the dataset.

        Args:
            dataset: HuggingFace dataset
            tokenizer: Tokenizer for text processing
        """
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item at index.

        Args:
            idx: Index

        Returns:
            Dictionary with model inputs
        """
        item = self.dataset[idx]

        # Convert to tensors
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long),
            'risk_scores': torch.tensor(item['risk_scores'], dtype=torch.float)
        }


class LegalRiskTrainer:
    """Comprehensive trainer for legal clause risk assessment models.

    This trainer provides full training capabilities including:
    - Multi-task learning (classification + regression)
    - MLflow experiment tracking
    - Early stopping and checkpointing
    - Advanced optimization strategies
    - Comprehensive evaluation and logging

    Attributes:
        config: Configuration object
        model: Model to train
        device: Training device (CPU/GPU)
        metrics_evaluator: Metrics evaluation instance
        optimizer: Model optimizer
        scheduler: Learning rate scheduler
        best_score: Best validation score achieved
        early_stopping_counter: Counter for early stopping
    """

    def __init__(
        self,
        config: Config,
        model: LegalClauseRiskModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None
    ):
        """Initialize the trainer.

        Args:
            config: Configuration object
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset (optional)
        """
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)

        # Training state
        self.best_score = 0.0
        self.early_stopping_counter = 0
        self.global_step = 0
        self.epoch = 0

        # Create data loaders (must come before scheduler which needs train_loader length)
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(val_dataset, shuffle=False)
        if test_dataset is not None:
            self.test_loader = self._create_dataloader(test_dataset, shuffle=False)

        # Initialize components
        self.metrics_evaluator = LegalRiskMetrics(config)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Setup MLflow
        self._setup_mlflow()

        # Create output directories
        self.output_dir = Path(config.get('paths.models_dir', 'models'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Trainer initialized with device: {self.device}")

    def train(self) -> Dict[str, Any]:
        """Execute full training loop.

        Returns:
            Training results and final metrics
        """
        logger.info("Starting training...")

        try:
            # Start MLflow run
            with mlflow.start_run():
                # Log configuration
                self._log_config()

                # Log model architecture
                self._log_model_info()

                # Training loop
                training_results = self._training_loop()

                # Final evaluation
                if self.test_dataset is not None:
                    test_results = self.evaluate(self.test_loader, "test")
                    training_results['test_metrics'] = test_results

                # Log final scalar results to MLflow
                final_scalars = {
                    'best_score': training_results.get('best_score', 0.0),
                    'total_epochs': float(training_results.get('total_epochs', 0)),
                }
                if 'test_metrics' in training_results:
                    tm = training_results['test_metrics']
                    if 'classification' in tm:
                        final_scalars['test_f1'] = tm['classification'].get('f1_macro', 0.0)
                    if 'regression' in tm:
                        final_scalars['test_mae'] = tm['regression'].get('mae', 0.0)
                    if 'overall_score' in tm:
                        final_scalars['test_overall'] = tm['overall_score']
                mlflow.log_metrics(final_scalars)

                # Save final model
                final_model_path = self.output_dir / "final_model.pth"
                self._save_checkpoint(final_model_path, is_best=False)

                logger.info("Training completed successfully")
                return training_results

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _training_loop(self) -> Dict[str, Any]:
        """Execute the main training loop.

        Returns:
            Training results
        """
        num_epochs = self.config.get('training.num_epochs', 10)
        early_stopping_patience = self.config.get('training.early_stopping_patience', 3)

        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_mae': []
        }

        for epoch in range(num_epochs):
            self.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Training phase
            train_metrics = self._train_epoch()
            training_history['train_loss'].append(train_metrics['loss'])

            # Validation phase
            val_metrics = self.evaluate(self.val_loader, "validation")
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_f1'].append(val_metrics['classification']['f1_macro'])
            training_history['val_mae'].append(val_metrics['regression']['mae'])

            # Log epoch metrics
            epoch_metrics = {
                f'epoch_{epoch + 1}/train_loss': train_metrics['loss'],
                f'epoch_{epoch + 1}/val_loss': val_metrics['loss'],
                f'epoch_{epoch + 1}/val_f1': val_metrics['classification']['f1_macro'],
                f'epoch_{epoch + 1}/val_mae': val_metrics['regression']['mae']
            }
            mlflow.log_metrics(epoch_metrics, step=epoch)

            # Check for improvement
            current_score = val_metrics['overall_score']
            is_best = current_score > self.best_score

            if is_best:
                self.best_score = current_score
                self.early_stopping_counter = 0

                # Save best model
                best_model_path = self.output_dir / "best_model.pth"
                self._save_checkpoint(best_model_path, is_best=True)

                logger.info(f"New best model saved! Score: {current_score:.4f}")

            else:
                self.early_stopping_counter += 1

            # Early stopping check
            if self.early_stopping_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Save regular checkpoint
            if (epoch + 1) % self.config.get('training.save_steps', 1000) == 0:
                checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                self._save_checkpoint(checkpoint_path, is_best=False)

        return {
            'best_score': self.best_score,
            'total_epochs': epoch + 1,
            'training_history': training_history,
            'early_stopped': self.early_stopping_counter >= early_stopping_patience
        }

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Training metrics for the epoch
        """
        self.model.train()

        total_loss = 0.0
        total_classification_loss = 0.0
        total_regression_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Training Epoch {self.epoch + 1}",
            leave=False
        )

        for batch in progress_bar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(**batch)

            loss = outputs['loss']
            classification_loss = outputs.get('classification_loss', 0)
            regression_loss = outputs.get('regression_loss', 0)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            max_grad_norm = self.config.get('training.max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            if classification_loss:
                total_classification_loss += classification_loss.item()
            if regression_loss:
                total_regression_loss += regression_loss.item()
            num_batches += 1

            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

            # Log step metrics
            if self.global_step % self.config.get('training.logging_steps', 100) == 0:
                step_metrics = {
                    'step_loss': loss.item(),
                    'step_lr': self.optimizer.param_groups[0]["lr"]
                }
                mlflow.log_metrics(step_metrics, step=self.global_step)

        # Calculate epoch averages
        epoch_metrics = {
            'loss': total_loss / num_batches,
            'classification_loss': total_classification_loss / num_batches,
            'regression_loss': total_regression_loss / num_batches
        }

        return epoch_metrics

    def evaluate(self, dataloader: DataLoader, split_name: str) -> Dict[str, Any]:
        """Evaluate model on a dataset.

        Args:
            dataloader: DataLoader for evaluation
            split_name: Name of the dataset split

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        all_predictions_class = []
        all_predictions_score = []
        all_labels_class = []
        all_labels_score = []
        all_probabilities = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)

                if outputs.get('loss') is not None:
                    total_loss += outputs['loss'].item()

                # Get predictions
                logits = outputs['logits']
                risk_scores = outputs['risk_scores']

                # Convert to predictions
                probabilities = torch.softmax(logits, dim=-1)
                class_predictions = torch.argmax(logits, dim=-1)

                # Store results
                all_predictions_class.extend(class_predictions.cpu().numpy())
                # Use squeeze(-1) to only remove the last dim, not the batch dim
                score_preds = risk_scores.squeeze(-1).cpu().numpy()
                # Replace NaN with neutral score 5.0
                score_preds = np.nan_to_num(score_preds, nan=5.0)
                all_predictions_score.extend(score_preds)
                all_probabilities.extend(probabilities.cpu().numpy())

                all_labels_class.extend(batch['labels'].cpu().numpy())
                all_labels_score.extend(batch['risk_scores'].cpu().numpy())

                num_batches += 1

        # Calculate metrics
        eval_metrics = self.metrics_evaluator.evaluate_multi_task(
            classification_true=all_labels_class,
            classification_pred=all_predictions_class,
            regression_true=all_labels_score,
            regression_pred=all_predictions_score,
            classification_prob=np.array(all_probabilities)
        )

        eval_metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0.0

        logger.info(f"{split_name} evaluation completed:")
        logger.info(f"  Loss: {eval_metrics['loss']:.4f}")
        logger.info(f"  F1 Score: {eval_metrics['classification']['f1_macro']:.4f}")
        logger.info(f"  MAE: {eval_metrics['regression']['mae']:.4f}")
        logger.info(f"  Overall Score: {eval_metrics['overall_score']:.4f}")

        return eval_metrics

    def _setup_device(self) -> torch.device:
        """Setup training device.

        Returns:
            Training device
        """
        use_cuda = self.config.get('training.use_cuda', True)

        if use_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU device")

        return device

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer.

        Returns:
            Optimizer instance
        """
        optimizer_name = self.config.get('training.optimizer', 'AdamW')
        learning_rate = self.config.get('training.learning_rate', 2e-5)
        weight_decay = self.config.get('training.weight_decay', 0.01)

        if optimizer_name.lower() == 'adamw':
            optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        logger.info(f"Created optimizer: {optimizer_name} (lr={learning_rate})")
        return optimizer

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler.

        Returns:
            Scheduler instance or None
        """
        scheduler_name = self.config.get('training.scheduler', 'linear')

        if scheduler_name.lower() == 'linear':
            num_epochs = self.config.get('training.num_epochs', 10)
            warmup_steps = self.config.get('training.warmup_steps', 500)

            total_steps = len(self.train_loader) * num_epochs

            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            logger.info(f"Created linear scheduler: warmup={warmup_steps}, total={total_steps}")
            return scheduler

        elif scheduler_name.lower() == 'none':
            return None

        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create data loader.

        Args:
            dataset: Dataset to wrap
            shuffle: Whether to shuffle data

        Returns:
            DataLoader instance
        """
        batch_size = self.config.get('training.batch_size', 16)
        num_workers = self.config.get('training.dataloader_num_workers', 4)

        # Data is already padded to max_length during tokenization,
        # so default collation (stacking) is sufficient
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        tracking_uri = self.config.get('mlflow.tracking_uri', 'mlruns')
        experiment_name = self.config.get('mlflow.experiment_name', 'legal_clause_risk_scorer')

        mlflow.set_tracking_uri(tracking_uri)

        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Could not setup MLflow experiment: {e}")

        mlflow.set_experiment(experiment_name)

    def _log_config(self) -> None:
        """Log configuration to MLflow."""
        config_dict = self.config.to_dict()

        # Flatten nested configuration for MLflow, only logging scalar values
        flattened_config = {}
        for section, values in config_dict.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, (str, int, float, bool)):
                        flattened_config[f"{section}_{key}"] = value
            elif isinstance(values, (str, int, float, bool)):
                flattened_config[section] = values

        try:
            mlflow.log_params(flattened_config)
        except Exception as e:
            logger.warning(f"Could not log all config params to MLflow: {e}")

    def _log_model_info(self) -> None:
        """Log model information to MLflow."""
        from ..models.model import count_parameters

        param_counts = count_parameters(self.model)

        mlflow.log_params({
            'model_total_params': param_counts['total_parameters'],
            'model_trainable_params': param_counts['trainable_parameters'],
            'model_frozen_params': param_counts['frozen_parameters']
        })

    def _save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': self.best_score,
            'config': self.config.to_dict()
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

        # Log model to MLflow
        if is_best:
            mlflow.pytorch.log_model(
                self.model,
                "best_model",
                registered_model_name="legal_clause_risk_scorer"
            )

        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_score = checkpoint.get('best_score', 0.0)

        logger.info(f"Checkpoint loaded: {path}")


def create_trainer(
    config: Config,
    model: LegalClauseRiskModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None
) -> LegalRiskTrainer:
    """Factory function to create a trainer instance.

    Args:
        config: Configuration object
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset (optional)

    Returns:
        Initialized trainer
    """
    return LegalRiskTrainer(config, model, train_dataset, val_dataset, test_dataset)