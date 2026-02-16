#!/usr/bin/env python3
"""Training script for legal clause risk assessment model.

This script orchestrates the complete training pipeline including data loading,
model initialization, training execution, and result logging.

Usage:
    python scripts/train.py [--config CONFIG_PATH] [--resume CHECKPOINT_PATH]

Examples:
    # Train with default configuration
    python scripts/train.py

    # Train with custom configuration
    python scripts/train.py --config configs/custom.yaml

    # Resume training from checkpoint
    python scripts/train.py --resume models/checkpoint_epoch_5.pth
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legal_clause_risk_scorer.utils.config import (
    load_config,
    setup_logging,
    set_random_seeds,
    create_output_directories
)
from legal_clause_risk_scorer.data.loader import LegalDataLoader
from legal_clause_risk_scorer.models.model import LegalClauseRiskModel
from legal_clause_risk_scorer.training.trainer import LegalRiskTrainer, LegalRiskDataset
from legal_clause_risk_scorer.evaluation.metrics import LegalRiskMetrics


logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train legal clause risk assessment model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: configs/default.yaml)"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for models and logs"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override MLflow experiment name"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run setup and validation without training"
    )

    return parser.parse_args()


def validate_environment() -> None:
    """Validate training environment and dependencies.

    Raises:
        RuntimeError: If environment validation fails
    """
    import torch
    import transformers
    import datasets

    logger.info("Validating training environment...")

    # Check PyTorch installation
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
        logger.info(f"Current device: {torch.cuda.get_device_name()}")
    else:
        logger.info("CUDA not available, using CPU")

    # Check transformers version
    logger.info(f"Transformers version: {transformers.__version__}")

    # Check datasets version
    logger.info(f"Datasets version: {datasets.__version__}")

    logger.info("Environment validation completed")


def load_datasets(config):
    """Load and prepare datasets for training.

    Args:
        config: Configuration object

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, data_loader)
    """
    logger.info("Loading datasets...")

    # Create data loader
    data_loader = LegalDataLoader(config)

    # Load and combine datasets
    dataset_dict = data_loader.combine_datasets()

    # Tokenize datasets
    tokenized_datasets = data_loader.tokenize_dataset(dataset_dict)

    # Create PyTorch datasets
    train_dataset = LegalRiskDataset(tokenized_datasets['train'], data_loader.tokenizer)
    val_dataset = LegalRiskDataset(tokenized_datasets['validation'], data_loader.tokenizer)
    test_dataset = LegalRiskDataset(tokenized_datasets['test'], data_loader.tokenizer)

    # Log dataset statistics
    stats = data_loader.get_dataset_statistics(dataset_dict)
    logger.info("Dataset statistics:")
    for split_name, split_stats in stats.items():
        logger.info(f"  {split_name}: {split_stats['total_samples']} samples")
        logger.info(f"    Avg text length: {split_stats['avg_text_length']:.1f}")
        logger.info(f"    Avg risk score: {split_stats['avg_risk_score']:.2f}")

    return train_dataset, val_dataset, test_dataset, data_loader


def initialize_model(config, tokenizer) -> LegalClauseRiskModel:
    """Initialize model for training.

    Args:
        config: Configuration object
        tokenizer: Tokenizer instance

    Returns:
        Initialized model
    """
    logger.info("Initializing model...")

    # Create model
    model = LegalClauseRiskModel(config)

    # Store tokenizer reference
    model.tokenizer = tokenizer

    # Log model information
    from legal_clause_risk_scorer.models.model import count_parameters
    param_counts = count_parameters(model)

    logger.info(f"Model initialized:")
    logger.info(f"  Total parameters: {param_counts['total_parameters']:,}")
    logger.info(f"  Trainable parameters: {param_counts['trainable_parameters']:,}")
    logger.info(f"  Frozen parameters: {param_counts['frozen_parameters']:,}")

    return model


def train_model(config, model, train_dataset, val_dataset, test_dataset, resume_path=None):
    """Execute model training.

    Args:
        config: Configuration object
        model: Model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        resume_path: Path to checkpoint for resuming training

    Returns:
        Training results
    """
    logger.info("Starting model training...")

    # Create trainer
    trainer = LegalRiskTrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )

    # Resume from checkpoint if specified
    if resume_path:
        logger.info(f"Resuming training from: {resume_path}")
        trainer.load_checkpoint(Path(resume_path))

    # Execute training
    results = trainer.train()

    logger.info("Training completed!")
    logger.info(f"Best validation score: {results['best_score']:.4f}")
    logger.info(f"Total epochs: {results['total_epochs']}")

    return results


def save_training_artifacts(config, results, output_dir):
    """Save training artifacts and results.

    Args:
        config: Configuration object
        results: Training results
        output_dir: Output directory
    """
    logger.info("Saving training artifacts...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save final configuration
    config.save(output_dir / "final_config.yaml")

    # Save training results
    import json
    import numpy as np

    def make_serializable(obj):
        """Recursively convert numpy types to native Python types for JSON."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        return obj

    with open(output_dir / "training_results.json", 'w') as f:
        json.dump(make_serializable(results), f, indent=2)

    logger.info(f"Training artifacts saved to: {output_dir}")


def main() -> int:
    """Main training function.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse arguments
        args = parse_arguments()

        # Load configuration
        config = load_config(args.config)

        # Override configuration with command line arguments
        if args.output_dir:
            config.set('paths.models_dir', args.output_dir)

        if args.experiment_name:
            config.set('mlflow.experiment_name', args.experiment_name)

        if args.debug:
            config.set('logging.level', 'DEBUG')

        # Setup logging
        setup_logging(config)

        # Set random seeds
        set_random_seeds(config)

        # Create output directories
        create_output_directories(config)

        logger.info("=" * 60)
        logger.info("LEGAL CLAUSE RISK SCORER - TRAINING")
        logger.info("=" * 60)

        # Validate environment
        validate_environment()

        if args.dry_run:
            logger.info("Dry run completed successfully")
            return 0

        # Load datasets
        train_dataset, val_dataset, test_dataset, data_loader = load_datasets(config)

        # Initialize model
        model = initialize_model(config, data_loader.tokenizer)

        # Train model
        results = train_model(
            config=config,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            resume_path=args.resume
        )

        # Save artifacts
        output_dir = config.get('paths.models_dir', 'models')
        save_training_artifacts(config, results, output_dir)

        # Print final results
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Best validation score: {results['best_score']:.4f}")
        print(f"Total epochs trained: {results['total_epochs']}")

        if 'test_metrics' in results:
            test_f1 = results['test_metrics']['classification']['f1_macro']
            test_mae = results['test_metrics']['regression']['mae']
            print(f"Test F1 score: {test_f1:.4f}")
            print(f"Test MAE: {test_mae:.4f}")

        print(f"Models saved in: {output_dir}")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)