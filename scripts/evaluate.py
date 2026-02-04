#!/usr/bin/env python3
"""Evaluation script for legal clause risk assessment model.

This script provides comprehensive evaluation capabilities including model testing,
inference on new data, and generation of detailed evaluation reports.

Usage:
    python scripts/evaluate.py --model MODEL_PATH [--data DATA_PATH] [--config CONFIG_PATH]

Examples:
    # Evaluate trained model on test set
    python scripts/evaluate.py --model models/best_model.pth

    # Evaluate on custom data
    python scripts/evaluate.py --model models/best_model.pth --data custom_clauses.csv

    # Generate detailed report with visualizations
    python scripts/evaluate.py --model models/best_model.pth --report --visualizations
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legal_clause_risk_scorer.utils.config import load_config, setup_logging
from legal_clause_risk_scorer.data.loader import LegalDataLoader
from legal_clause_risk_scorer.data.preprocessing import LegalTextPreprocessor
from legal_clause_risk_scorer.models.model import LegalClauseRiskModel
from legal_clause_risk_scorer.training.trainer import LegalRiskDataset
from legal_clause_risk_scorer.evaluation.metrics import LegalRiskMetrics


logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Evaluate legal clause risk assessment model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: configs/default.yaml)"
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to custom data file (CSV with 'text' column)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for evaluation results"
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed evaluation report"
    )

    parser.add_argument(
        "--visualizations",
        action="store_true",
        help="Generate evaluation visualizations"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    return parser.parse_args()


class ModelEvaluator:
    """Comprehensive model evaluator for legal clause risk assessment.

    This class provides evaluation capabilities for trained models including
    test set evaluation, custom data inference, and report generation.

    Attributes:
        config: Configuration object
        model: Loaded model
        tokenizer: Tokenizer instance
        metrics_evaluator: Metrics evaluation instance
        device: Evaluation device
    """

    def __init__(self, config, model_path: str):
        """Initialize the evaluator.

        Args:
            config: Configuration object
            model_path: Path to the trained model
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Initialize components
        self.data_loader = LegalDataLoader(config)
        self.tokenizer = self.data_loader.tokenizer
        self.preprocessor = LegalTextPreprocessor(config)
        self.metrics_evaluator = LegalRiskMetrics(config)

        logger.info(f"Model evaluator initialized on device: {self.device}")

    def _load_model(self, model_path: str) -> LegalClauseRiskModel:
        """Load trained model from checkpoint.

        Args:
            model_path: Path to model checkpoint

        Returns:
            Loaded model

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            logger.info(f"Loading model from: {model_path}")

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Create model instance
            model = LegalClauseRiskModel(self.config)

            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Assume the file contains just the state dict
                model.load_state_dict(checkpoint)

            logger.info("Model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def evaluate_test_set(self) -> Dict[str, any]:
        """Evaluate model on the standard test set.

        Returns:
            Evaluation results
        """
        logger.info("Evaluating on test set...")

        # Load test dataset
        dataset_dict = self.data_loader.combine_datasets()
        tokenized_datasets = self.data_loader.tokenize_dataset(dataset_dict)
        test_dataset = LegalRiskDataset(tokenized_datasets['test'], self.tokenizer)

        # Run evaluation
        results = self._evaluate_dataset(test_dataset, "test_set")

        logger.info(f"Test set evaluation completed:")
        logger.info(f"  F1 Score: {results['classification']['f1_macro']:.4f}")
        logger.info(f"  MAE: {results['regression']['mae']:.4f}")
        logger.info(f"  Overall Score: {results['overall_score']:.4f}")

        return results

    def evaluate_custom_data(self, data_path: str, max_samples: Optional[int] = None) -> Dict[str, any]:
        """Evaluate model on custom data file.

        Args:
            data_path: Path to CSV file with 'text' column
            max_samples: Maximum number of samples to process

        Returns:
            Evaluation results and predictions
        """
        logger.info(f"Evaluating on custom data: {data_path}")

        # Load data
        df = pd.read_csv(data_path)

        if 'text' not in df.columns:
            raise ValueError("Data file must contain 'text' column")

        texts = df['text'].tolist()

        if max_samples:
            texts = texts[:max_samples]

        logger.info(f"Processing {len(texts)} samples...")

        # Generate predictions
        predictions = self.predict_batch(texts)

        # Create results dictionary
        results = {
            'predictions': predictions,
            'num_samples': len(texts),
            'statistics': self._calculate_prediction_statistics(predictions)
        }

        # If ground truth labels are available, calculate metrics
        if 'risk_category' in df.columns or 'risk_score' in df.columns:
            logger.info("Ground truth found, calculating metrics...")

            # Prepare ground truth
            y_true_class = []
            y_true_score = []

            if 'risk_category' in df.columns:
                category_map = {'low_risk': 0, 'medium_risk': 1, 'high_risk': 2}
                y_true_class = [category_map.get(cat, 1) for cat in df['risk_category'].tolist()[:len(texts)]]

            if 'risk_score' in df.columns:
                y_true_score = df['risk_score'].tolist()[:len(texts)]

            # Calculate metrics
            if y_true_class and y_true_score:
                y_pred_class = [self._category_to_int(pred['risk_category']) for pred in predictions]
                y_pred_score = [pred['risk_score'] for pred in predictions]

                metrics = self.metrics_evaluator.evaluate_multi_task(
                    classification_true=y_true_class,
                    classification_pred=y_pred_class,
                    regression_true=y_true_score,
                    regression_pred=y_pred_score
                )

                results['metrics'] = metrics

        return results

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Union[str, float]]]:
        """Generate predictions for a batch of texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            List of predictions
        """
        logger.info(f"Generating predictions for {len(texts)} texts...")

        # Preprocess texts
        cleaned_texts = [self.preprocessor.clean_legal_text(text) for text in texts]

        predictions = []

        with torch.no_grad():
            for i in tqdm(range(0, len(cleaned_texts), batch_size), desc="Processing batches"):
                batch_texts = cleaned_texts[i:i + batch_size]

                # Tokenize batch
                tokenized = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )

                # Move to device
                input_ids = tokenized['input_ids'].to(self.device)
                attention_mask = tokenized['attention_mask'].to(self.device)

                # Generate predictions
                outputs = self.model.predict_risk(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Process outputs
                for j in range(len(batch_texts)):
                    pred = {
                        'text': texts[i + j],
                        'cleaned_text': batch_texts[j],
                        'risk_category': outputs['risk_categories'][j],
                        'risk_score': float(outputs['risk_scores'][j]),
                        'confidence': float(outputs['confidence_scores'][j]),
                        'category_probabilities': {
                            'low_risk': float(outputs['category_probabilities'][j][0]),
                            'medium_risk': float(outputs['category_probabilities'][j][1]),
                            'high_risk': float(outputs['category_probabilities'][j][2])
                        }
                    }
                    predictions.append(pred)

        logger.info("Predictions generated successfully")
        return predictions

    def predict_single(self, text: str) -> Dict[str, Union[str, float]]:
        """Generate prediction for a single text.

        Args:
            text: Input text

        Returns:
            Prediction dictionary
        """
        predictions = self.predict_batch([text], batch_size=1)
        return predictions[0]

    def _evaluate_dataset(self, dataset: LegalRiskDataset, split_name: str) -> Dict[str, any]:
        """Evaluate model on a PyTorch dataset.

        Args:
            dataset: Dataset to evaluate
            split_name: Name of the dataset split

        Returns:
            Evaluation metrics
        """
        from torch.utils.data import DataLoader

        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0  # Set to 0 for evaluation to avoid multiprocessing issues
        )

        all_predictions_class = []
        all_predictions_score = []
        all_labels_class = []
        all_labels_score = []
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)

                # Get predictions
                logits = outputs['logits']
                risk_scores = outputs['risk_scores']

                # Convert to predictions
                probabilities = torch.softmax(logits, dim=-1)
                class_predictions = torch.argmax(logits, dim=-1)

                # Store results
                all_predictions_class.extend(class_predictions.cpu().numpy())
                all_predictions_score.extend(risk_scores.squeeze().cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                all_labels_class.extend(batch['labels'].cpu().numpy())
                all_labels_score.extend(batch['risk_scores'].cpu().numpy())

        # Calculate metrics
        metrics = self.metrics_evaluator.evaluate_multi_task(
            classification_true=all_labels_class,
            classification_pred=all_predictions_class,
            regression_true=all_labels_score,
            regression_pred=all_predictions_score,
            classification_prob=np.array(all_probabilities)
        )

        return metrics

    def _calculate_prediction_statistics(self, predictions: List[Dict]) -> Dict[str, any]:
        """Calculate statistics for a set of predictions.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Statistics dictionary
        """
        risk_scores = [pred['risk_score'] for pred in predictions]
        risk_categories = [pred['risk_category'] for pred in predictions]
        confidences = [pred['confidence'] for pred in predictions]

        # Calculate category distribution
        category_counts = {}
        for category in risk_categories:
            category_counts[category] = category_counts.get(category, 0) + 1

        category_percentages = {
            cat: (count / len(predictions)) * 100
            for cat, count in category_counts.items()
        }

        return {
            'risk_score_mean': np.mean(risk_scores),
            'risk_score_std': np.std(risk_scores),
            'risk_score_min': np.min(risk_scores),
            'risk_score_max': np.max(risk_scores),
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'category_counts': category_counts,
            'category_percentages': category_percentages,
            'high_risk_percentage': category_percentages.get('high_risk', 0.0)
        }

    def _category_to_int(self, category: str) -> int:
        """Convert category string to integer.

        Args:
            category: Category string

        Returns:
            Integer representation
        """
        category_map = {'low_risk': 0, 'medium_risk': 1, 'high_risk': 2}
        return category_map.get(category, 1)

    def generate_evaluation_report(
        self,
        results: Dict[str, any],
        output_dir: str,
        include_visualizations: bool = False
    ) -> str:
        """Generate comprehensive evaluation report.

        Args:
            results: Evaluation results
            output_dir: Output directory
            include_visualizations: Whether to include visualizations

        Returns:
            Path to generated report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "evaluation_report.md"

        with open(report_path, 'w') as f:
            f.write("# Legal Clause Risk Assessment - Evaluation Report\n\n")

            # Model information
            f.write("## Model Information\n")
            f.write(f"- Model: {self.config.get('model.base_model')}\n")
            f.write(f"- Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- Device: {self.device}\n\n")

            # Results summary
            if 'metrics' in results:
                metrics = results['metrics']
                f.write("## Performance Metrics\n\n")

                # Classification metrics
                if 'classification' in metrics:
                    class_metrics = metrics['classification']
                    f.write("### Classification Performance\n")
                    f.write(f"- **Accuracy**: {class_metrics.get('accuracy', 0):.3f}\n")
                    f.write(f"- **Precision (macro)**: {class_metrics.get('precision_macro', 0):.3f}\n")
                    f.write(f"- **Recall (macro)**: {class_metrics.get('recall_macro', 0):.3f}\n")
                    f.write(f"- **F1 Score (macro)**: {class_metrics.get('f1_macro', 0):.3f}\n\n")

                # Regression metrics
                if 'regression' in metrics:
                    regr_metrics = metrics['regression']
                    f.write("### Regression Performance\n")
                    f.write(f"- **MAE**: {regr_metrics.get('mae', 0):.3f}\n")
                    f.write(f"- **RMSE**: {regr_metrics.get('rmse', 0):.3f}\n")
                    f.write(f"- **R² Score**: {regr_metrics.get('r2', 0):.3f}\n\n")

                # Overall score
                f.write(f"### Overall Performance Score: {metrics.get('overall_score', 0):.3f}\n\n")

            # Prediction statistics
            if 'statistics' in results:
                stats = results['statistics']
                f.write("## Prediction Statistics\n")
                f.write(f"- **Number of samples**: {results.get('num_samples', 0)}\n")
                f.write(f"- **Average risk score**: {stats['risk_score_mean']:.2f} ± {stats['risk_score_std']:.2f}\n")
                f.write(f"- **Risk score range**: {stats['risk_score_min']:.2f} - {stats['risk_score_max']:.2f}\n")
                f.write(f"- **Average confidence**: {stats['confidence_mean']:.2f} ± {stats['confidence_std']:.2f}\n")
                f.write(f"- **High-risk percentage**: {stats['high_risk_percentage']:.1f}%\n\n")

                # Category distribution
                f.write("### Risk Category Distribution\n")
                for category, percentage in stats['category_percentages'].items():
                    count = stats['category_counts'][category]
                    f.write(f"- **{category}**: {count} samples ({percentage:.1f}%)\n")

            f.write("\n---\n")
            f.write("*Report generated by Legal Clause Risk Scorer*\n")

        # Generate visualizations if requested
        if include_visualizations and 'metrics' in results:
            self._generate_visualizations(results['metrics'], output_dir)

        logger.info(f"Evaluation report saved to: {report_path}")
        return str(report_path)

    def _generate_visualizations(self, metrics: Dict[str, any], output_dir: Path) -> None:
        """Generate evaluation visualizations.

        Args:
            metrics: Metrics dictionary
            output_dir: Output directory
        """
        logger.info("Generating evaluation visualizations...")

        # Note: The actual visualization generation would depend on having
        # the raw predictions data, which we'd need to modify the evaluation
        # pipeline to collect. For now, we'll create placeholder files.

        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Create placeholder visualization files
        placeholder_files = [
            "confusion_matrix.png",
            "risk_score_distribution.png",
            "confidence_distribution.png",
            "category_performance.png"
        ]

        for filename in placeholder_files:
            placeholder_path = viz_dir / filename
            placeholder_path.touch()

        logger.info(f"Visualization placeholders created in: {viz_dir}")


def main() -> int:
    """Main evaluation function.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse arguments
        args = parse_arguments()

        # Load configuration
        config = load_config(args.config)

        if args.debug:
            config.set('logging.level', 'DEBUG')

        # Setup logging
        setup_logging(config)

        logger.info("=" * 60)
        logger.info("LEGAL CLAUSE RISK SCORER - EVALUATION")
        logger.info("=" * 60)

        # Initialize evaluator
        evaluator = ModelEvaluator(config, args.model)

        # Run evaluation
        if args.data:
            # Evaluate on custom data
            results = evaluator.evaluate_custom_data(args.data, args.max_samples)
            eval_type = "custom_data"
        else:
            # Evaluate on test set
            results = evaluator.evaluate_test_set()
            eval_type = "test_set"

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        results_path = output_dir / f"{eval_type}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Generate report if requested
        if args.report:
            report_path = evaluator.generate_evaluation_report(
                results,
                args.output_dir,
                include_visualizations=args.visualizations
            )
            print(f"Evaluation report generated: {report_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED")
        print("=" * 60)

        if 'metrics' in results:
            metrics = results['metrics']
            if 'classification' in metrics:
                f1_score = metrics['classification']['f1_macro']
                print(f"F1 Score: {f1_score:.4f}")

            if 'regression' in metrics:
                mae = metrics['regression']['mae']
                print(f"MAE: {mae:.4f}")

            print(f"Overall Score: {metrics['overall_score']:.4f}")

        if 'statistics' in results:
            print(f"Samples processed: {results['num_samples']}")
            high_risk_pct = results['statistics']['high_risk_percentage']
            print(f"High-risk clauses: {high_risk_pct:.1f}%")

        print(f"Results saved to: {args.output_dir}")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)