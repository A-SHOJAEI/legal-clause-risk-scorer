"""Evaluation metrics for legal clause risk assessment.

This module provides comprehensive evaluation metrics for both classification
and regression tasks, including legal-domain specific metrics and visualization tools.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn.functional as F

from ..utils.config import Config


logger = logging.getLogger(__name__)


class LegalRiskMetrics:
    """Comprehensive metrics evaluator for legal clause risk assessment.

    This class provides evaluation capabilities for both classification and
    regression tasks with legal-domain specific metrics and visualizations.

    Attributes:
        config: Configuration object
        target_metrics: Target performance metrics from config
        class_names: Names of risk categories
    """

    def __init__(self, config: Config):
        """Initialize the metrics evaluator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.target_metrics = config.get('evaluation.targets', {})
        self.class_names = ['low_risk', 'medium_risk', 'high_risk']

        logger.info("Legal risk metrics evaluator initialized")

    def evaluate_classification(
        self,
        y_true: Union[List[int], np.ndarray, torch.Tensor],
        y_pred: Union[List[int], np.ndarray, torch.Tensor],
        y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """Evaluate classification performance.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)

        Returns:
            Dictionary of classification metrics
        """
        # Convert to numpy arrays
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        if y_prob is not None:
            y_prob = self._to_numpy(y_prob)

        logger.info(f"Evaluating classification: {len(y_true)} samples")

        metrics = {}

        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]

        # ROC AUC if probabilities are provided
        if y_prob is not None:
            try:
                if y_prob.ndim == 1:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                else:
                    # Multi-class classification
                    metrics['roc_auc_macro'] = roc_auc_score(
                        y_true, y_prob, multi_class='ovr', average='macro'
                    )

                    # Per-class ROC AUC
                    y_true_binarized = label_binarize(y_true, classes=list(range(len(self.class_names))))
                    for i, class_name in enumerate(self.class_names):
                        if i < y_prob.shape[1] and i < y_true_binarized.shape[1]:
                            metrics[f'roc_auc_{class_name}'] = roc_auc_score(
                                y_true_binarized[:, i], y_prob[:, i]
                            )

            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")

        # Legal-specific metrics
        metrics.update(self._calculate_legal_classification_metrics(y_true, y_pred))

        # Check against targets
        metrics.update(self._check_classification_targets(metrics))

        logger.info(f"Classification evaluation completed: F1={metrics['f1_macro']:.3f}")
        return metrics

    def evaluate_regression(
        self,
        y_true: Union[List[float], np.ndarray, torch.Tensor],
        y_pred: Union[List[float], np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """Evaluate regression performance for risk scores.

        Args:
            y_true: True risk scores
            y_pred: Predicted risk scores

        Returns:
            Dictionary of regression metrics
        """
        # Convert to numpy arrays
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)

        logger.info(f"Evaluating regression: {len(y_true)} samples")

        metrics = {}

        # Basic regression metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)

        # Additional regression metrics
        metrics['mean_error'] = np.mean(y_pred - y_true)
        metrics['std_error'] = np.std(y_pred - y_true)

        # Percentage-based metrics (avoid division by zero)
        nonzero_mask = y_true != 0
        if np.any(nonzero_mask):
            metrics['mape'] = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
        else:
            metrics['mape'] = 0.0

        # Score range analysis
        metrics['pred_range_min'] = np.min(y_pred)
        metrics['pred_range_max'] = np.max(y_pred)
        metrics['pred_mean'] = np.mean(y_pred)
        metrics['pred_std'] = np.std(y_pred)

        metrics['true_range_min'] = np.min(y_true)
        metrics['true_range_max'] = np.max(y_true)
        metrics['true_mean'] = np.mean(y_true)
        metrics['true_std'] = np.std(y_true)

        # Legal-specific regression metrics
        metrics.update(self._calculate_legal_regression_metrics(y_true, y_pred))

        # Check against targets
        metrics.update(self._check_regression_targets(metrics))

        logger.info(f"Regression evaluation completed: MAE={metrics['mae']:.3f}")
        return metrics

    def evaluate_multi_task(
        self,
        classification_true: Union[List[int], np.ndarray, torch.Tensor],
        classification_pred: Union[List[int], np.ndarray, torch.Tensor],
        regression_true: Union[List[float], np.ndarray, torch.Tensor],
        regression_pred: Union[List[float], np.ndarray, torch.Tensor],
        classification_prob: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Evaluate both classification and regression performance.

        Args:
            classification_true: True classification labels
            classification_pred: Predicted classification labels
            regression_true: True regression scores
            regression_pred: Predicted regression scores
            classification_prob: Classification probabilities (optional)

        Returns:
            Combined metrics dictionary
        """
        logger.info("Evaluating multi-task performance")

        # Evaluate individual tasks
        classification_metrics = self.evaluate_classification(
            classification_true, classification_pred, classification_prob
        )
        regression_metrics = self.evaluate_regression(regression_true, regression_pred)

        # Combine metrics
        combined_metrics = {
            'classification': classification_metrics,
            'regression': regression_metrics
        }

        # Calculate cross-task consistency metrics
        consistency_metrics = self._calculate_task_consistency(
            classification_true, classification_pred,
            regression_true, regression_pred
        )

        combined_metrics['consistency'] = consistency_metrics

        # Overall performance score
        overall_score = self._calculate_overall_score(
            classification_metrics, regression_metrics, consistency_metrics
        )
        combined_metrics['overall_score'] = overall_score

        logger.info(f"Multi-task evaluation completed: Overall score={overall_score:.3f}")
        return combined_metrics

    def _calculate_legal_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate legal-domain specific classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Legal-specific metrics
        """
        metrics = {}

        # High-risk detection metrics (most critical for legal applications)
        high_risk_idx = len(self.class_names) - 1  # Assuming high_risk is last class

        high_risk_true = (y_true == high_risk_idx).astype(int)
        high_risk_pred = (y_pred == high_risk_idx).astype(int)

        if np.sum(high_risk_true) > 0:
            metrics['high_risk_precision'] = precision_score(high_risk_true, high_risk_pred, zero_division=0)
            metrics['high_risk_recall'] = recall_score(high_risk_true, high_risk_pred, zero_division=0)
            metrics['high_risk_f1'] = f1_score(high_risk_true, high_risk_pred, zero_division=0)

        # Conservative prediction bias (prefer higher risk when uncertain)
        over_predictions = np.sum(y_pred > y_true)
        under_predictions = np.sum(y_pred < y_true)
        total_predictions = len(y_true)

        metrics['over_prediction_rate'] = over_predictions / total_predictions
        metrics['under_prediction_rate'] = under_predictions / total_predictions
        metrics['conservative_bias'] = (over_predictions - under_predictions) / total_predictions

        # Risk escalation accuracy (detecting increases in risk)
        # This would be useful for comparing clauses
        metrics['exact_match_rate'] = accuracy_score(y_true, y_pred)

        return metrics

    def _calculate_legal_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate legal-domain specific regression metrics.

        Args:
            y_true: True risk scores
            y_pred: Predicted risk scores

        Returns:
            Legal-specific regression metrics
        """
        metrics = {}

        # Risk threshold analysis
        high_risk_threshold = 7.0
        medium_risk_threshold = 4.0

        # High-risk detection accuracy
        high_risk_true = y_true >= high_risk_threshold
        high_risk_pred = y_pred >= high_risk_threshold

        if np.sum(high_risk_true) > 0:
            metrics['high_risk_detection_accuracy'] = accuracy_score(high_risk_true, high_risk_pred)

        # Risk category alignment
        true_categories = self._scores_to_categories(y_true)
        pred_categories = self._scores_to_categories(y_pred)
        metrics['category_alignment_accuracy'] = accuracy_score(true_categories, pred_categories)

        # Conservative prediction analysis
        conservative_predictions = np.sum(y_pred > y_true)
        liberal_predictions = np.sum(y_pred < y_true)
        total = len(y_true)

        metrics['conservative_prediction_rate'] = conservative_predictions / total
        metrics['liberal_prediction_rate'] = liberal_predictions / total

        # Score distribution analysis
        metrics['prediction_bias'] = np.mean(y_pred - y_true)

        # Extreme value handling
        extreme_high_true = y_true >= 9.0
        extreme_high_pred = y_pred >= 9.0

        if np.sum(extreme_high_true) > 0:
            metrics['extreme_high_detection_rate'] = np.sum(
                extreme_high_true & extreme_high_pred
            ) / np.sum(extreme_high_true)

        return metrics

    def _calculate_task_consistency(
        self,
        class_true: np.ndarray,
        class_pred: np.ndarray,
        score_true: np.ndarray,
        score_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate consistency between classification and regression tasks.

        Args:
            class_true: True classification labels
            class_pred: Predicted classification labels
            score_true: True regression scores
            score_pred: Predicted regression scores

        Returns:
            Consistency metrics
        """
        metrics = {}

        # Ensure numpy arrays
        class_true = self._to_numpy(class_true)
        class_pred = self._to_numpy(class_pred)
        score_true = self._to_numpy(score_true)
        score_pred = self._to_numpy(score_pred)

        # Convert scores to categories for comparison
        score_categories_true = self._scores_to_categories(score_true)
        score_categories_pred = self._scores_to_categories(score_pred)

        # Classification-regression consistency
        metrics['class_score_consistency_true'] = accuracy_score(class_true, score_categories_true)
        metrics['class_score_consistency_pred'] = accuracy_score(class_pred, score_categories_pred)

        # Directional consistency (both tasks should agree on relative risk)
        # Use a random sample of pairs to avoid O(n^2) computation
        direction_agreement = 0
        total_pairs = 0
        max_samples = min(len(class_pred), 500)  # Limit to avoid O(n^2) explosion

        for i in range(max_samples):
            for j in range(i + 1, max_samples):
                total_pairs += 1

                # Check if both tasks agree on which sample is higher risk
                class_agrees = (class_pred[i] > class_pred[j]) == (score_pred[i] > score_pred[j])
                if class_agrees:
                    direction_agreement += 1

        if total_pairs > 0:
            metrics['directional_consistency'] = direction_agreement / total_pairs

        return metrics

    def _calculate_overall_score(
        self,
        class_metrics: Dict[str, float],
        regr_metrics: Dict[str, float],
        consistency_metrics: Dict[str, float]
    ) -> float:
        """Calculate overall performance score.

        Args:
            class_metrics: Classification metrics
            regr_metrics: Regression metrics
            consistency_metrics: Consistency metrics

        Returns:
            Overall performance score (0-1)
        """
        # Weight different components
        class_weight = 0.4
        regr_weight = 0.4
        consistency_weight = 0.2

        # Normalize classification score (using F1)
        class_score = class_metrics.get('f1_macro', 0.0)

        # Normalize regression score (using inverted MAE)
        mae = regr_metrics.get('mae', 10.0)
        regr_score = max(0.0, 1.0 - (mae / 10.0))  # Assuming max possible MAE is 10

        # Consistency score
        consistency_score = consistency_metrics.get('class_score_consistency_pred', 0.0)

        # Calculate weighted average
        overall = (
            class_weight * class_score +
            regr_weight * regr_score +
            consistency_weight * consistency_score
        )

        return overall

    def _check_classification_targets(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Check if classification metrics meet target thresholds.

        Args:
            metrics: Classification metrics

        Returns:
            Dictionary indicating which targets are met
        """
        target_checks = {}

        # Check F1 target
        f1_target = self.target_metrics.get('clause_detection_f1', 0.82)
        target_checks['meets_f1_target'] = metrics.get('f1_macro', 0.0) >= f1_target

        # Check recall target for unfavorable terms
        recall_target = self.target_metrics.get('unfavorable_term_recall', 0.88)
        high_risk_recall = metrics.get('recall_high_risk', 0.0)
        target_checks['meets_recall_target'] = high_risk_recall >= recall_target

        return target_checks

    def _check_regression_targets(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Check if regression metrics meet target thresholds.

        Args:
            metrics: Regression metrics

        Returns:
            Dictionary indicating which targets are met
        """
        target_checks = {}

        # Check MAE target
        mae_target = self.target_metrics.get('risk_score_mae', 1.2)
        target_checks['meets_mae_target'] = metrics.get('mae', float('inf')) <= mae_target

        return target_checks

    def _scores_to_categories(self, scores: np.ndarray) -> np.ndarray:
        """Convert numerical risk scores to categorical labels.

        Args:
            scores: Risk scores (1-10 scale)

        Returns:
            Categorical labels (0=low, 1=medium, 2=high)
        """
        categories = np.zeros(len(scores), dtype=int)
        categories[scores <= 3.5] = 0  # low_risk
        categories[(scores > 3.5) & (scores <= 7.0)] = 1  # medium_risk
        categories[scores > 7.0] = 2  # high_risk
        return categories

    def _to_numpy(self, data: Union[List, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert various data formats to numpy array.

        Args:
            data: Input data

        Returns:
            Numpy array
        """
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def generate_confusion_matrix(
        self,
        y_true: Union[List[int], np.ndarray, torch.Tensor],
        y_pred: Union[List[int], np.ndarray, torch.Tensor],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """Generate and optionally save confusion matrix plot.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot (optional)

        Returns:
            Confusion matrix array
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)

        cm = confusion_matrix(y_true, y_pred)

        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Legal Clause Risk Assessment - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()
        return cm

    def generate_metrics_report(
        self,
        metrics: Dict[str, Union[float, Dict[str, float]]],
        save_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive metrics report.

        Args:
            metrics: Metrics dictionary
            save_path: Path to save the report (optional)

        Returns:
            Formatted metrics report string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("LEGAL CLAUSE RISK ASSESSMENT - EVALUATION REPORT")
        report_lines.append("=" * 60)

        # Classification metrics
        if 'classification' in metrics:
            class_metrics = metrics['classification']
            report_lines.append("\n📊 CLASSIFICATION METRICS")
            report_lines.append("-" * 30)
            report_lines.append(f"Accuracy:         {class_metrics.get('accuracy', 0):.3f}")
            report_lines.append(f"Precision (macro): {class_metrics.get('precision_macro', 0):.3f}")
            report_lines.append(f"Recall (macro):    {class_metrics.get('recall_macro', 0):.3f}")
            report_lines.append(f"F1 Score (macro):  {class_metrics.get('f1_macro', 0):.3f}")

            # Per-class metrics
            report_lines.append("\nPer-Class Performance:")
            for class_name in self.class_names:
                precision = class_metrics.get(f'precision_{class_name}', 0)
                recall = class_metrics.get(f'recall_{class_name}', 0)
                f1 = class_metrics.get(f'f1_{class_name}', 0)
                report_lines.append(f"  {class_name:12} - P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")

        # Regression metrics
        if 'regression' in metrics:
            regr_metrics = metrics['regression']
            report_lines.append("\n📈 REGRESSION METRICS")
            report_lines.append("-" * 30)
            report_lines.append(f"MAE:              {regr_metrics.get('mae', 0):.3f}")
            report_lines.append(f"RMSE:             {regr_metrics.get('rmse', 0):.3f}")
            report_lines.append(f"R² Score:         {regr_metrics.get('r2', 0):.3f}")
            report_lines.append(f"Mean Prediction:  {regr_metrics.get('pred_mean', 0):.3f}")

        # Target compliance
        report_lines.append("\n🎯 TARGET COMPLIANCE")
        report_lines.append("-" * 30)

        if isinstance(metrics, dict) and 'classification' in metrics:
            f1_target = self.target_metrics.get('clause_detection_f1', 0.82)
            f1_actual = metrics['classification'].get('f1_macro', 0)
            f1_status = "✅ PASS" if f1_actual >= f1_target else "❌ FAIL"
            report_lines.append(f"F1 Score Target:  {f1_target:.3f} | Actual: {f1_actual:.3f} | {f1_status}")

        if isinstance(metrics, dict) and 'regression' in metrics:
            mae_target = self.target_metrics.get('risk_score_mae', 1.2)
            mae_actual = metrics['regression'].get('mae', float('inf'))
            mae_status = "✅ PASS" if mae_actual <= mae_target else "❌ FAIL"
            report_lines.append(f"MAE Target:       {mae_target:.3f} | Actual: {mae_actual:.3f} | {mae_status}")

        # Overall performance
        if 'overall_score' in metrics:
            overall = metrics['overall_score']
            report_lines.append(f"\n🏆 OVERALL PERFORMANCE SCORE: {overall:.3f}")

        report_lines.append("\n" + "=" * 60)

        report = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Metrics report saved to {save_path}")

        return report


def create_metrics_evaluator(config: Config) -> LegalRiskMetrics:
    """Factory function to create a metrics evaluator instance.

    Args:
        config: Configuration object

    Returns:
        Initialized metrics evaluator
    """
    return LegalRiskMetrics(config)