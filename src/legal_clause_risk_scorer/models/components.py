"""Custom model components for legal clause risk assessment.

This module implements specialized loss functions and custom neural network
components that are novel contributions to the legal text analysis domain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in legal risk classification.

    Focal loss down-weights easy examples and focuses training on hard negatives,
    which is particularly useful for legal text where high-risk clauses are rare
    but critical to detect accurately.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    Adapted for legal clause risk assessment.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """Initialize focal loss.

        Args:
            alpha: Class weights tensor of shape (num_classes,)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Loss reduction method ('none', 'mean', 'sum')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Predicted logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)

        Returns:
            Computed focal loss
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get probabilities
        probs = torch.softmax(inputs, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal term
        focal_term = (1 - pt) ** self.gamma

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiTaskLoss(nn.Module):
    """Multi-task loss combining classification and regression objectives.

    This custom loss function balances the classification task (risk category)
    and regression task (risk score) with learnable task-specific weights
    using uncertainty-based weighting.

    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to
    Weigh Losses for Scene Geometry and Semantics" (2018)
    """

    def __init__(
        self,
        use_uncertainty_weighting: bool = True,
        initial_log_var_class: float = 0.0,
        initial_log_var_reg: float = 0.0
    ):
        """Initialize multi-task loss.

        Args:
            use_uncertainty_weighting: Whether to use learnable uncertainty weights
            initial_log_var_class: Initial log variance for classification task
            initial_log_var_reg: Initial log variance for regression task
        """
        super().__init__()
        self.use_uncertainty_weighting = use_uncertainty_weighting

        if use_uncertainty_weighting:
            # Learnable log variance parameters for uncertainty weighting
            self.log_var_class = nn.Parameter(
                torch.tensor(initial_log_var_class, dtype=torch.float32)
            )
            self.log_var_reg = nn.Parameter(
                torch.tensor(initial_log_var_reg, dtype=torch.float32)
            )

    def forward(
        self,
        classification_loss: torch.Tensor,
        regression_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined multi-task loss.

        Args:
            classification_loss: Loss from classification task
            regression_loss: Loss from regression task

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        if self.use_uncertainty_weighting:
            # Uncertainty-based weighting
            # Loss = (1 / 2*sigma^2) * task_loss + log(sigma)
            # where sigma^2 = exp(log_var)

            weighted_class_loss = (
                0.5 * torch.exp(-self.log_var_class) * classification_loss
                + 0.5 * self.log_var_class
            )
            weighted_reg_loss = (
                0.5 * torch.exp(-self.log_var_reg) * regression_loss
                + 0.5 * self.log_var_reg
            )

            total_loss = weighted_class_loss + weighted_reg_loss

            loss_components = {
                'classification_loss': classification_loss.item(),
                'regression_loss': regression_loss.item(),
                'weighted_classification_loss': weighted_class_loss.item(),
                'weighted_regression_loss': weighted_reg_loss.item(),
                'class_uncertainty': torch.exp(self.log_var_class).item(),
                'reg_uncertainty': torch.exp(self.log_var_reg).item()
            }
        else:
            # Simple sum
            total_loss = classification_loss + regression_loss

            loss_components = {
                'classification_loss': classification_loss.item(),
                'regression_loss': regression_loss.item()
            }

        return total_loss, loss_components


class ConsistencyRegularization(nn.Module):
    """Consistency regularization between classification and regression outputs.

    This component enforces consistency between the predicted risk category
    and the predicted risk score, acting as a regularization term to ensure
    coherent predictions.
    """

    def __init__(self, weight: float = 0.1):
        """Initialize consistency regularization.

        Args:
            weight: Weight for the consistency loss term
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        risk_scores: torch.Tensor,
        predicted_categories: torch.Tensor
    ) -> torch.Tensor:
        """Compute consistency loss.

        Args:
            risk_scores: Predicted risk scores in range [1, 10]
            predicted_categories: Predicted category logits

        Returns:
            Consistency loss value
        """
        # Get predicted categories (0: low, 1: medium, 2: high)
        pred_cats = torch.argmax(predicted_categories, dim=-1)

        # Define expected score ranges for each category
        # Low: [1, 4), Medium: [4, 7), High: [7, 10]
        category_ranges = torch.tensor([
            [1.0, 4.0],  # low_risk
            [4.0, 7.0],  # medium_risk
            [7.0, 10.0]  # high_risk
        ], device=risk_scores.device)

        # Get expected range for each prediction
        batch_ranges = category_ranges[pred_cats]  # (batch_size, 2)

        # Calculate violation: how far the score is from expected range
        lower_bound = batch_ranges[:, 0]
        upper_bound = batch_ranges[:, 1]

        # Penalty for scores outside expected range
        below_range = F.relu(lower_bound - risk_scores.squeeze())
        above_range = F.relu(risk_scores.squeeze() - upper_bound)

        consistency_loss = self.weight * (below_range + above_range).mean()

        return consistency_loss


class LegalDomainLoss(nn.Module):
    """Domain-specific loss function for legal clause risk assessment.

    Combines focal loss for classification, MSE for regression, and
    consistency regularization between tasks.
    """

    def __init__(
        self,
        num_classes: int = 3,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        use_uncertainty_weighting: bool = True,
        consistency_weight: float = 0.1
    ):
        """Initialize legal domain loss.

        Args:
            num_classes: Number of risk categories
            class_weights: Weights for each class
            focal_gamma: Gamma parameter for focal loss
            use_uncertainty_weighting: Use uncertainty-based multi-task weighting
            consistency_weight: Weight for consistency regularization
        """
        super().__init__()

        self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.regression_loss = nn.MSELoss()
        self.multi_task_loss = MultiTaskLoss(
            use_uncertainty_weighting=use_uncertainty_weighting
        )
        self.consistency_reg = ConsistencyRegularization(weight=consistency_weight)

    def forward(
        self,
        classification_logits: torch.Tensor,
        regression_outputs: torch.Tensor,
        classification_targets: torch.Tensor,
        regression_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute total loss.

        Args:
            classification_logits: Predicted category logits
            regression_outputs: Predicted risk scores
            classification_targets: True category labels
            regression_targets: True risk scores

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Compute individual losses
        class_loss = self.focal_loss(classification_logits, classification_targets)
        reg_loss = self.regression_loss(regression_outputs.squeeze(), regression_targets)

        # Combine with multi-task weighting
        combined_loss, loss_components = self.multi_task_loss(class_loss, reg_loss)

        # Add consistency regularization
        consistency_loss = self.consistency_reg(regression_outputs, classification_logits)
        total_loss = combined_loss + consistency_loss

        # Update loss components
        loss_components['consistency_loss'] = consistency_loss.item()
        loss_components['total_loss'] = total_loss.item()

        return total_loss, loss_components


class ContextAwareAttention(nn.Module):
    """Context-aware attention mechanism for legal clause analysis.

    This custom attention layer learns to focus on legally-significant tokens
    (e.g., modal verbs, negations, time references) that are critical for
    risk assessment.
    """

    def __init__(
        self,
        hidden_size: int,
        num_context_features: int = 10
    ):
        """Initialize context-aware attention.

        Args:
            hidden_size: Hidden dimension size
            num_context_features: Number of context feature types to learn
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_context_features = num_context_features

        # Learn context-specific attention patterns
        self.context_query = nn.Parameter(
            torch.randn(num_context_features, hidden_size)
        )
        self.attention_projection = nn.Linear(hidden_size, num_context_features)

        # Combine context-aware features
        self.context_aggregation = nn.Linear(
            num_context_features * hidden_size,
            hidden_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply context-aware attention.

        Args:
            hidden_states: Input hidden states (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            Tuple of (context_vector, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute attention scores for each context type
        # (batch_size, seq_len, num_context_features)
        attention_scores = self.attention_projection(hidden_states)

        # Apply mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                ~attention_mask.unsqueeze(-1).bool(),
                float('-inf')
            )

        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)

        # Apply attention for each context type
        context_vectors = []
        for i in range(self.num_context_features):
            # (batch_size, 1, seq_len) x (batch_size, seq_len, hidden_size)
            context_vec = torch.bmm(
                attention_weights[:, :, i].unsqueeze(1),
                hidden_states
            ).squeeze(1)
            context_vectors.append(context_vec)

        # Concatenate all context vectors
        all_contexts = torch.cat(context_vectors, dim=-1)

        # Aggregate contexts
        output = self.context_aggregation(all_contexts)

        return output, attention_weights


# Factory function for creating loss functions
def create_loss_function(
    loss_type: str = 'legal_domain',
    **kwargs
) -> nn.Module:
    """Create a loss function instance.

    Args:
        loss_type: Type of loss ('focal', 'multi_task', 'legal_domain')
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function instance
    """
    if loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'multi_task':
        return MultiTaskLoss(**kwargs)
    elif loss_type == 'legal_domain':
        return LegalDomainLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
