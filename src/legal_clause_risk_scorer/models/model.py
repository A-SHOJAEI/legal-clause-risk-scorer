"""Core model architecture for legal clause risk assessment.

This module implements the main neural network models for classifying legal clauses
and predicting risk scores, including multi-task learning capabilities.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer
)
from sentence_transformers import SentenceTransformer

from ..utils.config import Config


logger = logging.getLogger(__name__)


class LegalClauseRiskModel(nn.Module):
    """Multi-task model for legal clause risk assessment.

    This model performs both classification (risk category) and regression
    (risk score) tasks using a shared transformer backbone with task-specific heads.

    Attributes:
        app_config: Application configuration
        backbone: Transformer backbone model
        classifier_head: Classification head for risk categories
        regression_head: Regression head for risk scores
        feature_projection: Optional projection layer for additional features
        dropout: Dropout layer
    """

    def __init__(self, config: Config):
        """Initialize the risk assessment model.

        Args:
            config: Application configuration
        """
        super().__init__()

        self.app_config = config
        self.num_labels = config.get('model.num_labels', 3)
        self.hidden_size = config.get('model.hidden_size', 768)
        self.dropout_rate = config.get('model.dropout', 0.1)

        # Initialize backbone
        self._init_backbone()

        # Initialize task-specific heads
        self._init_classification_head()
        self._init_regression_head()

        # Initialize additional components
        self._init_additional_components()

        logger.info(f"Initialized LegalClauseRiskModel with {self.num_parameters()} parameters")

    def _init_backbone(self) -> None:
        """Initialize the transformer backbone."""
        model_name = self.app_config.get('model.base_model', 'microsoft/deberta-v3-base')

        try:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
        except Exception as e:
            logger.error(f"Failed to load backbone model {model_name}: {e}")
            raise

        # Freeze backbone if specified
        if self.app_config.get('model.freeze_backbone', False):
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone parameters frozen")

    def _init_classification_head(self) -> None:
        """Initialize the classification head for risk categories."""
        self.classifier_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 4, self.num_labels)
        )

    def _init_regression_head(self) -> None:
        """Initialize the regression head for risk scores."""
        self.regression_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 4, 1),
            nn.Sigmoid()  # Output between 0-1, will be scaled to 1-10
        )

    def _init_additional_components(self) -> None:
        """Initialize additional model components."""
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)

        # Attention pooling for better sequence representation
        self.attention_pooling = AttentionPooling(self.hidden_size)

        # Feature projection for additional features (if used)
        self.feature_projection = None
        if self.app_config.get('model.use_additional_features', False):
            feature_dim = self.app_config.get('model.additional_feature_dim', 50)
            self.feature_projection = nn.Linear(feature_dim, self.hidden_size // 4)

        # Multi-head attention for clause-specific modeling
        self.clause_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=self.dropout_rate,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        risk_scores: Optional[torch.Tensor] = None,
        additional_features: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        """Forward pass of the model.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            labels: Classification labels (risk categories)
            risk_scores: Regression targets (risk scores)
            additional_features: Additional features
            return_dict: Whether to return a dictionary

        Returns:
            Model outputs including logits, predictions, and losses
        """
        # Get backbone outputs
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get sequence representation
        sequence_output = backbone_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Apply clause-specific attention
        attended_output, attention_weights = self.clause_attention(
            sequence_output,
            sequence_output,
            sequence_output,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )

        # Combine original and attended representations
        combined_output = sequence_output + attended_output
        combined_output = self.layer_norm(combined_output)

        # Pool sequence representation
        pooled_output = self.attention_pooling(combined_output, attention_mask)

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Incorporate additional features if provided
        if additional_features is not None and self.feature_projection is not None:
            projected_features = self.feature_projection(additional_features)
            pooled_output = torch.cat([pooled_output, projected_features], dim=-1)

        # Get task-specific predictions
        classification_logits = self.classifier_head(pooled_output)
        regression_output = self.regression_head(pooled_output)

        # Scale regression output to 1-10 range
        risk_score_predictions = regression_output * 9.0 + 1.0  # Scale [0,1] -> [1,10]

        # Calculate losses if labels are provided
        total_loss = None
        classification_loss = None
        regression_loss = None

        if labels is not None or risk_scores is not None:
            loss_components = []

            # Classification loss
            if labels is not None:
                classification_loss = F.cross_entropy(
                    classification_logits,
                    labels.long()
                )
                loss_components.append(classification_loss)

            # Regression loss
            if risk_scores is not None:
                regression_loss = F.mse_loss(
                    risk_score_predictions.squeeze(),
                    risk_scores.float()
                )
                loss_components.append(regression_loss)

            # Combine losses
            if loss_components:
                # Weight the losses
                classification_weight = self.app_config.get('training.classification_weight', 1.0)
                regression_weight = self.app_config.get('training.regression_weight', 1.0)

                total_loss = 0.0
                if classification_loss is not None:
                    total_loss += classification_weight * classification_loss
                if regression_loss is not None:
                    total_loss += regression_weight * regression_loss

        if return_dict:
            return {
                'loss': total_loss,
                'classification_loss': classification_loss,
                'regression_loss': regression_loss,
                'logits': classification_logits,
                'risk_scores': risk_score_predictions,
                'hidden_states': pooled_output,
                'attention_weights': attention_weights
            }
        else:
            outputs = (classification_logits, risk_score_predictions)
            if total_loss is not None:
                outputs = (total_loss,) + outputs
            return outputs

    def predict_risk(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """Predict risk category and score for input texts.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments

        Returns:
            Dictionary containing predictions and confidence scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )

            # Get predictions
            classification_probs = F.softmax(outputs['logits'], dim=-1)
            predicted_categories = torch.argmax(classification_probs, dim=-1)
            risk_scores = outputs['risk_scores']

            # Convert to interpretable format
            category_names = ['low_risk', 'medium_risk', 'high_risk']
            predicted_category_names = [category_names[idx] for idx in predicted_categories.cpu().tolist()]

            # Calculate confidence scores
            confidence_scores = torch.max(classification_probs, dim=-1)[0]

            return {
                'risk_categories': predicted_category_names,
                'risk_scores': risk_scores.squeeze().cpu().tolist(),
                'category_probabilities': classification_probs.cpu(),
                'confidence_scores': confidence_scores.cpu().tolist(),
                'raw_logits': outputs['logits'].cpu()
            }

    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get attention weights for interpretability.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Attention weights tensor
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            return outputs['attention_weights']

    def num_parameters(self) -> int:
        """Get total number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionPooling(nn.Module):
    """Attention-based pooling layer for sequence representations."""

    def __init__(self, hidden_size: int):
        """Initialize attention pooling.

        Args:
            hidden_size: Hidden dimension size
        """
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of attention pooling.

        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Pooled representation [batch_size, hidden_size]
        """
        # Calculate attention weights
        attention_weights = self.attention(hidden_states).squeeze(-1)  # [batch_size, seq_len]

        # Apply mask if provided
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                ~attention_mask.bool(), float('-inf')
            )

        # Apply softmax
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Apply attention weights
        weighted_output = torch.bmm(
            attention_weights.unsqueeze(1),  # [batch_size, 1, seq_len]
            hidden_states  # [batch_size, seq_len, hidden_size]
        ).squeeze(1)  # [batch_size, hidden_size]

        return weighted_output


class LegalEmbeddingModel(nn.Module):
    """Specialized embedding model for legal text using sentence transformers."""

    def __init__(self, config: Config):
        """Initialize the embedding model.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config

        # Initialize sentence transformer
        embedding_model_name = config.get(
            'model.embedding_model',
            'sentence-transformers/all-MiniLM-L6-v2'
        )

        try:
            self.sentence_model = SentenceTransformer(embedding_model_name)
            self.embedding_dim = self.sentence_model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to load embedding model {embedding_model_name}: {e}")
            raise

        # Add legal domain-specific layers
        self.legal_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )

        logger.info(f"Initialized legal embedding model with dimension {self.embedding_dim}")

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings.

        Args:
            texts: List of input texts

        Returns:
            Tensor of embeddings
        """
        # Get sentence embeddings
        embeddings = self.sentence_model.encode(texts, convert_to_tensor=True)

        # Apply legal-specific projection
        legal_embeddings = self.legal_projection(embeddings)

        return legal_embeddings

    def similarity_search(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Find most similar texts to query.

        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score) tuples
        """
        # Encode query and candidates
        query_embedding = self.encode([query_text])
        candidate_embeddings = self.encode(candidate_texts)

        # Calculate similarities
        similarities = F.cosine_similarity(
            query_embedding,
            candidate_embeddings,
            dim=-1
        )

        # Get top-k results
        top_scores, top_indices = torch.topk(similarities, min(top_k, len(candidate_texts)))

        return list(zip(top_indices.cpu().tolist(), top_scores.cpu().tolist()))


def create_model(config: Config, model_type: str = "risk_assessment") -> nn.Module:
    """Factory function to create model instances.

    Args:
        config: Configuration object
        model_type: Type of model to create ('risk_assessment' or 'embedding')

    Returns:
        Initialized model instance

    Raises:
        ValueError: If unknown model type is specified
    """
    if model_type == "risk_assessment":
        return LegalClauseRiskModel(config)
    elif model_type == "embedding":
        return LegalEmbeddingModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_pretrained_model(
    model_path: str,
    config: Config,
    model_type: str = "risk_assessment"
) -> nn.Module:
    """Load a pretrained model from disk.

    Args:
        model_path: Path to the saved model
        config: Configuration object
        model_type: Type of model to load

    Returns:
        Loaded model instance

    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    try:
        # Create model instance
        model = create_model(config, model_type)

        # Load state dict
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')

        model.load_state_dict(state_dict)

        logger.info(f"Loaded pretrained model from {model_path}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


def save_model(model: nn.Module, save_path: str) -> None:
    """Save a model to disk.

    Args:
        model: Model to save
        save_path: Path to save the model

    Raises:
        RuntimeError: If model saving fails
    """
    try:
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

    except Exception as e:
        logger.error(f"Failed to save model to {save_path}: {e}")
        raise RuntimeError(f"Model saving failed: {e}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters in a model.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params
    }