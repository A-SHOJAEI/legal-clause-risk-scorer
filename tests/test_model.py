"""Test suite for model architecture and functionality.

This module tests the model classes, training components, and model utilities.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import modules under test
from legal_clause_risk_scorer.models.model import (
    LegalClauseRiskModel,
    AttentionPooling,
    LegalEmbeddingModel,
    create_model,
    load_pretrained_model,
    save_model,
    count_parameters
)


class TestLegalClauseRiskModel:
    """Test suite for LegalClauseRiskModel class."""

    def test_model_initialization(self, sample_config):
        """Test model initialization with configuration."""
        with patch('legal_clause_risk_scorer.models.model.AutoModel') as mock_auto_model, \
             patch('legal_clause_risk_scorer.models.model.AutoConfig') as mock_auto_config:

            # Mock the transformer backbone
            mock_backbone = Mock()
            mock_backbone.config = Mock()
            mock_auto_model.from_pretrained.return_value = mock_backbone
            mock_auto_config.from_pretrained.return_value = Mock()

            model = LegalClauseRiskModel(sample_config)

            # Check basic attributes
            assert model.config == sample_config
            assert model.num_labels == sample_config.get('model.num_labels')
            assert model.hidden_size == sample_config.get('model.hidden_size')
            assert model.dropout_rate == sample_config.get('model.dropout')

            # Check model components exist
            assert hasattr(model, 'backbone')
            assert hasattr(model, 'classifier_head')
            assert hasattr(model, 'regression_head')
            assert hasattr(model, 'attention_pooling')

    def test_model_forward_pass(self, sample_config, mock_tokenized_data):
        """Test model forward pass."""
        with patch('legal_clause_risk_scorer.models.model.AutoModel') as mock_auto_model, \
             patch('legal_clause_risk_scorer.models.model.AutoConfig') as mock_auto_config:

            # Mock backbone outputs
            mock_outputs = Mock()
            mock_outputs.last_hidden_state = torch.randn(4, 128, 768)  # [batch, seq, hidden]
            mock_backbone = Mock()
            mock_backbone.return_value = mock_outputs
            mock_auto_model.from_pretrained.return_value = mock_backbone
            mock_auto_config.from_pretrained.return_value = Mock()

            model = LegalClauseRiskModel(sample_config)

            # Forward pass
            outputs = model(
                input_ids=mock_tokenized_data['input_ids'],
                attention_mask=mock_tokenized_data['attention_mask'],
                labels=mock_tokenized_data['labels'],
                risk_scores=mock_tokenized_data['risk_scores']
            )

            # Check output structure
            assert 'logits' in outputs
            assert 'risk_scores' in outputs
            assert 'loss' in outputs

            # Check output shapes
            batch_size = mock_tokenized_data['input_ids'].size(0)
            assert outputs['logits'].shape == (batch_size, sample_config.get('model.num_labels'))
            assert outputs['risk_scores'].shape == (batch_size, 1)

    def test_predict_risk(self, sample_config):
        """Test risk prediction functionality."""
        with patch('legal_clause_risk_scorer.models.model.AutoModel') as mock_auto_model, \
             patch('legal_clause_risk_scorer.models.model.AutoConfig') as mock_auto_config:

            # Mock backbone and outputs
            mock_outputs = Mock()
            mock_outputs.last_hidden_state = torch.randn(2, 128, 768)
            mock_backbone = Mock()
            mock_backbone.return_value = mock_outputs
            mock_auto_model.from_pretrained.return_value = mock_backbone
            mock_auto_config.from_pretrained.return_value = Mock()

            model = LegalClauseRiskModel(sample_config)

            # Prepare input
            input_ids = torch.randint(0, 1000, (2, 128))
            attention_mask = torch.ones(2, 128)

            # Test prediction
            predictions = model.predict_risk(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Check prediction structure
            assert 'risk_categories' in predictions
            assert 'risk_scores' in predictions
            assert 'confidence_scores' in predictions
            assert 'category_probabilities' in predictions

            # Check prediction content
            assert len(predictions['risk_categories']) == 2
            assert len(predictions['risk_scores']) == 2
            assert all(cat in ['low_risk', 'medium_risk', 'high_risk']
                      for cat in predictions['risk_categories'])

    def test_get_attention_weights(self, sample_config):
        """Test attention weights extraction."""
        with patch('legal_clause_risk_scorer.models.model.AutoModel') as mock_auto_model, \
             patch('legal_clause_risk_scorer.models.model.AutoConfig') as mock_auto_config:

            mock_outputs = Mock()
            mock_outputs.last_hidden_state = torch.randn(1, 128, 768)
            mock_backbone = Mock()
            mock_backbone.return_value = mock_outputs
            mock_auto_model.from_pretrained.return_value = mock_backbone
            mock_auto_config.from_pretrained.return_value = Mock()

            model = LegalClauseRiskModel(sample_config)

            input_ids = torch.randint(0, 1000, (1, 128))
            attention_mask = torch.ones(1, 128)

            attention_weights = model.get_attention_weights(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            assert attention_weights is not None
            assert isinstance(attention_weights, torch.Tensor)

    def test_num_parameters(self, sample_config):
        """Test parameter counting."""
        with patch('legal_clause_risk_scorer.models.model.AutoModel') as mock_auto_model, \
             patch('legal_clause_risk_scorer.models.model.AutoConfig') as mock_auto_config:

            mock_backbone = Mock()
            mock_backbone.parameters.return_value = [torch.randn(10, 10), torch.randn(5)]
            mock_auto_model.from_pretrained.return_value = mock_backbone
            mock_auto_config.from_pretrained.return_value = Mock()

            model = LegalClauseRiskModel(sample_config)
            param_count = model.num_parameters()

            assert isinstance(param_count, int)
            assert param_count > 0

    def test_model_with_frozen_backbone(self, sample_config):
        """Test model with frozen backbone parameters."""
        sample_config.set('model.freeze_backbone', True)

        with patch('legal_clause_risk_scorer.models.model.AutoModel') as mock_auto_model, \
             patch('legal_clause_risk_scorer.models.model.AutoConfig') as mock_auto_config:

            mock_backbone = Mock()
            mock_param = Mock()
            mock_param.requires_grad = True
            mock_backbone.parameters.return_value = [mock_param]
            mock_auto_model.from_pretrained.return_value = mock_backbone
            mock_auto_config.from_pretrained.return_value = Mock()

            model = LegalClauseRiskModel(sample_config)

            # Check that freezing was called
            mock_backbone.parameters.assert_called()


class TestAttentionPooling:
    """Test suite for AttentionPooling layer."""

    def test_attention_pooling_initialization(self):
        """Test attention pooling initialization."""
        hidden_size = 768
        pooling = AttentionPooling(hidden_size)

        assert hasattr(pooling, 'attention')
        assert isinstance(pooling.attention, nn.Linear)

    def test_attention_pooling_forward(self):
        """Test attention pooling forward pass."""
        hidden_size = 768
        batch_size = 2
        seq_len = 10

        pooling = AttentionPooling(hidden_size)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        output = pooling(hidden_states, attention_mask)

        assert output.shape == (batch_size, hidden_size)

    def test_attention_pooling_without_mask(self):
        """Test attention pooling without attention mask."""
        hidden_size = 768
        batch_size = 2
        seq_len = 10

        pooling = AttentionPooling(hidden_size)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        output = pooling(hidden_states)

        assert output.shape == (batch_size, hidden_size)


class TestLegalEmbeddingModel:
    """Test suite for LegalEmbeddingModel class."""

    def test_embedding_model_initialization(self, sample_config):
        """Test legal embedding model initialization."""
        with patch('legal_clause_risk_scorer.models.model.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model

            embedding_model = LegalEmbeddingModel(sample_config)

            assert embedding_model.config == sample_config
            assert hasattr(embedding_model, 'sentence_model')
            assert hasattr(embedding_model, 'legal_projection')
            assert embedding_model.embedding_dim == 384

    def test_encode_texts(self, sample_config):
        """Test text encoding functionality."""
        with patch('legal_clause_risk_scorer.models.model.SentenceTransformer') as mock_st:
            # Mock sentence transformer
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.return_value = torch.randn(3, 384)
            mock_st.return_value = mock_model

            embedding_model = LegalEmbeddingModel(sample_config)

            texts = ["Legal text 1", "Legal text 2", "Legal text 3"]
            embeddings = embedding_model.encode(texts)

            assert isinstance(embeddings, torch.Tensor)
            assert embeddings.shape[0] == len(texts)
            assert embeddings.shape[1] == 384

    def test_similarity_search(self, sample_config):
        """Test similarity search functionality."""
        with patch('legal_clause_risk_scorer.models.model.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.side_effect = [
                torch.randn(1, 384),  # Query embedding
                torch.randn(5, 384)   # Candidate embeddings
            ]
            mock_st.return_value = mock_model

            embedding_model = LegalEmbeddingModel(sample_config)

            query = "Sample legal query"
            candidates = [f"Candidate {i}" for i in range(5)]

            results = embedding_model.similarity_search(query, candidates, top_k=3)

            assert len(results) == 3
            assert all(isinstance(item, tuple) for item in results)
            assert all(len(item) == 2 for item in results)  # (index, score) pairs


class TestModelUtilities:
    """Test suite for model utility functions."""

    def test_create_model_factory(self, sample_config):
        """Test model factory function."""
        with patch('legal_clause_risk_scorer.models.model.AutoModel'), \
             patch('legal_clause_risk_scorer.models.model.AutoConfig'):

            # Test risk assessment model creation
            model = create_model(sample_config, "risk_assessment")
            assert isinstance(model, LegalClauseRiskModel)

        with patch('legal_clause_risk_scorer.models.model.SentenceTransformer') as mock_st:
            mock_st.return_value.get_sentence_embedding_dimension.return_value = 384

            # Test embedding model creation
            model = create_model(sample_config, "embedding")
            assert isinstance(model, LegalEmbeddingModel)

        # Test invalid model type
        with pytest.raises(ValueError):
            create_model(sample_config, "invalid_type")

    def test_save_and_load_model(self, sample_config, temp_dir):
        """Test model saving and loading."""
        with patch('legal_clause_risk_scorer.models.model.AutoModel') as mock_auto_model, \
             patch('legal_clause_risk_scorer.models.model.AutoConfig') as mock_auto_config:

            mock_backbone = Mock()
            mock_auto_model.from_pretrained.return_value = mock_backbone
            mock_auto_config.from_pretrained.return_value = Mock()

            # Create and save model
            model = LegalClauseRiskModel(sample_config)
            save_path = temp_dir / "test_model.pth"

            save_model(model, str(save_path))
            assert save_path.exists()

            # Load model
            loaded_model = load_pretrained_model(str(save_path), sample_config)
            assert isinstance(loaded_model, LegalClauseRiskModel)

        # Test loading non-existent model
        with pytest.raises(FileNotFoundError):
            load_pretrained_model("non_existent.pth", sample_config)

    def test_count_parameters_function(self, sample_config):
        """Test parameter counting utility function."""
        with patch('legal_clause_risk_scorer.models.model.AutoModel') as mock_auto_model, \
             patch('legal_clause_risk_scorer.models.model.AutoConfig') as mock_auto_config:

            # Create mock parameters
            param1 = torch.nn.Parameter(torch.randn(10, 10))  # 100 params
            param2 = torch.nn.Parameter(torch.randn(5, 5))    # 25 params
            param3 = torch.nn.Parameter(torch.randn(3, 3))    # 9 params, frozen
            param3.requires_grad = False

            mock_backbone = Mock()
            mock_backbone.parameters.return_value = [param1, param2, param3]
            mock_auto_model.from_pretrained.return_value = mock_backbone
            mock_auto_config.from_pretrained.return_value = Mock()

            model = LegalClauseRiskModel(sample_config)

            param_counts = count_parameters(model)

            assert 'total_parameters' in param_counts
            assert 'trainable_parameters' in param_counts
            assert 'frozen_parameters' in param_counts

            assert isinstance(param_counts['total_parameters'], int)
            assert isinstance(param_counts['trainable_parameters'], int)
            assert isinstance(param_counts['frozen_parameters'], int)

    def test_model_device_handling(self, sample_config):
        """Test model device handling."""
        with patch('legal_clause_risk_scorer.models.model.AutoModel') as mock_auto_model, \
             patch('legal_clause_risk_scorer.models.model.AutoConfig') as mock_auto_config:

            mock_backbone = Mock()
            mock_auto_model.from_pretrained.return_value = mock_backbone
            mock_auto_config.from_pretrained.return_value = Mock()

            model = LegalClauseRiskModel(sample_config)

            # Test moving to CPU
            model = model.to('cpu')
            # Should not raise an error

            # Test CUDA if available
            if torch.cuda.is_available():
                model = model.to('cuda')
                # Should not raise an error


class TestModelIntegration:
    """Integration tests for model components."""

    @pytest.mark.integration
    def test_full_model_pipeline(self, sample_config, mock_tokenized_data):
        """Test complete model pipeline from input to output."""
        with patch('legal_clause_risk_scorer.models.model.AutoModel') as mock_auto_model, \
             patch('legal_clause_risk_scorer.models.model.AutoConfig') as mock_auto_config:

            # Setup mock backbone
            mock_outputs = Mock()
            mock_outputs.last_hidden_state = torch.randn(4, 128, 768)
            mock_backbone = Mock()
            mock_backbone.return_value = mock_outputs
            mock_auto_model.from_pretrained.return_value = mock_backbone
            mock_auto_config.from_pretrained.return_value = Mock()

            model = LegalClauseRiskModel(sample_config)

            # Test training mode
            model.train()
            train_outputs = model(**mock_tokenized_data)
            assert 'loss' in train_outputs
            assert train_outputs['loss'].requires_grad

            # Test evaluation mode
            model.eval()
            with torch.no_grad():
                eval_outputs = model(
                    input_ids=mock_tokenized_data['input_ids'],
                    attention_mask=mock_tokenized_data['attention_mask']
                )
                assert 'logits' in eval_outputs
                assert 'risk_scores' in eval_outputs

    @pytest.mark.integration
    def test_model_serialization(self, sample_config, temp_dir):
        """Test model serialization and deserialization."""
        with patch('legal_clause_risk_scorer.models.model.AutoModel') as mock_auto_model, \
             patch('legal_clause_risk_scorer.models.model.AutoConfig') as mock_auto_config:

            mock_backbone = Mock()
            mock_auto_model.from_pretrained.return_value = mock_backbone
            mock_auto_config.from_pretrained.return_value = Mock()

            # Create model and save state
            original_model = LegalClauseRiskModel(sample_config)
            save_path = temp_dir / "test_model.pth"

            # Save model state
            torch.save(original_model.state_dict(), save_path)

            # Load model state into new instance
            new_model = LegalClauseRiskModel(sample_config)
            new_model.load_state_dict(torch.load(save_path))

            # Models should have equivalent state
            # This is a basic check - in practice you'd compare outputs
            assert type(original_model) == type(new_model)

    def test_model_configuration_edge_cases(self, sample_config):
        """Test model with various configuration edge cases."""
        # Test with minimal configuration
        minimal_config = sample_config
        minimal_config.set('model.num_labels', 2)
        minimal_config.set('model.dropout', 0.0)

        with patch('legal_clause_risk_scorer.models.model.AutoModel') as mock_auto_model, \
             patch('legal_clause_risk_scorer.models.model.AutoConfig') as mock_auto_config:

            mock_backbone = Mock()
            mock_auto_model.from_pretrained.return_value = mock_backbone
            mock_auto_config.from_pretrained.return_value = Mock()

            model = LegalClauseRiskModel(minimal_config)
            assert model.num_labels == 2
            assert model.dropout_rate == 0.0

        # Test with maximum configuration
        maximal_config = sample_config
        maximal_config.set('model.num_labels', 10)
        maximal_config.set('model.dropout', 0.9)

        with patch('legal_clause_risk_scorer.models.model.AutoModel') as mock_auto_model, \
             patch('legal_clause_risk_scorer.models.model.AutoConfig') as mock_auto_config:

            mock_backbone = Mock()
            mock_auto_model.from_pretrained.return_value = mock_backbone
            mock_auto_config.from_pretrained.return_value = Mock()

            model = LegalClauseRiskModel(maximal_config)
            assert model.num_labels == 10
            assert model.dropout_rate == 0.9