"""Test suite for data loading and preprocessing modules.

This module tests the data loading utilities, preprocessing functions,
and dataset creation capabilities.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import modules under test
from legal_clause_risk_scorer.data.loader import LegalDataLoader, create_data_loader
from legal_clause_risk_scorer.data.preprocessing import (
    LegalTextPreprocessor,
    create_preprocessor,
    preprocess_dataset_batch
)


class TestLegalDataLoader:
    """Test suite for LegalDataLoader class."""

    def test_init(self, sample_config):
        """Test data loader initialization."""
        with patch('legal_clause_risk_scorer.data.loader.AutoTokenizer'):
            loader = LegalDataLoader(sample_config)
            assert loader.config == sample_config
            assert hasattr(loader, 'tokenizer')
            assert hasattr(loader, 'risk_categories')

    def test_extract_clause_type(self, sample_config):
        """Test clause type extraction from questions."""
        with patch('legal_clause_risk_scorer.data.loader.AutoTokenizer'):
            loader = LegalDataLoader(sample_config)

            # Test termination clause detection
            question = "Does the contract mention termination procedures?"
            clause_type = loader._extract_clause_type(question)
            assert clause_type == 'termination'

            # Test non-compete clause detection
            question = "Are there any restrictions on competing businesses?"
            clause_type = loader._extract_clause_type(question)
            assert clause_type == 'non_compete'

            # Test general clause fallback
            question = "What is the general structure of the contract?"
            clause_type = loader._extract_clause_type(question)
            assert clause_type == 'general'

    def test_calculate_risk_score(self, sample_config):
        """Test risk score calculation."""
        with patch('legal_clause_risk_scorer.data.loader.AutoTokenizer'):
            loader = LegalDataLoader(sample_config)

            # Test high-risk clause
            high_risk_text = "Employee may be terminated without cause at any time"
            score = loader._calculate_risk_score('termination', '', high_risk_text)
            assert score > 6.0

            # Test low-risk clause
            low_risk_text = "Employee may terminate with reasonable notice"
            score = loader._calculate_risk_score('termination', '', low_risk_text)
            assert score < 5.0

            # Test score bounds
            extreme_high_risk = "terminate without cause immediately at will unlimited"
            score = loader._calculate_risk_score('termination', '', extreme_high_risk)
            assert 1.0 <= score <= 10.0

    def test_get_risk_category(self, sample_config):
        """Test risk score to category conversion."""
        with patch('legal_clause_risk_scorer.data.loader.AutoTokenizer'):
            loader = LegalDataLoader(sample_config)

            assert loader._get_risk_category(2.5) == 'low_risk'
            assert loader._get_risk_category(5.0) == 'medium_risk'
            assert loader._get_risk_category(8.5) == 'high_risk'

            # Test boundary conditions
            assert loader._get_risk_category(3.5) == 'low_risk'
            assert loader._get_risk_category(3.6) == 'medium_risk'
            assert loader._get_risk_category(7.0) == 'medium_risk'
            assert loader._get_risk_category(7.1) == 'high_risk'

    def test_extract_clauses_from_text(self, sample_config):
        """Test clause extraction from legal text."""
        with patch('legal_clause_risk_scorer.data.loader.AutoTokenizer'):
            loader = LegalDataLoader(sample_config)

            legal_text = """
            1. This agreement shall be governed by state law.
            2. Employee agrees to confidentiality terms.
            3. Termination may occur with 30 days notice.
            """

            clauses = loader._extract_clauses_from_text(legal_text)
            assert len(clauses) > 0
            assert all(len(clause) >= 50 for clause in clauses)  # Minimum length check

    def test_infer_clause_type(self, sample_config):
        """Test clause type inference from content."""
        with patch('legal_clause_risk_scorer.data.loader.AutoTokenizer'):
            loader = LegalDataLoader(sample_config)

            # Test termination clause
            termination_text = "This agreement may be terminated by either party"
            assert loader._infer_clause_type(termination_text) == 'termination_clauses'

            # Test non-compete clause
            noncompete_text = "Employee shall not compete with similar businesses"
            assert loader._infer_clause_type(noncompete_text) == 'non_compete'

            # Test general clause
            general_text = "This document constitutes the entire agreement"
            assert loader._infer_clause_type(general_text) == 'general'

    def test_balance_dataset(self, sample_config):
        """Test dataset balancing functionality."""
        with patch('legal_clause_risk_scorer.data.loader.AutoTokenizer'):
            loader = LegalDataLoader(sample_config)

            # Create unbalanced test data
            data = []
            # Add many high-risk samples
            for i in range(20):
                data.append({'risk_category': 'high_risk', 'text': f'high risk text {i}'})
            # Add few low-risk samples
            for i in range(5):
                data.append({'risk_category': 'low_risk', 'text': f'low risk text {i}'})
            # Add some medium-risk samples
            for i in range(10):
                data.append({'risk_category': 'medium_risk', 'text': f'medium risk text {i}'})

            df = pd.DataFrame(data)
            balanced_df = loader._balance_dataset(df)

            # Check that all categories have the same count (minimum count)
            category_counts = balanced_df['risk_category'].value_counts()
            assert len(set(category_counts.values)) == 1  # All counts should be equal
            assert min(category_counts.values) == 5  # Should be equal to smallest class

    @patch('legal_clause_risk_scorer.data.loader.load_dataset')
    def test_load_cuad_dataset_mock(self, mock_load_dataset, sample_config):
        """Test CUAD dataset loading with mocked data."""
        # Mock the dataset
        mock_dataset = {
            'train': [
                {
                    'context': 'This is a sample legal contract text.',
                    'qas': [
                        {
                            'question': 'Does this mention termination?',
                            'answers': [
                                {
                                    'text': 'termination clause text',
                                    'answer_start': 0
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        mock_load_dataset.return_value = mock_dataset

        with patch('legal_clause_risk_scorer.data.loader.AutoTokenizer'):
            loader = LegalDataLoader(sample_config)
            dataset = loader.load_cuad_dataset()

            assert len(dataset) > 0
            mock_load_dataset.assert_called_once()

    def test_tokenize_dataset(self, sample_config, sample_dataset):
        """Test dataset tokenization."""
        with patch('legal_clause_risk_scorer.data.loader.AutoTokenizer') as mock_tokenizer_class:
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.return_value = {
                'input_ids': [1, 2, 3],
                'attention_mask': [1, 1, 1]
            }
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            loader = LegalDataLoader(sample_config)

            # Create a simple dataset dict
            from datasets import DatasetDict
            dataset_dict = DatasetDict({
                'train': sample_dataset,
                'validation': sample_dataset.select(range(2)),
                'test': sample_dataset.select(range(2))
            })

            tokenized = loader.tokenize_dataset(dataset_dict)

            assert 'train' in tokenized
            assert 'validation' in tokenized
            assert 'test' in tokenized

    def test_get_dataset_statistics(self, sample_config, sample_legal_texts):
        """Test dataset statistics calculation."""
        with patch('legal_clause_risk_scorer.data.loader.AutoTokenizer'):
            loader = LegalDataLoader(sample_config)

            # Create mock dataset dict
            from datasets import Dataset, DatasetDict
            df = pd.DataFrame(sample_legal_texts)
            dataset = Dataset.from_pandas(df)
            dataset_dict = DatasetDict({'train': dataset})

            stats = loader.get_dataset_statistics(dataset_dict)

            assert 'train' in stats
            train_stats = stats['train']
            assert 'total_samples' in train_stats
            assert 'avg_text_length' in train_stats
            assert 'avg_risk_score' in train_stats
            assert 'risk_category_distribution' in train_stats

            assert train_stats['total_samples'] == len(sample_legal_texts)

    def test_create_data_loader_factory(self, sample_config):
        """Test data loader factory function."""
        with patch('legal_clause_risk_scorer.data.loader.AutoTokenizer'):
            loader = create_data_loader(sample_config)
            assert isinstance(loader, LegalDataLoader)
            assert loader.config == sample_config


class TestLegalTextPreprocessor:
    """Test suite for LegalTextPreprocessor class."""

    def test_init(self, sample_config):
        """Test preprocessor initialization."""
        with patch('spacy.load') as mock_spacy, \
             patch('legal_clause_risk_scorer.data.preprocessing.AutoTokenizer'):

            mock_nlp = Mock()
            mock_spacy.return_value = mock_nlp

            preprocessor = LegalTextPreprocessor(sample_config)
            assert preprocessor.config == sample_config
            assert hasattr(preprocessor, 'legal_stopwords')
            assert hasattr(preprocessor, 'legal_abbreviations')

    def test_clean_legal_text(self, sample_config):
        """Test legal text cleaning."""
        with patch('spacy.load'), \
             patch('legal_clause_risk_scorer.data.preprocessing.AutoTokenizer'):

            preprocessor = LegalTextPreprocessor(sample_config)

            # Test basic cleaning
            dirty_text = "   This   is   a   test   contract   with   excessive   spaces.   "
            clean_text = preprocessor.clean_legal_text(dirty_text)
            assert clean_text.strip() == clean_text  # No leading/trailing whitespace
            assert '   ' not in clean_text  # No multiple spaces

            # Test empty text
            assert preprocessor.clean_legal_text("") == ""
            assert preprocessor.clean_legal_text(None) == ""

            # Test text with legal abbreviations
            abbrev_text = "The Company, Inc. vs. the Employee et al."
            clean_abbrev = preprocessor.clean_legal_text(abbrev_text)
            assert "incorporated" in clean_abbrev
            assert "versus" in clean_abbrev

    def test_remove_special_formatting(self, sample_config):
        """Test special formatting removal."""
        with patch('spacy.load'), \
             patch('legal_clause_risk_scorer.data.preprocessing.AutoTokenizer'):

            preprocessor = LegalTextPreprocessor(sample_config)

            text_with_formatting = """
            This is normal text.
            Page 1
            Some content here.
            _______________
            More content.
            """

            clean_text = preprocessor._remove_special_formatting(text_with_formatting)
            assert "Page 1" not in clean_text
            assert "_______________" not in clean_text

    def test_normalize_whitespace(self, sample_config):
        """Test whitespace normalization."""
        with patch('spacy.load'), \
             patch('legal_clause_risk_scorer.data.preprocessing.AutoTokenizer'):

            preprocessor = LegalTextPreprocessor(sample_config)

            messy_text = "This   has    multiple     spaces.\n\n\n\nAnd   multiple   newlines."
            normalized = preprocessor._normalize_whitespace(messy_text)

            assert "   " not in normalized  # No multiple spaces
            assert "\n\n\n" not in normalized  # No excessive newlines

    def test_expand_legal_abbreviations(self, sample_config):
        """Test legal abbreviation expansion."""
        with patch('spacy.load'), \
             patch('legal_clause_risk_scorer.data.preprocessing.AutoTokenizer'):

            preprocessor = LegalTextPreprocessor(sample_config)

            abbrev_text = "The company Inc. filed vs. the defendant."
            expanded = preprocessor._expand_legal_abbreviations(abbrev_text)

            assert "incorporated" in expanded
            assert "versus" in expanded

    def test_extract_features(self, sample_config, sample_legal_texts):
        """Test feature extraction from texts."""
        with patch('spacy.load'), \
             patch('legal_clause_risk_scorer.data.preprocessing.AutoTokenizer'), \
             patch('nltk.sent_tokenize') as mock_sent_tokenize:

            mock_sent_tokenize.side_effect = lambda text: text.split('.')

            preprocessor = LegalTextPreprocessor(sample_config)
            texts = [item['text'] for item in sample_legal_texts]

            features = preprocessor.extract_features(texts)

            # Check feature keys
            expected_features = [
                'text_length', 'word_count', 'sentence_count',
                'legal_terms_count', 'risk_keywords_count', 'modal_verbs_count',
                'avg_sentence_length', 'avg_word_length', 'tfidf'
            ]

            for feature in expected_features:
                assert feature in features

            # Check feature shapes
            n_texts = len(texts)
            assert len(features['text_length']) == n_texts
            assert len(features['word_count']) == n_texts
            assert features['tfidf'].shape[0] == n_texts

    def test_count_legal_terms(self, sample_config):
        """Test legal terms counting."""
        with patch('spacy.load'), \
             patch('legal_clause_risk_scorer.data.preprocessing.AutoTokenizer'):

            preprocessor = LegalTextPreprocessor(sample_config)

            # Text with legal terms
            legal_text = "This contract establishes liability and warranty obligations."
            count = preprocessor._count_legal_terms(legal_text)
            assert count >= 3  # Should find 'contract', 'liability', 'warranty'

            # Text without legal terms
            normal_text = "The quick brown fox jumps over the lazy dog."
            count = preprocessor._count_legal_terms(normal_text)
            assert count == 0

    def test_count_risk_keywords(self, sample_config):
        """Test risk keyword counting."""
        with patch('spacy.load'), \
             patch('legal_clause_risk_scorer.data.preprocessing.AutoTokenizer'):

            preprocessor = LegalTextPreprocessor(sample_config)

            # High-risk text
            risk_text = "Employee may be terminated without cause at sole discretion."
            count = preprocessor._count_risk_keywords(risk_text)
            assert count >= 2  # Should find 'without cause', 'sole discretion'

            # Low-risk text
            safe_text = "Employee will receive fair compensation."
            count = preprocessor._count_risk_keywords(safe_text)
            assert count == 0

    def test_count_modal_verbs(self, sample_config):
        """Test modal verb counting."""
        with patch('spacy.load'), \
             patch('legal_clause_risk_scorer.data.preprocessing.AutoTokenizer'):

            preprocessor = LegalTextPreprocessor(sample_config)

            # Text with modal verbs
            modal_text = "Employee shall comply and must adhere to all policies."
            count = preprocessor._count_modal_verbs(modal_text)
            assert count >= 2  # Should find 'shall', 'must'

            # Text without modal verbs
            normal_text = "The employee works in the office."
            count = preprocessor._count_modal_verbs(normal_text)
            assert count == 0

    def test_augment_data(self, sample_config):
        """Test data augmentation."""
        with patch('spacy.load'), \
             patch('legal_clause_risk_scorer.data.preprocessing.AutoTokenizer'):

            preprocessor = LegalTextPreprocessor(sample_config)

            original_texts = ["This is a test contract.", "Another legal document."]
            original_labels = [0, 1]

            augmented_texts, augmented_labels = preprocessor.augment_data(
                original_texts, original_labels, augment_ratio=0.5
            )

            # Should have more texts after augmentation
            assert len(augmented_texts) > len(original_texts)
            assert len(augmented_labels) > len(original_labels)
            assert len(augmented_texts) == len(augmented_labels)

    def test_validate_preprocessing(self, sample_config):
        """Test preprocessing validation."""
        with patch('spacy.load'), \
             patch('legal_clause_risk_scorer.data.preprocessing.AutoTokenizer'):

            preprocessor = LegalTextPreprocessor(sample_config)

            # Valid texts
            valid_texts = ["This is valid text.", "Another valid text."]
            result = preprocessor.validate_preprocessing(valid_texts)
            assert result['all_texts_valid'] is True
            assert len(result['issues']) == 0

            # Invalid texts
            invalid_texts = ["Valid text.", "", None, "x" * 15000]  # Empty, None, too long
            result = preprocessor.validate_preprocessing(invalid_texts)
            assert result['all_texts_valid'] is False
            assert len(result['issues']) > 0

    def test_create_preprocessor_factory(self, sample_config):
        """Test preprocessor factory function."""
        with patch('spacy.load'), \
             patch('legal_clause_risk_scorer.data.preprocessing.AutoTokenizer'):

            preprocessor = create_preprocessor(sample_config)
            assert isinstance(preprocessor, LegalTextPreprocessor)
            assert preprocessor.config == sample_config

    def test_preprocess_dataset_batch(self, sample_config):
        """Test batch preprocessing function."""
        with patch('spacy.load'), \
             patch('legal_clause_risk_scorer.data.preprocessing.AutoTokenizer'):

            texts = ["  This is a messy text.  ", "Another text with issues."]

            # Test clean-only mode
            cleaned_texts = preprocess_dataset_batch(texts, sample_config, clean_only=True)
            assert len(cleaned_texts) == len(texts)
            assert all(isinstance(text, str) for text in cleaned_texts)

            # Test full preprocessing mode
            processed_texts = preprocess_dataset_batch(texts, sample_config, clean_only=False)
            assert len(processed_texts) == len(texts)
            assert all(isinstance(text, str) for text in processed_texts)


class TestDataIntegration:
    """Integration tests for data loading and preprocessing."""

    @pytest.mark.integration
    def test_data_pipeline_integration(self, sample_config, temp_dir):
        """Test complete data pipeline from loading to preprocessing."""
        with patch('legal_clause_risk_scorer.data.loader.AutoTokenizer'), \
             patch('legal_clause_risk_scorer.data.loader.load_dataset') as mock_load_dataset, \
             patch('spacy.load'):

            # Mock dataset
            mock_data = {
                'train': [
                    {
                        'context': 'Sample legal contract text for testing purposes.',
                        'qas': [
                            {
                                'question': 'Does this mention termination?',
                                'answers': [
                                    {
                                        'text': 'termination procedures',
                                        'answer_start': 10
                                    }
                                ]
                            }
                        ]
                    }
                ] * 5  # Repeat for more data
            }
            mock_load_dataset.return_value = mock_data

            # Test data loading
            loader = LegalDataLoader(sample_config)
            dataset = loader.load_cuad_dataset()
            assert len(dataset) > 0

            # Test preprocessing
            preprocessor = LegalTextPreprocessor(sample_config)
            sample_texts = [item['text'] for item in dataset[:3]]  # Take first 3 items
            cleaned_texts = [preprocessor.clean_legal_text(text) for text in sample_texts]

            assert len(cleaned_texts) == len(sample_texts)
            assert all(isinstance(text, str) for text in cleaned_texts)

    def test_csv_data_loading(self, sample_config, sample_csv_file):
        """Test loading data from CSV file."""
        # Read the CSV file
        df = pd.read_csv(sample_csv_file)

        assert len(df) > 0
        assert 'text' in df.columns
        assert 'risk_score' in df.columns
        assert 'risk_category' in df.columns

        # Test that risk scores are in valid range
        assert all(1 <= score <= 10 for score in df['risk_score'])

        # Test that risk categories are valid
        valid_categories = {'low_risk', 'medium_risk', 'high_risk'}
        assert all(cat in valid_categories for cat in df['risk_category'])