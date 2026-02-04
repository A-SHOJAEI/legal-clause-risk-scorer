"""Data loading utilities for legal clause risk assessment.

This module provides functions to load and prepare datasets for training and evaluation,
including the CUAD and LEDGAR datasets with proper preprocessing and labeling.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from ..utils.config import Config


logger = logging.getLogger(__name__)


class LegalDataLoader:
    """Data loader for legal contract datasets with risk scoring capabilities.

    This class handles loading CUAD and LEDGAR datasets, preprocessing text,
    and creating risk labels based on clause types and content analysis.

    Attributes:
        config: Configuration object
        tokenizer: Tokenizer for text preprocessing
        risk_categories: Dictionary of risk categories and their configurations
    """

    def __init__(self, config: Config, tokenizer_name: Optional[str] = None):
        """Initialize the data loader.

        Args:
            config: Configuration object
            tokenizer_name: Name of the tokenizer to use. If None, uses config default.
        """
        self.config = config
        self.risk_categories = config.get('data.risk_categories', {})

        # Initialize tokenizer
        if tokenizer_name is None:
            tokenizer_name = config.get('model.base_model', 'microsoft/deberta-v3-base')

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Initialized data loader with tokenizer: {tokenizer_name}")

    def load_cuad_dataset(self) -> Dataset:
        """Load and preprocess the CUAD dataset.

        Returns:
            Preprocessed CUAD dataset with risk labels

        Raises:
            RuntimeError: If dataset loading fails
        """
        try:
            logger.info("Loading CUAD dataset...")
            dataset_name = self.config.get('data.cuad_dataset', 'cuad')
            dataset = load_dataset(dataset_name)

            # Process the dataset to extract clause information
            processed_data = []

            for split_name, split_data in dataset.items():
                logger.info(f"Processing CUAD {split_name} split: {len(split_data)} samples")

                for example in split_data:
                    # Extract contract text and answers
                    context = example.get('context', '')
                    questions = example.get('qas', [])

                    for qa in questions:
                        question = qa.get('question', '')
                        answers = qa.get('answers', [])

                        # Extract clause type from question
                        clause_type = self._extract_clause_type(question)

                        # Process each answer
                        for answer in answers:
                            answer_text = answer.get('text', '')
                            start_idx = answer.get('answer_start', 0)

                            if answer_text.strip():
                                # Extract surrounding context
                                clause_context = self._extract_clause_context(
                                    context, answer_text, start_idx
                                )

                                # Calculate risk score
                                risk_score = self._calculate_risk_score(
                                    clause_type, clause_context, answer_text
                                )

                                processed_data.append({
                                    'text': clause_context,
                                    'clause_text': answer_text,
                                    'clause_type': clause_type,
                                    'risk_score': risk_score,
                                    'risk_category': self._get_risk_category(risk_score),
                                    'source': 'cuad'
                                })

            # Convert to dataset
            df = pd.DataFrame(processed_data)
            dataset = Dataset.from_pandas(df)

            logger.info(f"CUAD dataset loaded: {len(dataset)} samples")
            return dataset

        except Exception as e:
            logger.error(f"Error loading CUAD dataset: {e}")
            raise RuntimeError(f"Failed to load CUAD dataset: {e}")

    def load_ledgar_dataset(self) -> Dataset:
        """Load and preprocess the LEDGAR dataset.

        Returns:
            Preprocessed LEDGAR dataset with risk labels

        Raises:
            RuntimeError: If dataset loading fails
        """
        try:
            logger.info("Loading LEDGAR dataset...")
            dataset_name = self.config.get('data.ledgar_dataset', 'lexlms/ledgar')
            dataset = load_dataset(dataset_name)

            processed_data = []

            for split_name, split_data in dataset.items():
                logger.info(f"Processing LEDGAR {split_name} split: {len(split_data)} samples")

                for example in split_data:
                    text = example.get('text', '')
                    labels = example.get('labels', [])

                    if text.strip():
                        # Extract clauses from legal text
                        clauses = self._extract_clauses_from_text(text)

                        for clause_text in clauses:
                            # Determine clause type from content
                            clause_type = self._infer_clause_type(clause_text)

                            # Calculate risk score
                            risk_score = self._calculate_risk_score(
                                clause_type, text, clause_text
                            )

                            processed_data.append({
                                'text': clause_text,
                                'clause_text': clause_text,
                                'clause_type': clause_type,
                                'risk_score': risk_score,
                                'risk_category': self._get_risk_category(risk_score),
                                'source': 'ledgar'
                            })

            # Convert to dataset
            df = pd.DataFrame(processed_data)
            dataset = Dataset.from_pandas(df)

            logger.info(f"LEDGAR dataset loaded: {len(dataset)} samples")
            return dataset

        except Exception as e:
            logger.error(f"Error loading LEDGAR dataset: {e}")
            raise RuntimeError(f"Failed to load LEDGAR dataset: {e}")

    def combine_datasets(self) -> DatasetDict:
        """Combine CUAD and LEDGAR datasets and create train/val/test splits.

        Returns:
            Combined dataset with train/validation/test splits

        Raises:
            RuntimeError: If dataset combination fails
        """
        try:
            logger.info("Combining datasets...")

            # Load individual datasets
            cuad_data = self.load_cuad_dataset()
            ledgar_data = self.load_ledgar_dataset()

            # Combine datasets
            combined_data = []

            # Add CUAD data
            for example in cuad_data:
                combined_data.append(example)

            # Add LEDGAR data
            for example in ledgar_data:
                combined_data.append(example)

            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(combined_data)

            # Remove duplicates based on clause text
            df = df.drop_duplicates(subset=['clause_text'])

            # Balance the dataset by risk category
            df = self._balance_dataset(df)

            # Create train/val/test splits
            train_split = self.config.get('data.train_split', 0.7)
            val_split = self.config.get('data.val_split', 0.15)
            test_split = self.config.get('data.test_split', 0.15)

            # First split: separate test set
            train_val_df, test_df = train_test_split(
                df,
                test_size=test_split,
                stratify=df['risk_category'],
                random_state=42
            )

            # Second split: separate train and validation
            val_size = val_split / (train_split + val_split)
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_size,
                stratify=train_val_df['risk_category'],
                random_state=42
            )

            # Convert back to datasets
            train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
            val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
            test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

            # Create DatasetDict
            dataset_dict = DatasetDict({
                'train': train_dataset,
                'validation': val_dataset,
                'test': test_dataset
            })

            logger.info(f"Dataset splits created:")
            logger.info(f"  Train: {len(train_dataset)} samples")
            logger.info(f"  Validation: {len(val_dataset)} samples")
            logger.info(f"  Test: {len(test_dataset)} samples")

            return dataset_dict

        except Exception as e:
            logger.error(f"Error combining datasets: {e}")
            raise RuntimeError(f"Failed to combine datasets: {e}")

    def _extract_clause_type(self, question: str) -> str:
        """Extract clause type from CUAD question.

        Args:
            question: Question text from CUAD dataset

        Returns:
            Clause type identifier
        """
        # Common clause type patterns in CUAD questions
        clause_patterns = {
            'termination': ['terminat', 'dismiss', 'end'],
            'non_compete': ['compet', 'restrict', 'solicit'],
            'intellectual_property': ['intellectual', 'patent', 'copyright', 'invention'],
            'confidentiality': ['confidential', 'proprietary', 'secret'],
            'compensation': ['compensat', 'salary', 'payment', 'benefit'],
            'dispute_resolution': ['dispute', 'arbitrat', 'jurisdiction', 'governing']
        }

        question_lower = question.lower()

        for clause_type, patterns in clause_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                return clause_type

        return 'general'

    def _extract_clause_context(self, context: str, answer_text: str, start_idx: int) -> str:
        """Extract surrounding context for a clause.

        Args:
            context: Full contract text
            answer_text: The clause answer text
            start_idx: Starting index of the answer in context

        Returns:
            Context text around the clause
        """
        # Extract context window around the clause
        window_size = 200  # characters before and after

        start = max(0, start_idx - window_size)
        end = min(len(context), start_idx + len(answer_text) + window_size)

        return context[start:end].strip()

    def _extract_clauses_from_text(self, text: str) -> List[str]:
        """Extract individual clauses from legal document text.

        Args:
            text: Full legal document text

        Returns:
            List of extracted clause texts
        """
        # Split text into clauses based on common patterns
        # This is a simplified approach - in production, you might use more sophisticated NLP
        clause_separators = [
            r'\n\s*\d+\.',  # Numbered clauses
            r'\n\s*\([a-z]\)',  # Lettered subclauses
            r'\n\s*[A-Z][A-Z\s]+\.',  # All caps section headers
            r';\s*provided',  # Legal provisos
            r';\s*except',  # Legal exceptions
        ]

        # Split by separators
        clauses = [text]
        for separator in clause_separators:
            new_clauses = []
            for clause in clauses:
                parts = re.split(separator, clause)
                new_clauses.extend([p.strip() for p in parts if p.strip()])
            clauses = new_clauses

        # Filter and clean clauses
        cleaned_clauses = []
        for clause in clauses:
            # Remove very short or very long clauses
            if 50 <= len(clause) <= 1000:
                cleaned_clauses.append(clause.strip())

        return cleaned_clauses[:10]  # Limit to first 10 clauses to avoid too much data

    def _infer_clause_type(self, clause_text: str) -> str:
        """Infer clause type from clause content.

        Args:
            clause_text: The clause text to analyze

        Returns:
            Inferred clause type
        """
        text_lower = clause_text.lower()

        # Use risk categories from config to infer type
        for category, config in self.risk_categories.items():
            keywords = config.get('keywords', [])
            if any(keyword in text_lower for keyword in keywords):
                return category

        return 'general'

    def _calculate_risk_score(
        self,
        clause_type: str,
        context: str,
        clause_text: str
    ) -> float:
        """Calculate risk score for a clause.

        Args:
            clause_type: Type of the clause
            context: Surrounding context
            clause_text: The clause text

        Returns:
            Risk score between 1.0 and 10.0
        """
        base_score = 5.0  # Neutral score

        # Get clause type weight
        type_config = self.risk_categories.get(clause_type, {})
        type_weight = type_config.get('weight', 0.1)

        # Risk indicators (patterns that suggest higher risk)
        risk_indicators = [
            'without cause', 'at will', 'immediate termination',
            'sole discretion', 'absolute', 'unlimited',
            'any reason', 'no notice', 'waive', 'forfeit',
            'perpetual', 'irrevocable', 'exclusive'
        ]

        # Protective indicators (patterns that suggest lower risk)
        protective_indicators = [
            'reasonable notice', 'good cause', 'mutual consent',
            'fair compensation', 'reasonable', 'limited',
            'subject to', 'with exception', 'unless'
        ]

        text_lower = clause_text.lower()

        # Count indicators
        risk_count = sum(1 for indicator in risk_indicators if indicator in text_lower)
        protective_count = sum(1 for indicator in protective_indicators if indicator in text_lower)

        # Calculate score adjustment
        score_adjustment = (risk_count - protective_count) * type_weight * 10

        # Apply adjustment
        final_score = base_score + score_adjustment

        # Clamp to valid range
        return max(1.0, min(10.0, final_score))

    def _get_risk_category(self, risk_score: float) -> str:
        """Convert numerical risk score to categorical label.

        Args:
            risk_score: Numerical risk score (1-10)

        Returns:
            Risk category label
        """
        if risk_score <= 3.5:
            return 'low_risk'
        elif risk_score <= 7.0:
            return 'medium_risk'
        else:
            return 'high_risk'

    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance dataset across risk categories.

        Args:
            df: Input DataFrame

        Returns:
            Balanced DataFrame
        """
        # Get minimum class size
        class_counts = df['risk_category'].value_counts()
        min_count = class_counts.min()

        # Sample from each class
        balanced_dfs = []
        for category in class_counts.index:
            category_df = df[df['risk_category'] == category]
            if len(category_df) > min_count:
                sampled_df = category_df.sample(n=min_count, random_state=42)
            else:
                sampled_df = category_df
            balanced_dfs.append(sampled_df)

        # Combine balanced data
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)

        logger.info(f"Dataset balanced: {len(balanced_df)} samples")
        logger.info(f"Class distribution: {balanced_df['risk_category'].value_counts().to_dict()}")

        return balanced_df

    def tokenize_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Tokenize text data for model input.

        Args:
            dataset: Input dataset

        Returns:
            Tokenized dataset ready for model training
        """
        def tokenize_function(examples: Dict[str, List]) -> Dict[str, List]:
            """Tokenization function for dataset mapping."""
            # Tokenize text
            tokenized = self.tokenizer(
                examples['text'],
                truncation=self.config.get('data.truncation', True),
                padding=self.config.get('data.padding', 'max_length'),
                max_length=self.config.get('data.max_sequence_length', 512),
                return_tensors=None
            )

            # Add labels
            # Convert risk categories to numerical labels
            label_map = {'low_risk': 0, 'medium_risk': 1, 'high_risk': 2}
            tokenized['labels'] = [label_map[cat] for cat in examples['risk_category']]

            # Add risk scores as regression targets
            tokenized['risk_scores'] = examples['risk_score']

            return tokenized

        logger.info("Tokenizing datasets...")

        # Apply tokenization to all splits
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )

        logger.info("Tokenization completed")
        return tokenized_dataset

    def get_dataset_statistics(self, dataset: DatasetDict) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get comprehensive statistics about the dataset.

        Args:
            dataset: Input dataset

        Returns:
            Dictionary containing dataset statistics
        """
        stats = {}

        for split_name, split_data in dataset.items():
            split_stats = {
                'total_samples': len(split_data),
                'avg_text_length': np.mean([len(text) for text in split_data['text']]),
                'avg_risk_score': np.mean(split_data['risk_score']),
                'risk_category_distribution': {}
            }

            # Calculate category distribution
            categories = split_data['risk_category']
            for category in ['low_risk', 'medium_risk', 'high_risk']:
                count = categories.count(category)
                split_stats['risk_category_distribution'][category] = {
                    'count': count,
                    'percentage': (count / len(categories)) * 100
                }

            stats[split_name] = split_stats

        return stats


def create_data_loader(config: Config) -> LegalDataLoader:
    """Factory function to create a data loader instance.

    Args:
        config: Configuration object

    Returns:
        Initialized data loader
    """
    return LegalDataLoader(config)