"""Data preprocessing utilities for legal text analysis.

This module provides comprehensive preprocessing capabilities for legal documents,
including text cleaning, normalization, feature extraction, and augmentation.
"""

import logging
import re
import string
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import AutoTokenizer

from ..utils.config import Config


logger = logging.getLogger(__name__)


class LegalTextPreprocessor:
    """Comprehensive text preprocessor for legal documents.

    This class provides various text preprocessing capabilities specifically
    designed for legal documents, including cleaning, normalization, and
    feature extraction.

    Attributes:
        config: Configuration object
        nlp: spaCy language model
        tokenizer: Hugging Face tokenizer
        legal_stopwords: Set of legal-specific stopwords
        legal_abbreviations: Dictionary of common legal abbreviations
    """

    def __init__(self, config: Config):
        """Initialize the text preprocessor.

        Args:
            config: Configuration object
        """
        self.config = config

        # Initialize spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Some features may be limited.")
            self.nlp = None

        # Initialize tokenizer
        model_name = config.get('model.base_model', 'microsoft/deberta-v3-base')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Legal-specific preprocessing resources
        self._initialize_legal_resources()

        logger.info("Legal text preprocessor initialized")

    def _initialize_legal_resources(self) -> None:
        """Initialize legal-specific preprocessing resources."""
        # Legal stopwords (beyond standard English stopwords)
        self.legal_stopwords = {
            'party', 'parties', 'agreement', 'contract', 'document',
            'section', 'subsection', 'paragraph', 'clause', 'provision',
            'thereof', 'hereof', 'herein', 'hereto', 'hereunder',
            'whereas', 'whereby', 'whereof', 'witnesseth'
        }

        # Common legal abbreviations and their expansions
        self.legal_abbreviations = {
            'inc\\.': 'incorporated',
            'corp\\.': 'corporation',
            'ltd\\.': 'limited',
            'llc': 'limited liability company',
            'co\\.': 'company',
            'vs\\.': 'versus',
            'et al\\.': 'and others',
            'i\\.e\\.': 'that is',
            'e\\.g\\.': 'for example',
            'etc\\.': 'and so forth',
            'cf\\.': 'compare',
            'id\\.': 'the same',
            'op\\. cit\\.': 'in the work cited',
            'loc\\. cit\\.': 'in the place cited',
            'supra': 'above',
            'infra': 'below'
        }

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

    def clean_legal_text(self, text: str) -> str:
        """Clean and normalize legal text.

        Args:
            text: Input legal text

        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""

        # Store original for debugging
        original_length = len(text)

        # Basic cleaning
        text = self._remove_special_formatting(text)
        text = self._normalize_whitespace(text)
        text = self._expand_legal_abbreviations(text)
        text = self._normalize_citations(text)
        text = self._remove_boilerplate(text)

        logger.debug(f"Text cleaned: {original_length} -> {len(text)} characters")
        return text.strip()

    def _remove_special_formatting(self, text: str) -> str:
        """Remove special formatting and artifacts from legal documents.

        Args:
            text: Input text

        Returns:
            Text with formatting removed
        """
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Remove excessive punctuation
        text = re.sub(r'[_]{3,}', ' ', text)
        text = re.sub(r'[-]{3,}', ' ', text)
        text = re.sub(r'[.]{3,}', '...', text)

        # Remove table-like structures
        text = re.sub(r'\n\s*\|.*?\|\s*\n', '\n', text)

        # Remove signature blocks
        text = re.sub(
            r'\n\s*_+\s*\n.*?signature.*?\n',
            '\n',
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.

        Args:
            text: Input text

        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)

        # Replace multiple newlines with single newline
        text = re.sub(r'\n\s*\n+', '\n\n', text)

        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)

        return text

    def _expand_legal_abbreviations(self, text: str) -> str:
        """Expand common legal abbreviations.

        Args:
            text: Input text

        Returns:
            Text with expanded abbreviations
        """
        for abbrev, expansion in self.legal_abbreviations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + abbrev + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)

        return text

    def _normalize_citations(self, text: str) -> str:
        """Normalize legal citations and references.

        Args:
            text: Input text

        Returns:
            Text with normalized citations
        """
        # Normalize section references
        text = re.sub(r'§\s*(\d+)', r'Section \1', text)
        text = re.sub(r'Sec\.\s*(\d+)', r'Section \1', text)

        # Normalize subsection references
        text = re.sub(r'(\d+)\s*\(([a-z])\)', r'\1(\2)', text)

        # Normalize case citations (simplified)
        text = re.sub(
            r'(\w+)\s+v\.\s+(\w+),?\s+\d+.*?\(\d{4}\)',
            r'\1 versus \2',
            text
        )

        return text

    def _remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate language from legal documents.

        Args:
            text: Input text

        Returns:
            Text with boilerplate removed
        """
        # Common boilerplate patterns
        boilerplate_patterns = [
            r'this agreement is entered into.*?between.*?\.',
            r'in witness whereof.*?executed.*?\.',
            r'the parties hereto.*?agree.*?follows:?',
            r'now, therefore.*?consideration.*?agree.*?follows:?'
        ]

        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

        return text

    def extract_features(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Extract various features from legal texts.

        Args:
            texts: List of input texts

        Returns:
            Dictionary of feature arrays
        """
        logger.info(f"Extracting features from {len(texts)} texts")

        features = {}

        # Basic text statistics
        features['text_length'] = np.array([len(text) for text in texts])
        features['word_count'] = np.array([len(text.split()) for text in texts])
        features['sentence_count'] = np.array([len(nltk.sent_tokenize(text)) for text in texts])

        # Legal-specific features
        features['legal_terms_count'] = np.array([
            self._count_legal_terms(text) for text in texts
        ])
        features['risk_keywords_count'] = np.array([
            self._count_risk_keywords(text) for text in texts
        ])
        features['modal_verbs_count'] = np.array([
            self._count_modal_verbs(text) for text in texts
        ])

        # Readability features
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
        features['avg_word_length'] = np.array([
            np.mean([len(word) for word in text.split()]) if text.split() else 0
            for text in texts
        ])

        # TF-IDF features (limited to top features for efficiency)
        tfidf_features = self._extract_tfidf_features(texts)
        features['tfidf'] = tfidf_features

        logger.info(f"Extracted {len(features)} feature types")
        return features

    def _count_legal_terms(self, text: str) -> int:
        """Count legal terms in text.

        Args:
            text: Input text

        Returns:
            Number of legal terms found
        """
        legal_terms = [
            'contract', 'agreement', 'liability', 'obligation', 'covenant',
            'warranty', 'indemnity', 'breach', 'damages', 'remedy',
            'jurisdiction', 'governing', 'enforce', 'terminate', 'void',
            'null', 'binding', 'consideration', 'performance', 'default'
        ]

        text_lower = text.lower()
        return sum(1 for term in legal_terms if term in text_lower)

    def _count_risk_keywords(self, text: str) -> int:
        """Count risk-indicating keywords in text.

        Args:
            text: Input text

        Returns:
            Number of risk keywords found
        """
        risk_keywords = [
            'without cause', 'sole discretion', 'at will', 'immediate',
            'unlimited', 'perpetual', 'irrevocable', 'waive', 'forfeit',
            'exclude', 'prohibit', 'restrict', 'penalty', 'forfeitur'
        ]

        text_lower = text.lower()
        return sum(1 for keyword in risk_keywords if keyword in text_lower)

    def _count_modal_verbs(self, text: str) -> int:
        """Count modal verbs indicating obligation or possibility.

        Args:
            text: Input text

        Returns:
            Number of modal verbs found
        """
        modal_verbs = [
            'shall', 'must', 'may', 'might', 'should', 'could',
            'would', 'will', 'can', 'ought'
        ]

        words = text.lower().split()
        return sum(1 for word in words if word in modal_verbs)

    def _extract_tfidf_features(
        self,
        texts: List[str],
        max_features: int = 100
    ) -> np.ndarray:
        """Extract TF-IDF features from texts.

        Args:
            texts: List of input texts
            max_features: Maximum number of TF-IDF features

        Returns:
            TF-IDF feature matrix
        """
        # Clean texts for TF-IDF
        cleaned_texts = [self._preprocess_for_tfidf(text) for text in texts]

        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.8  # Maximum document frequency
        )

        try:
            # Fit and transform texts
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            return tfidf_matrix.toarray()

        except ValueError as e:
            logger.warning(f"TF-IDF extraction failed: {e}")
            # Return zero matrix if TF-IDF fails
            return np.zeros((len(texts), max_features))

    def _preprocess_for_tfidf(self, text: str) -> str:
        """Preprocess text specifically for TF-IDF extraction.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def augment_data(
        self,
        texts: List[str],
        labels: List[Union[str, int]],
        augment_ratio: float = 0.2
    ) -> Tuple[List[str], List[Union[str, int]]]:
        """Augment training data to improve model robustness.

        Args:
            texts: List of input texts
            labels: List of corresponding labels
            augment_ratio: Ratio of original data to generate as augmented data

        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        logger.info(f"Augmenting data with ratio {augment_ratio}")

        augmented_texts = texts.copy()
        augmented_labels = labels.copy()

        num_to_augment = int(len(texts) * augment_ratio)

        for i in range(num_to_augment):
            # Select random text to augment
            idx = np.random.randint(0, len(texts))
            original_text = texts[idx]
            original_label = labels[idx]

            # Apply augmentation
            augmented_text = self._apply_text_augmentation(original_text)

            augmented_texts.append(augmented_text)
            augmented_labels.append(original_label)

        logger.info(f"Augmented dataset: {len(texts)} -> {len(augmented_texts)} samples")
        return augmented_texts, augmented_labels

    def _apply_text_augmentation(self, text: str) -> str:
        """Apply text augmentation techniques.

        Args:
            text: Input text

        Returns:
            Augmented text
        """
        # Randomly choose augmentation technique
        techniques = [
            self._synonym_replacement,
            self._random_insertion,
            self._random_swap,
            self._paraphrase_substitution
        ]

        technique = np.random.choice(techniques)
        return technique(text)

    def _synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms.

        Args:
            text: Input text

        Returns:
            Text with some words replaced by synonyms
        """
        # Simple synonym replacement using a predefined dictionary
        legal_synonyms = {
            'agreement': 'contract',
            'contract': 'agreement',
            'party': 'entity',
            'entity': 'party',
            'obligation': 'duty',
            'duty': 'obligation',
            'terminate': 'end',
            'end': 'terminate'
        }

        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in legal_synonyms and np.random.random() < 0.1:
                words[i] = legal_synonyms[word.lower()]

        return ' '.join(words)

    def _random_insertion(self, text: str) -> str:
        """Insert random legal terms.

        Args:
            text: Input text

        Returns:
            Text with randomly inserted terms
        """
        insertion_terms = ['furthermore', 'additionally', 'moreover', 'specifically']

        words = text.split()
        if len(words) > 5:  # Only insert in longer texts
            insert_pos = np.random.randint(1, len(words))
            term = np.random.choice(insertion_terms)
            words.insert(insert_pos, term)

        return ' '.join(words)

    def _random_swap(self, text: str) -> str:
        """Randomly swap adjacent words.

        Args:
            text: Input text

        Returns:
            Text with some words swapped
        """
        words = text.split()
        if len(words) > 3:
            # Swap 1-2 pairs of adjacent words
            num_swaps = min(2, len(words) // 4)
            for _ in range(num_swaps):
                idx = np.random.randint(0, len(words) - 1)
                words[idx], words[idx + 1] = words[idx + 1], words[idx]

        return ' '.join(words)

    def _paraphrase_substitution(self, text: str) -> str:
        """Substitute common legal phrases with paraphrases.

        Args:
            text: Input text

        Returns:
            Text with paraphrased phrases
        """
        paraphrases = {
            'in accordance with': 'pursuant to',
            'pursuant to': 'in accordance with',
            'subject to': 'conditional upon',
            'conditional upon': 'subject to',
            'with respect to': 'regarding',
            'regarding': 'with respect to'
        }

        for original, paraphrase in paraphrases.items():
            if original in text.lower() and np.random.random() < 0.3:
                text = re.sub(
                    re.escape(original),
                    paraphrase,
                    text,
                    flags=re.IGNORECASE
                )

        return text

    def validate_preprocessing(self, texts: List[str]) -> Dict[str, Union[bool, str]]:
        """Validate preprocessing results.

        Args:
            texts: List of preprocessed texts

        Returns:
            Validation results
        """
        validation_results = {
            'all_texts_valid': True,
            'issues': []
        }

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                validation_results['all_texts_valid'] = False
                validation_results['issues'].append(f"Text {i} is not a string")

            if len(text.strip()) == 0:
                validation_results['all_texts_valid'] = False
                validation_results['issues'].append(f"Text {i} is empty after preprocessing")

            if len(text) > 10000:  # Arbitrary large limit
                validation_results['issues'].append(f"Text {i} is very long ({len(text)} chars)")

        logger.info(f"Preprocessing validation: {validation_results}")
        return validation_results


def create_preprocessor(config: Config) -> LegalTextPreprocessor:
    """Factory function to create a text preprocessor instance.

    Args:
        config: Configuration object

    Returns:
        Initialized text preprocessor
    """
    return LegalTextPreprocessor(config)


def preprocess_dataset_batch(
    texts: List[str],
    config: Config,
    clean_only: bool = False
) -> List[str]:
    """Preprocess a batch of texts with configuration.

    Args:
        texts: List of input texts
        config: Configuration object
        clean_only: If True, only apply cleaning without feature extraction

    Returns:
        List of preprocessed texts
    """
    preprocessor = create_preprocessor(config)

    if clean_only:
        return [preprocessor.clean_legal_text(text) for text in texts]
    else:
        # Apply full preprocessing pipeline
        cleaned_texts = [preprocessor.clean_legal_text(text) for text in texts]
        return cleaned_texts