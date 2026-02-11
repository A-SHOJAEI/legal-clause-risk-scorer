#!/usr/bin/env python3
"""Prediction script for legal clause risk assessment.

This script loads a trained model and generates risk predictions for legal clauses.
Supports both single clause prediction and batch processing.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.legal_clause_risk_scorer.models.model import LegalClauseRiskModel, load_pretrained_model
from src.legal_clause_risk_scorer.utils.config import Config
from src.legal_clause_risk_scorer.data.preprocessing import LegalTextPreprocessor


def load_model_and_tokenizer(
    model_path: str,
    config_path: Optional[str] = None
) -> tuple:
    """Load trained model and tokenizer.

    Args:
        model_path: Path to the trained model checkpoint
        config_path: Optional path to config file

    Returns:
        Tuple of (model, tokenizer, config)
    """
    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"

    config = Config.from_yaml(str(config_path))

    # Load tokenizer
    base_model = config.get('model.base_model', 'microsoft/deberta-v3-base')
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load model
    model = LegalClauseRiskModel(config)

    # Load state dict
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location='cpu')

    model.load_state_dict(state_dict)

    # Set to evaluation mode
    model.eval()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Model loaded from {model_path}")
    print(f"Using device: {device}")

    return model, tokenizer, config, device


def preprocess_clause(
    clause: str,
    preprocessor: LegalTextPreprocessor
) -> str:
    """Preprocess a legal clause.

    Args:
        clause: Raw clause text
        preprocessor: Text preprocessor

    Returns:
        Cleaned clause text
    """
    return preprocessor.clean_legal_text(clause)


def predict_single(
    model: LegalClauseRiskModel,
    tokenizer: AutoTokenizer,
    clause: str,
    device: torch.device,
    max_length: int = 512
) -> Dict:
    """Make prediction for a single clause.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        clause: Input clause text
        device: Computation device
        max_length: Maximum sequence length

    Returns:
        Dictionary with predictions
    """
    # Tokenize
    inputs = tokenizer(
        clause,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Process outputs
    logits = outputs['logits']
    risk_scores = outputs['risk_scores']

    # Get predicted category
    probs = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
    confidence = torch.max(probs, dim=-1)[0].item()

    category_names = ['low_risk', 'medium_risk', 'high_risk']

    return {
        'clause': clause,
        'risk_category': category_names[predicted_class],
        'risk_score': float(risk_scores.squeeze().item()),
        'confidence': float(confidence),
        'category_probabilities': {
            'low_risk': float(probs[0, 0].item()),
            'medium_risk': float(probs[0, 1].item()),
            'high_risk': float(probs[0, 2].item())
        }
    }


def predict_batch(
    model: LegalClauseRiskModel,
    tokenizer: AutoTokenizer,
    clauses: List[str],
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 512
) -> List[Dict]:
    """Make predictions for a batch of clauses.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        clauses: List of input clauses
        device: Computation device
        batch_size: Batch size for processing
        max_length: Maximum sequence length

    Returns:
        List of prediction dictionaries
    """
    results = []

    for i in range(0, len(clauses), batch_size):
        batch = clauses[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)

        # Process outputs
        logits = outputs['logits']
        risk_scores = outputs['risk_scores']

        probs = torch.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probs, dim=-1)
        confidences = torch.max(probs, dim=-1)[0]

        category_names = ['low_risk', 'medium_risk', 'high_risk']

        # Create results for batch
        for j, clause in enumerate(batch):
            results.append({
                'clause': clause,
                'risk_category': category_names[predicted_classes[j].item()],
                'risk_score': float(risk_scores[j].squeeze().item()),
                'confidence': float(confidences[j].item()),
                'category_probabilities': {
                    'low_risk': float(probs[j, 0].item()),
                    'medium_risk': float(probs[j, 1].item()),
                    'high_risk': float(probs[j, 2].item())
                }
            })

    return results


def format_prediction(prediction: Dict) -> str:
    """Format prediction for display.

    Args:
        prediction: Prediction dictionary

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("LEGAL CLAUSE RISK ASSESSMENT")
    lines.append("=" * 80)
    lines.append(f"\nClause: {prediction['clause']}")
    lines.append(f"\nRisk Category: {prediction['risk_category'].upper().replace('_', ' ')}")
    lines.append(f"Risk Score: {prediction['risk_score']:.2f}/10.0")
    lines.append(f"Confidence: {prediction['confidence']:.2%}")
    lines.append("\nCategory Probabilities:")
    for cat, prob in prediction['category_probabilities'].items():
        lines.append(f"  {cat.replace('_', ' ').title()}: {prob:.2%}")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description="Predict risk scores for legal clauses"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/best_model.pth',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (optional)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input clause text or path to file with clauses'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple clauses from file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for predictions (JSON format)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for processing'
    )

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer, config, device = load_model_and_tokenizer(
        args.model_path,
        args.config
    )

    # Initialize preprocessor
    preprocessor = LegalTextPreprocessor(config)

    # Handle input
    if args.input is None:
        # Use sample clauses for demonstration
        sample_clauses = [
            "Employee may be terminated immediately without cause or notice.",
            "Either party may terminate this agreement with 30 days written notice.",
            "Employee agrees not to compete with the company for 2 years after termination.",
            "All intellectual property created during employment belongs to the company.",
            "Disputes shall be resolved through binding arbitration in accordance with AAA rules."
        ]
        print("No input provided. Using sample clauses for demonstration:\n")
        clauses = sample_clauses
    elif Path(args.input).exists() and args.batch:
        # Load from file
        with open(args.input, 'r') as f:
            clauses = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(clauses)} clauses from {args.input}\n")
    else:
        # Single clause from command line
        clauses = [args.input]

    # Preprocess clauses
    clauses = [preprocess_clause(c, preprocessor) for c in clauses]

    # Make predictions
    if len(clauses) == 1:
        prediction = predict_single(
            model, tokenizer, clauses[0], device, args.max_length
        )
        predictions = [prediction]
    else:
        predictions = predict_batch(
            model, tokenizer, clauses, device, args.batch_size, args.max_length
        )

    # Display results
    for prediction in predictions:
        print(format_prediction(prediction))
        print()

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)

        print(f"\nPredictions saved to {args.output}")

    # Print summary statistics
    if len(predictions) > 1:
        print("\nSUMMARY STATISTICS")
        print("=" * 80)

        risk_scores = [p['risk_score'] for p in predictions]
        categories = [p['risk_category'] for p in predictions]

        print(f"Total clauses analyzed: {len(predictions)}")
        print(f"\nRisk Score Statistics:")
        print(f"  Mean: {np.mean(risk_scores):.2f}")
        print(f"  Median: {np.median(risk_scores):.2f}")
        print(f"  Min: {np.min(risk_scores):.2f}")
        print(f"  Max: {np.max(risk_scores):.2f}")
        print(f"  Std Dev: {np.std(risk_scores):.2f}")

        print(f"\nCategory Distribution:")
        for cat in ['low_risk', 'medium_risk', 'high_risk']:
            count = categories.count(cat)
            pct = count / len(categories) * 100
            print(f"  {cat.replace('_', ' ').title()}: {count} ({pct:.1f}%)")

        print("=" * 80)


if __name__ == "__main__":
    main()
