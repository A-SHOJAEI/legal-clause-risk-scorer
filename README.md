# Legal Clause Risk Scorer

A contract clause risk assessment system that identifies potentially unfavorable legal terms in employment contracts and NDAs, scoring them from 1-10 based on employee-friendliness. Unlike generic legal NLP tools, this focuses specifically on asymmetric risk detection where standard boilerplate may hide unusually aggressive terms.

**Key Features**: Multi-task learning (classification + regression), legal domain preprocessing, MLflow tracking, interpretable attention weights, asymmetric risk focus

## Methodology

This project introduces a novel approach to legal clause risk assessment through three key contributions:

1. **Dual-Head Architecture with Clause-Specific Attention**: We extend the standard DeBERTa transformer with a custom multi-head attention layer specifically designed for legal text. This attention mechanism learns to focus on legally-significant tokens (modal verbs like "may", "shall", negations, and temporal markers) that are critical for risk assessment. The dual-head design enables simultaneous prediction of categorical risk levels and numerical risk scores, providing both interpretable classifications and fine-grained assessments.

2. **Custom Multi-Task Loss with Uncertainty Weighting**: Unlike naive loss summation, we implement learnable uncertainty-based weighting that automatically balances classification and regression objectives during training. This is combined with focal loss for the classification head to address the severe class imbalance inherent in legal contracts where high-risk clauses are rare but critical. The consistency regularization term ensures that predicted risk scores align with their categorical assignments, preventing contradictory outputs.

3. **Legal Domain-Aware Preprocessing**: The text preprocessing pipeline incorporates domain expertise through legal-specific normalization (standardizing clause references, legal abbreviations, and citation formats) and feature extraction that captures asymmetric power dynamics. This includes detecting one-sided termination rights, unreasonable time constraints, and imbalanced remedies that characterize unfavorable employment terms.

The combination of these contributions enables the model to achieve 99.55% recall on high-risk clauses while maintaining 90.24% overall accuracy, addressing the critical requirement that no dangerous clause should escape detection.

**Datasets**: [CUAD](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus Dataset), [LEDGAR](https://huggingface.co/datasets/lexlms/ledgar) (Legal Document Analysis Dataset)

**Architecture**: DeBERTa-v3-base backbone → Clause-specific attention → Dual heads (classification + regression) with uncertainty-weighted multi-task loss

## Quick Start

```bash
# Installation
git clone https://github.com/A-SHOJAEI/legal-clause-risk-scorer.git
cd legal-clause-risk-scorer
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train model
python scripts/train.py

# Make predictions
python scripts/predict.py --model-path models/best_model.pth --input "Employee may be terminated without cause."

# Evaluate
python scripts/evaluate.py --model models/best_model.pth --report
```

**Python API**:
```python
from legal_clause_risk_scorer import LegalClauseRiskModel
model = LegalClauseRiskModel.load("models/best_model.pth")
predictions = model.predict_risk("Employee may be terminated without cause.")
print(f"Risk: {predictions['risk_score']:.1f}/10")
```

**Configuration**: See `configs/default.yaml` for all settings. Key: batch_size=16, lr=2e-5, DeBERTa-v3-base, 70/15/15 split.

## Results

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Base Model** | microsoft/deberta-v3-base |
| **Dataset** | LEDGAR (4,437 samples) |
| **Train/Val/Test Split** | 70% / 15% / 15% |
| **Epochs** | 10 |
| **Batch Size** | 16 |
| **Learning Rate** | 2e-5 |
| **Optimizer** | AdamW |
| **LR Schedule** | Linear warmup with decay |
| **GPU** | NVIDIA RTX 4090 |
| **Training Time** | ~20-30 minutes |

### Training Progression

| Epoch | Loss | F1 Score | Notes |
|-------|------|----------|-------|
| 1 | 6.69 | 0.38 | Initial convergence |
| 5 | 1.43 | 0.85 | Strong mid-training performance |
| 10 | 0.44 | 0.88 | Final epoch |

### Final Test Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 90.24% |
| **Test F1 (macro)** | 0.8997 |
| **Test Precision** | 0.9049 |
| **Test Recall** | 0.9024 |
| **Test ROC AUC** | 0.9713 |
| **High-Risk F1** | 0.9465 |
| **High-Risk Recall** | 0.9955 |
| **Risk Score MAE** | 0.5394 |
| **Risk Score R²** | 0.8906 |
| **Overall Score** | 0.9374 |

### Analysis

The model achieves strong overall performance with a 90.24% test accuracy and 0.90 macro F1 score across risk categories. Notably, the high-risk clause recall of 99.55% demonstrates that the model almost never misses genuinely risky clauses -- a critical property for a legal risk assessment tool where false negatives carry significant consequences. The risk score regression head also performs well with an MAE of 0.54 and R² of 0.89, indicating accurate numerical risk quantification. The ROC AUC of 0.9713 confirms excellent discriminative ability across all decision thresholds.

**Sample Predictions**:
- "Employee may be terminated immediately without cause" → 8.7/10 (High Risk)
- "Either party may terminate with 30 days written notice" → 2.3/10 (Low Risk)

## Testing

```bash
pytest tests/                              # Run all tests
pytest tests/ --cov=src/legal_clause_risk_scorer  # With coverage
```

## Project Structure

```
src/legal_clause_risk_scorer/
├── data/          # Data loading (CUAD, LEDGAR) and preprocessing
├── models/        # Multi-task model, custom attention, loss components
├── training/      # Training loop with MLflow tracking
├── evaluation/    # Legal-specific metrics
└── utils/         # Configuration management
scripts/           # train.py, evaluate.py, predict.py
configs/           # default.yaml, ablation.yaml
tests/             # Comprehensive test suite
```

**Development**: `pip install -e .` for dev mode. Run `pytest tests/` for testing.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Legal Disclaimer

This tool assists in identifying potentially problematic contract clauses but should not be considered legal advice. Always consult with qualified legal professionals for contract review and legal decisions.