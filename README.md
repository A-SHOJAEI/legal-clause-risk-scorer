# Legal Clause Risk Scorer


A contract clause risk assessment system that identifies potentially unfavorable legal terms in employment contracts and NDAs, scoring them from 1-10 based on employee-friendliness. Unlike generic legal NLP tools, this focuses specifically on asymmetric risk detection where standard boilerplate may hide unusually aggressive terms.

## Project Overview

### Key Features

- **Multi-task Learning**: Simultaneous classification (risk categories) and regression (risk scores) using shared transformer backbone
- **Legal Domain Expertise**: Specialized preprocessing and evaluation metrics tailored for legal contracts
- **Production Ready**: MLflow tracking, checkpointing, early stopping, and comprehensive evaluation
- **Interpretable**: Attention weights and feature analysis for understanding model decisions
- **Asymmetric Risk Focus**: Designed to detect clauses that disproportionately favor employers

### Achieved Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Test Accuracy** | 90.24% | Overall clause classification accuracy |
| **Test F1 (macro)** | 0.8997 | Macro-averaged F1 across risk categories |
| **High-Risk Recall** | 0.9955 | Detection rate for high-risk clauses |
| **Risk Score MAE** | 0.5394 | Mean absolute error for risk score prediction |
| **Risk Score R²** | 0.8906 | Coefficient of determination for risk scores |
| **Overall Score** | 0.9374 | Combined multi-task evaluation score |

## Datasets

- **[CUAD](https://www.atticusprojectai.org/cuad)**: Contract Understanding Atticus Dataset
- **[LEDGAR](https://huggingface.co/datasets/lexlms/ledgar)**: Legal Document Analysis Dataset

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Legal Clause Risk Scorer │
├─────────────────┬──────────────────┬────────────────────────┤
│ Data Layer │ Model Layer │ Application Layer │
├─────────────────┼──────────────────┼────────────────────────┤
│ • CUAD Loader │ • DeBERTa Base │ • Training Scripts │
│ • LEDGAR Loader │ • Multi-task │ • Evaluation Scripts │
│ • Text Cleaner │ Heads │ • Interactive Notebook │
│ • Feature Ext. │ • Attention │ • MLflow Tracking │
│ • Risk Labeler │ Pooling │ • Model Serving │
└─────────────────┴──────────────────┴────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/A-SHOJAEI/legal-clause-risk-scorer.git
cd legal-clause-risk-scorer

# Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

#### 1. Train a Model

```bash
# Train with default configuration
python scripts/train.py

# Train with custom configuration
python scripts/train.py --config configs/custom.yaml

# Resume from checkpoint
python scripts/train.py --resume models/checkpoint_epoch_5.pth
```

#### 2. Evaluate Performance

```bash
# Evaluate on test set
python scripts/evaluate.py --model models/best_model.pth --report

# Evaluate on custom data
python scripts/evaluate.py --model models/best_model.pth --data my_clauses.csv --visualizations

# Generate detailed report
python scripts/evaluate.py --model models/best_model.pth --report --output-dir results/
```

#### 3. Interactive Analysis

```bash
# Launch Jupyter notebook for exploration
jupyter notebook notebooks/exploration.ipynb
```

### Python API Usage

```python
from legal_clause_risk_scorer import (
 load_config,
 LegalClauseRiskModel,
 LegalDataLoader,
 LegalTextPreprocessor
)

# Load configuration
config = load_config()

# Initialize components
model = LegalClauseRiskModel(config)
data_loader = LegalDataLoader(config)
preprocessor = LegalTextPreprocessor(config)

# Load and preprocess data
dataset = data_loader.combine_datasets()
tokenized_data = data_loader.tokenize_dataset(dataset)

# Analyze a legal clause
clause = "Employee may be terminated without cause at company's sole discretion."
cleaned_clause = preprocessor.clean_legal_text(clause)

# Get risk assessment (assuming trained model)
predictions = model.predict_risk(input_text=cleaned_clause)
print(f"Risk Category: {predictions['risk_category']}")
print(f"Risk Score: {predictions['risk_score']:.1f}/10")
print(f"Confidence: {predictions['confidence']:.2f}")
```

## Configuration

The system uses YAML configuration files for all settings. Key configuration sections:

### Data Configuration
```yaml
data:
 max_sequence_length: 512
 train_split: 0.7
 val_split: 0.15
 test_split: 0.15
 risk_categories:
 termination_clauses:
 weight: 0.25
 keywords: ["terminate", "dismissal", "at-will"]
```

### Model Configuration
```yaml
model:
 base_model: "microsoft/deberta-v3-base"
 num_labels: 3
 dropout: 0.1
 hidden_size: 768
```

### Training Configuration
```yaml
training:
 batch_size: 16
 learning_rate: 2e-5
 num_epochs: 10
 early_stopping_patience: 3
 use_cuda: true
```

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

### Sample Predictions

```
 Clause: "Employee may be terminated immediately without cause or notice."
 Risk Score: 8.7/10 (High Risk)
 Risk Factors: without cause, immediately, without notice
 Recommendation: Negotiate for reasonable notice period and cause requirements

 Clause: "Either party may terminate with 30 days written notice."
 Risk Score: 2.3/10 (Low Risk)
 Protective Terms: written notice, either party
 Assessment: Fair termination clause with mutual protections
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=src/legal_clause_risk_scorer --cov-report=html

# Run only unit tests (fast)
pytest tests/ -m "unit"

# Run only integration tests
pytest tests/ -m "integration"

# Skip slow tests
pytest tests/ -m "not slow"

# Verbose output
pytest tests/ -v
```

### Test Structure

```
tests/
├── conftest.py # Shared fixtures and test configuration
├── test_data.py # Data loading and preprocessing tests
├── test_model.py # Model architecture and functionality tests
└── test_training.py # Training system and evaluation tests
```

## 📁 Project Structure

```
legal-clause-risk-scorer/
├── src/
│ └── legal_clause_risk_scorer/
│ ├── __init__.py
│ ├── data/ # Data loading and preprocessing
│ │ ├── __init__.py
│ │ ├── loader.py # CUAD and LEDGAR dataset loaders
│ │ └── preprocessing.py # Legal text preprocessing
│ ├── models/ # Model architectures
│ │ ├── __init__.py
│ │ └── model.py # Multi-task risk assessment model
│ ├── training/ # Training infrastructure
│ │ ├── __init__.py
│ │ └── trainer.py # Training loop with MLflow
│ ├── evaluation/ # Evaluation metrics and reporting
│ │ ├── __init__.py
│ │ └── metrics.py # Legal-specific evaluation metrics
│ └── utils/ # Utilities and configuration
│ ├── __init__.py
│ └── config.py # Configuration management
├── tests/ # Comprehensive test suite
│ ├── __init__.py
│ ├── conftest.py # Test fixtures and configuration
│ ├── test_data.py # Data module tests
│ ├── test_model.py # Model tests
│ └── test_training.py # Training and evaluation tests
├── configs/
│ └── default.yaml # Default configuration
├── scripts/
│ ├── train.py # Training script
│ └── evaluate.py # Evaluation script
├── notebooks/
│ └── exploration.ipynb # Interactive exploration notebook
├── requirements.txt # Python dependencies
├── pyproject.toml # Project metadata and build config
├── README.md # This file
└── .gitignore # Git ignore patterns
```

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

## Technical Details

### Model Architecture

- **Backbone**: DeBERTa-v3-base transformer
- **Multi-task Heads**:
 - Classification: 3-class risk categorization
 - Regression: 1-10 risk score prediction
- **Attention Pooling**: Learnable attention weights for sequence representation
- **Legal Projection**: Domain-specific feature projection

### Data Processing Pipeline

1. **Text Cleaning**: Remove formatting artifacts, normalize legal abbreviations
2. **Feature Extraction**: Legal terms, risk keywords, modal verbs, readability metrics
3. **Risk Labeling**: Rule-based + ML hybrid approach for risk score assignment
4. **Tokenization**: Transformer-compatible token sequences with attention masks

### Training Strategy

- **Multi-task Loss**: Weighted combination of classification and regression losses
- **Early Stopping**: Patience-based stopping on validation F1 score
- **Learning Rate**: Linear warmup schedule with decay
- **Regularization**: Dropout, weight decay, gradient clipping

## 🆘 Support

- **Documentation**: [Full documentation](https://legal-clause-risk-scorer.readthedocs.io/)

## ⚖ Legal Disclaimer

**IMPORTANT**: This tool is designed to assist in identifying potentially problematic contract clauses but should not be considered legal advice. Always consult with qualified legal professionals for contract review and legal decisions. The authors and contributors are not liable for any legal consequences arising from the use of this software.

## 🔮 Roadmap

### Upcoming Features

- [ ] **Multi-language Support**: Extend to contracts in other languages
- [ ] **Web Interface**: Django-based web application for non-technical users
- [ ] **API Service**: REST API for integration with legal tech platforms
- [ ] **Clause Comparison**: Side-by-side analysis of contract variations
- [ ] **Negotiation Suggestions**: AI-powered improvement recommendations
- [ ] **Industry Specialization**: Models trained for specific industries
- [ ] **Real-time Alerts**: Monitoring for changes in contract templates

### Long-term Vision

- **Comprehensive Legal AI Platform**: Expand beyond risk scoring to full contract analysis
- **Regulatory Compliance**: Ensure adherence to legal AI guidelines and regulations
- **Academic Partnerships**: Collaborate with law schools and legal research institutions
- **Open Dataset Initiative**: Contribute to open legal NLP datasets


<p align="center">
 <em>Protecting employee interests through AI-powered legal analysis</em>
</p>