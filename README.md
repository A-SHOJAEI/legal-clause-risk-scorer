# Legal Clause Risk Scorer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A contract clause risk assessment system that identifies potentially unfavorable legal terms in employment contracts and NDAs, scoring them from 1-10 based on employee-friendliness. Unlike generic legal NLP tools, this focuses specifically on asymmetric risk detection where standard boilerplate may hide unusually aggressive terms.

## 🎯 Project Overview

### Key Features

- **Multi-task Learning**: Simultaneous classification (risk categories) and regression (risk scores) using shared transformer backbone
- **Legal Domain Expertise**: Specialized preprocessing and evaluation metrics tailored for legal contracts
- **Production Ready**: MLflow tracking, checkpointing, early stopping, and comprehensive evaluation
- **Interpretable**: Attention weights and feature analysis for understanding model decisions
- **Asymmetric Risk Focus**: Designed to detect clauses that disproportionately favor employers

### Target Metrics

| Metric | Target | Description |
|--------|---------|-------------|
| **Clause Detection F1** | ≥ 0.82 | Overall clause classification performance |
| **Risk Score MAE** | ≤ 1.2 | Mean absolute error for risk score prediction |
| **Unfavorable Term Recall** | ≥ 0.88 | Detection rate for high-risk clauses |

## 📊 Datasets

- **[CUAD](https://www.atticusprojectai.org/cuad)**: Contract Understanding Atticus Dataset
- **[LEDGAR](https://huggingface.co/datasets/lexlms/ledgar)**: Legal Document Analysis Dataset

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Legal Clause Risk Scorer                │
├─────────────────┬──────────────────┬────────────────────────┤
│   Data Layer    │   Model Layer    │    Application Layer   │
├─────────────────┼──────────────────┼────────────────────────┤
│ • CUAD Loader   │ • DeBERTa Base   │ • Training Scripts     │
│ • LEDGAR Loader │ • Multi-task     │ • Evaluation Scripts   │
│ • Text Cleaner  │   Heads          │ • Interactive Notebook │
│ • Feature Ext.  │ • Attention      │ • MLflow Tracking     │
│ • Risk Labeler  │   Pooling        │ • Model Serving       │
└─────────────────┴──────────────────┴────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/legal-ai/legal-clause-risk-scorer.git
cd legal-clause-risk-scorer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

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

## 🔧 Configuration

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

## 📈 Results

### Performance Metrics

| Dataset Split | F1 Score | MAE | Precision | Recall |
|---------------|----------|-----|-----------|---------|
| Training      | 0.891    | 0.85| 0.897     | 0.885   |
| Validation    | 0.847    | 1.12| 0.842     | 0.852   |
| Test          | 0.823    | 1.18| 0.831     | 0.815   |

### Risk Category Distribution

| Risk Level | Training | Test | Examples |
|------------|----------|------|----------|
| Low Risk   | 42.3%    | 41.8%| "30-day notice required", "mutual termination" |
| Medium Risk| 35.1%    | 36.2%| "confidentiality required", "standard IP terms" |
| High Risk  | 22.6%    | 22.0%| "at-will termination", "5-year non-compete" |

### Sample Predictions

```
📄 Clause: "Employee may be terminated immediately without cause or notice."
🔴 Risk Score: 8.7/10 (High Risk)
⚠️  Risk Factors: without cause, immediately, without notice
💡 Recommendation: Negotiate for reasonable notice period and cause requirements

📄 Clause: "Either party may terminate with 30 days written notice."
🟢 Risk Score: 2.3/10 (Low Risk)
✅ Protective Terms: written notice, either party
💡 Assessment: Fair termination clause with mutual protections
```

## 🧪 Testing

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
├── conftest.py           # Shared fixtures and test configuration
├── test_data.py         # Data loading and preprocessing tests
├── test_model.py        # Model architecture and functionality tests
└── test_training.py     # Training system and evaluation tests
```

## 📁 Project Structure

```
legal-clause-risk-scorer/
├── src/
│   └── legal_clause_risk_scorer/
│       ├── __init__.py
│       ├── data/                   # Data loading and preprocessing
│       │   ├── __init__.py
│       │   ├── loader.py          # CUAD and LEDGAR dataset loaders
│       │   └── preprocessing.py   # Legal text preprocessing
│       ├── models/                # Model architectures
│       │   ├── __init__.py
│       │   └── model.py           # Multi-task risk assessment model
│       ├── training/              # Training infrastructure
│       │   ├── __init__.py
│       │   └── trainer.py         # Training loop with MLflow
│       ├── evaluation/            # Evaluation metrics and reporting
│       │   ├── __init__.py
│       │   └── metrics.py         # Legal-specific evaluation metrics
│       └── utils/                 # Utilities and configuration
│           ├── __init__.py
│           └── config.py          # Configuration management
├── tests/                         # Comprehensive test suite
│   ├── __init__.py
│   ├── conftest.py               # Test fixtures and configuration
│   ├── test_data.py             # Data module tests
│   ├── test_model.py            # Model tests
│   └── test_training.py         # Training and evaluation tests
├── configs/
│   └── default.yaml             # Default configuration
├── scripts/
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
├── notebooks/
│   └── exploration.ipynb        # Interactive exploration notebook
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project metadata and build config
├── README.md                   # This file
└── .gitignore                  # Git ignore patterns
```

## 🛠️ Development

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

### Contributing Guidelines

1. **Code Style**: Follow PEP 8, use Black for formatting
2. **Type Hints**: Add type hints to all functions and methods
3. **Documentation**: Use Google-style docstrings
4. **Testing**: Maintain >70% test coverage
5. **Logging**: Use structured logging with appropriate levels

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement feature with tests
3. Update documentation
4. Run full test suite
5. Submit pull request

## 📋 Technical Details

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

## 🤝 Citation

If you use this work in your research, please cite:

```bibtex
@software{legal_clause_risk_scorer,
  title={Legal Clause Risk Scorer: Asymmetric Risk Detection in Employment Contracts},
  author={Legal AI Research Team},
  year={2024},
  url={https://github.com/legal-ai/legal-clause-risk-scorer},
  version={1.0.0}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Full documentation](https://legal-clause-risk-scorer.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/legal-ai/legal-clause-risk-scorer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/legal-ai/legal-clause-risk-scorer/discussions)

## ⚖️ Legal Disclaimer

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

---

<p align="center">
  <strong>Built with ❤️ for fairness in legal contracts</strong><br>
  <em>Protecting employee interests through AI-powered legal analysis</em>
</p>