# Contributing to ECG ML Classification

Thank you for your interest in contributing! This project aims to provide a simple, educational implementation of ECG classification using deep learning.

## Development Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd ecg-ml
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download PTB-XL dataset:**
```bash
# Option 1: Full dataset (~3GB)
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
mv physionet.org/files/ptb-xl/1.0.3/* data/ptbxl/

# Option 2: Minimal files for development
mkdir -p data/ptbxl
wget -O data/ptbxl/ptbxl_database.csv https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv
wget -O data/ptbxl/scp_statements.csv https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv
```

## Code Style

- Use **black** for code formatting: `black src/`
- Keep functions focused and well-documented
- Add type hints where helpful
- Follow existing naming conventions

## Testing

Run basic tests:
```bash
cd src
python -c "import models, data, metrics; print('✓ All imports work')"
```

## Project Structure

```
src/
├── data.py      # Dataset loading and preprocessing  
├── models.py    # CNN architectures (1D and 2D)
├── train.py     # Training with cross-validation
├── eval.py      # Model evaluation with metrics
├── interpret.py # Saliency map generation
└── metrics.py   # Evaluation metrics computation
```

## Contribution Guidelines

1. **Keep it simple**: This is an educational project - avoid over-engineering
2. **Minimal dependencies**: Only add dependencies if truly necessary
3. **Educational value**: Code should be readable and well-commented
4. **Performance**: Maintain lightweight, fast training
5. **Clinical relevance**: Changes should improve diagnostic accuracy

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with clear commit messages
4. Ensure code passes basic import tests
5. Update documentation if needed
6. Submit a pull request with description of changes

## Questions?

Open an issue for:
- Bug reports
- Feature requests  
- Documentation improvements
- Questions about the code

We aim to keep this project accessible to ML students and researchers working on medical applications!
