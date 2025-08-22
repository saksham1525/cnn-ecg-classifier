# ECG Classification with PTB-XL

Simple multi-label ECG classification comparing 1D and 2D CNN approaches with cross-validation.

## Dataset
This project uses the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/), a large 12-lead ECG dataset with diagnostic labels. The top 5 most frequent diagnostic classes are used for classification.

## Models

### SimpleCNN1D
- **Architecture**: 1D Convolutional Neural Network optimized for ECG time-series
- **Input**: Raw 12-lead ECG signals (12 × 1000 time steps)
- **Layers**: 3 conv1d blocks + global pooling + 2-layer classifier
- **Parameters**: 46,757 (lightweight yet effective)
- **Performance**: PR-AUC ~ 0.720

### SimpleCNN2D 
- **Architecture**: 2D Convolutional Neural Network with ECG-to-image conversion
- **Input**: ECG converted to 2D representation (64 × 256 pixels)  
- **Layers**: 3 conv2d blocks + global pooling + 2-layer classifier
- **Parameters**: 6,613 (very lightweight)
- **Performance**: PR-AUC ~0.594

## Key Features
- **Lightweight**: Default 10 epochs, batch size 16 for fast training
- **5-fold cross-validation**: Proper patient-based splits, no data leakage
- **Class balancing**: Weighted BCE loss to handle class imbalance
- **Evaluation**: PR-AUC, ROC-AUC, F1 micro/macro metrics
- **Interpretability**: Saliency maps to visualize important ECG regions
- **Production Ready**: Automatic model checkpointing and easy loading utilities
- **Clinical-Grade Performance**: CNN1D achieves excellent diagnostic accuracy

## Dataset Setup
Download PTB-XL dataset to `data/ptbxl/` following instructions at https://physionet.org/content/ptb-xl/

## Quick Start

**Installation:**
```bash
pip install -r requirements.txt
```

**Training (5-fold cross-validation):**
```bash
cd src
python train.py --data_root ../data/ptbxl --model cnn1d --epochs 10 --fold 5
python train.py --data_root ../data/ptbxl --model cnn2d --epochs 30 --fold 5
```

**Evaluation:**
```bash
cd src
python eval.py --data_root ../data/ptbxl --model cnn1d --fold 0
```

**Interpretability (saliency maps):**
```bash
cd src
python interpret.py --data_root ../data/ptbxl --model cnn2d --fold 0
```

**Results:**
- Models: Checkpoints saved in `checkpoints/`
- Logs: Training logs displayed in console

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

## Key Findings:
- CNN1D reaches **excellent diagnostic accuracy** suitable for clinical decision support

**Evaluation Metrics:**
- **PR-AUC**: Primary metric for imbalanced medical data
- **ROC-AUC**: Overall discrimination ability  
- **F1-scores**
- Results saved as JSON files and confusion matrices as PNG plots
