"""Simple model evaluation script."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from data import PTBXLDataset, get_cv_splits, create_dataloader, get_global_labels
from models import load_model
from metrics import compute_metrics


def evaluate_model(model: torch.nn.Module, dataloader, device: torch.device) -> tuple:
    """Evaluate model and return predictions and targets."""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return np.array(all_preds), np.array(all_targets)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         label_names: list, save_path: Path) -> None:
    """Plot simple confusion matrix for the first label."""
    if len(y_true.shape) > 1:
        # For multi-label, show only first label
        y_true = y_true[:, 0]
        y_pred = y_pred[:, 0]
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - {label_names[0] if label_names else "Label 0"}')
    plt.colorbar()
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate ECG classification model")
    parser.add_argument("--data_root", required=True, help="Path to PTB-XL dataset")
    parser.add_argument("--model", choices=["cnn1d", "cnn2d"], required=True, help="Model architecture")
    parser.add_argument("--model_path", help="Path to trained model (auto-inferred if not provided)")
    parser.add_argument("--fold", type=int, default=0, help="Fold number to evaluate")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Auto-infer model path if not provided
    if not args.model_path:
        args.model_path = f"checkpoints/model_{args.model}_fold{args.fold}.pt"
    
    print(f"Evaluating {args.model} model on fold {args.fold}")
    print(f"Using model: {args.model_path}")
    
    # Get global labels and validation split for the specified fold
    global_labels = get_global_labels(args.data_root, top_k=5)
    splits = get_cv_splits(args.data_root, k=5, seed=42)
    _, val_indices = splits[args.fold]
    
    # Create dataset and dataloader with consistent labels
    val_dataset = PTBXLDataset(args.data_root, val_indices, args.model, False, global_labels)
    val_loader = create_dataloader(val_dataset, 16, False)
    
    # Load model
    num_labels = val_dataset.num_labels
    model = load_model(args.model_path, args.model, num_labels, device)
    
    print(f"Loaded model from {args.model_path}")
    print(f"Dataset has {len(val_dataset)} samples with {num_labels} labels")
    
    # Evaluate
    y_pred, y_true = evaluate_model(model, val_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    y_pred_binary = (y_pred >= args.threshold).astype(int)
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(f"PR-AUC (macro): {metrics['pr_auc_macro']:.4f}")
    print(f"ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")
    print(f"F1 (micro): {metrics['f1_micro']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'metrics': metrics,
        'threshold': args.threshold,
        'fold': args.fold,
        'model_type': args.model,
        'num_labels': num_labels,
        'label_names': val_dataset.label_names
    }
    
    results_path = output_dir / f'eval_fold_{args.fold}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Optional: plot confusion matrix for first label
    plot_path = output_dir / f'confusion_matrix_fold_{args.fold}.png'
    plot_confusion_matrix(y_true, y_pred_binary, val_dataset.label_names, plot_path)
    
    print(f"\nResults saved to {results_path}")
    print(f"Confusion matrix saved to {plot_path}")


if __name__ == "__main__":
    main()
