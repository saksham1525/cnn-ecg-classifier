"""Simple training script with 5-fold CV and weighted BCE loss."""

import logging
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data import PTBXLDataset, get_cv_splits, create_dataloader, get_global_labels
from models import create_model
from metrics import compute_metrics


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset: PTBXLDataset) -> torch.Tensor:
    """Compute class weights for balanced BCE loss."""
    all_labels = []
    for i in range(len(dataset)):
        _, labels = dataset[i]
        all_labels.append(labels.numpy())
    
    all_labels = np.stack(all_labels)  # [N, num_classes]
    pos_counts = all_labels.sum(axis=0)  # Positive samples per class
    neg_counts = len(all_labels) - pos_counts  # Negative samples per class
    
    # Weight = neg_count / pos_count for balancing
    weights = neg_counts / (pos_counts + 1e-8)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_fold(model: nn.Module, train_loader, val_loader, epochs: int, 
                   lr: float, device: torch.device, model_type: str, fold: int) -> dict:
    """Train one fold and return best validation metrics."""
    
    # Weighted BCE loss
    class_weights = compute_class_weights(train_loader.dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_pr_auc = 0.0
    best_metrics = {}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                
                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Compute metrics
        metrics = compute_metrics(val_targets, val_preds)
        
        # Save best model by PR-AUC
        if metrics['pr_auc_macro'] > best_pr_auc:
            best_pr_auc = metrics['pr_auc_macro']
            best_metrics = metrics.copy()
            # Save to checkpoints directory
            checkpoint_path = Path("checkpoints") / f"model_{model_type}_fold{fold}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            
        # Log every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"Epoch {epoch+1:2d}/{epochs}: PR-AUC {metrics['pr_auc_macro']:.4f}")
    
    return best_metrics


def main():
    """Main training function with 5-fold cross-validation."""
    parser = argparse.ArgumentParser(description="Train ECG classification models")
    parser.add_argument("--data_root", required=True, help="Path to PTB-XL dataset")
    parser.add_argument("--model", choices=["cnn1d", "cnn2d"], default="cnn1d", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=5, help="Number of CV folds")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create checkpoints directory
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    logging.info(f"Starting {args.fold}-fold CV training with {args.model}")
    logging.info(f"Device: {device}")
    
    # Get global labels for consistency across folds
    global_labels = get_global_labels(args.data_root, top_k=5)
    logging.info(f"Using {len(global_labels)} global labels: {global_labels}")
    
    # Get CV splits
    splits = get_cv_splits(args.data_root, k=args.fold, seed=args.seed)
    fold_results = []
    
    # Cross-validation loop
    for fold, (train_indices, val_indices) in enumerate(splits):
        logging.info(f"\n=== Fold {fold + 1}/{args.fold} ===")
        
        # Create datasets and loaders with consistent labels
        train_dataset = PTBXLDataset(args.data_root, train_indices, args.model, True, global_labels)
        val_dataset = PTBXLDataset(args.data_root, val_indices, args.model, False, global_labels)
        
        train_loader = create_dataloader(train_dataset, args.batch_size, True)
        val_loader = create_dataloader(val_dataset, args.batch_size, False)
        
        # Create model
        num_labels = train_dataset.num_labels
        model = create_model(args.model, num_labels).to(device)
        
        # Train fold
        fold_metrics = train_one_fold(
            model, train_loader, val_loader, args.epochs, 
            args.lr, device, args.model, fold
        )
        
        fold_results.append(fold_metrics)
        logging.info(f"Fold {fold + 1} Best PR-AUC: {fold_metrics['pr_auc_macro']:.4f}")
    
    # Compute average metrics
    avg_metrics = {}
    for metric in fold_results[0].keys():
        values = [fold[metric] for fold in fold_results]
        avg_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    # Save results
    results = {
        'args': vars(args),
        'fold_results': fold_results,
        'average_metrics': avg_metrics
    }
    
    with open(output_dir / 'cv_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    logging.info(f"\n=== Final Results ===")
    pr_auc = avg_metrics['pr_auc_macro']
    f1 = avg_metrics['f1_micro']
    logging.info(f"PR-AUC: {pr_auc['mean']:.4f} ± {pr_auc['std']:.4f}")
    logging.info(f"F1-micro: {f1['mean']:.4f} ± {f1['std']:.4f}")


if __name__ == "__main__":
    main()
