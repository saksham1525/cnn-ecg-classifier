"""Simple model interpretation using saliency maps."""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from data import PTBXLDataset, get_cv_splits, create_dataloader, get_global_labels
from models import load_model


def compute_saliency_map(model: torch.nn.Module, input_tensor: torch.Tensor, 
                        class_idx: int) -> np.ndarray:
    """Compute saliency map by taking gradient of input with respect to target class."""
    model.eval()
    input_tensor.requires_grad_()
    
    # Forward pass
    output = model(input_tensor)
    
    # Backward pass on target class
    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()
    
    # Saliency is absolute value of gradients
    if len(input_tensor.grad.shape) == 3:  # 1D CNN: [batch, leads, time]
        saliency = torch.abs(input_tensor.grad[0])  # [leads, time]
    else:  # 2D CNN: [batch, channels, height, width]
        saliency = torch.abs(input_tensor.grad[0, 0])  # [height, width]
    
    return saliency.detach().cpu().numpy()


def plot_saliency_1d(ecg_waveform: np.ndarray, saliency: np.ndarray, 
                     class_name: str, save_path: Path) -> None:
    """Plot saliency map for 1D ECG (12-lead format)."""
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for i in range(12):
        # Plot ECG signal
        time_axis = np.arange(len(ecg_waveform[i]))
        axes[i].plot(time_axis, ecg_waveform[i], color='blue', alpha=0.8)
        
        # Create background highlighting based on saliency
        saliency_norm = saliency[i] / (np.max(saliency[i]) + 1e-8)  # Normalize to [0, 1]
        
        # Color background based on importance
        for j in range(len(time_axis) - 1):
            alpha = saliency_norm[j] * 0.5  # Scale opacity
            axes[i].axvspan(time_axis[j], time_axis[j+1], alpha=alpha, color='red')
        
        axes[i].set_title(f'Lead {lead_names[i]}')
        axes[i].set_ylabel('Amplitude')
        if i >= 8:  # Bottom row
            axes[i].set_xlabel('Time (samples)')
    
    plt.suptitle(f'Saliency Map - {class_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_saliency_2d(saliency: np.ndarray, class_name: str, save_path: Path) -> None:
    """Plot saliency map for 2D representation."""
    plt.figure(figsize=(12, 6))
    plt.imshow(saliency, cmap='hot', aspect='auto')
    plt.title(f'Saliency Map (2D) - {class_name}')
    plt.xlabel('Time')
    plt.ylabel('Spatial Dimension')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Generate saliency map interpretation for a sample."""
    parser = argparse.ArgumentParser(description="Generate model interpretation")
    parser.add_argument("--data_root", required=True, help="Path to PTB-XL dataset")
    parser.add_argument("--model", choices=["cnn1d", "cnn2d"], required=True, help="Model architecture")
    parser.add_argument("--model_path", help="Path to trained model (auto-inferred if not provided)")
    parser.add_argument("--fold", type=int, default=0, help="Fold number")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to interpret")
    parser.add_argument("--output_dir", default="outputs")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Auto-infer model path if not provided
    if not args.model_path:
        args.model_path = f"checkpoints/model_{args.model}_fold{args.fold}.pt"
    
    print(f"Generating interpretation for {args.model} model, sample {args.sample_idx}")
    print(f"Using model: {args.model_path}")
    
    # Load dataset with consistent labels
    global_labels = get_global_labels(args.data_root, top_k=5)
    splits = get_cv_splits(args.data_root, k=5, seed=42)
    _, val_indices = splits[args.fold]
    val_dataset = PTBXLDataset(args.data_root, val_indices, args.model, False, global_labels)
    
    # Get specific sample
    input_tensor, targets = val_dataset[args.sample_idx]
    input_batch = input_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    # Load model
    num_labels = val_dataset.num_labels
    model = load_model(args.model_path, args.model, num_labels, device)
    
    # Get prediction
    with torch.no_grad():
        prediction = torch.sigmoid(model(input_batch))
        predicted_probs = prediction.cpu().numpy()[0]
    
    # Find most confident prediction
    most_confident_class = np.argmax(predicted_probs)
    confidence = predicted_probs[most_confident_class]
    
    print(f"Most confident prediction: Class {most_confident_class} (confidence: {confidence:.3f})")
    print(f"Ground truth: {targets.numpy()}")
    
    # Generate saliency map
    saliency = compute_saliency_map(model, input_batch, most_confident_class)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Plot and save
    class_name = f"Class_{most_confident_class}"
    if hasattr(val_dataset, 'label_names') and most_confident_class < len(val_dataset.label_names):
        class_name = val_dataset.label_names[most_confident_class]
    
    save_path = output_dir / f'saliency_{args.model}_fold_{args.fold}_sample_{args.sample_idx}.png'
    
    if args.model == "cnn1d":
        # For 1D CNN, convert back to original ECG format for visualization
        if len(input_tensor.shape) == 2:  # [leads, time]
            ecg_data = input_tensor.numpy()
        else:  # This shouldn't happen but just in case
            ecg_data = input_tensor.squeeze().numpy()
        plot_saliency_1d(ecg_data, saliency, class_name, save_path)
    else:
        # For 2D CNN, plot the 2D saliency map
        plot_saliency_2d(saliency, class_name, save_path)
    
    print(f"Saliency map saved to {save_path}")
    print("Interpretation completed!")


if __name__ == "__main__":
    main()
