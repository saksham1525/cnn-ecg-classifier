"""Simple CNN models for ECG classification."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any


class SimpleCNN1D(nn.Module):
    """Simple 1D CNN for ECG classification."""
    
    def __init__(self, num_labels: int = 5, dropout: float = 0.2):
        super().__init__()
        self.num_labels = num_labels
        
        # Simple 1D CNN: 12-lead input -> features -> classifier
        self.conv_layers = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(64, num_labels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 12, time_steps)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)  # (batch, 128)
        return self.classifier(x)


class SimpleCNN2D(nn.Module):
    """Simple 2D CNN for ECG classification with 2D representation."""
    
    def __init__(self, num_labels: int = 5, dropout: float = 0.2):
        super().__init__()
        self.num_labels = num_labels
        
        # Simple 2D CNN for spectrogram-like representation
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(16, num_labels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, height, width) - 2D representation
        x = self.conv_layers(x)
        x = self.global_pool(x).flatten(1)  # (batch, 32)
        return self.classifier(x)


def render_ecg_to_2d(ecg_data: np.ndarray, height: int = 64, width: int = 256) -> np.ndarray:
    """Convert 12-lead ECG to simple 2D representation.
    
    Args:
        ecg_data: ECG data of shape (12, time_steps)
        height: Target height for 2D representation
        width: Target width for 2D representation
        
    Returns:
        2D representation of shape (height, width)
    """
    # Simple approach: stack leads vertically and resize
    n_leads, n_samples = ecg_data.shape
    
    # Interpolate to target width
    if n_samples != width:
        x_old = np.linspace(0, 1, n_samples)
        x_new = np.linspace(0, 1, width)
        ecg_resized = np.array([np.interp(x_new, x_old, lead) for lead in ecg_data])
    else:
        ecg_resized = ecg_data
    
    # Create 2D representation by tiling leads
    leads_per_row = height // n_leads
    if leads_per_row == 0:
        leads_per_row = 1
    
    # Simple stacking approach
    result = np.zeros((height, width))
    for i in range(min(n_leads, height)):
        row_start = i * (height // n_leads)
        row_end = min(row_start + (height // n_leads), height)
        result[row_start:row_end] = np.tile(ecg_resized[i], (row_end - row_start, 1))
    
    return result


def create_model(model_type: str, num_labels: int = 5) -> nn.Module:
    """Create model based on type."""
    if model_type == "cnn1d":
        return SimpleCNN1D(num_labels=num_labels)
    elif model_type == "cnn2d":
        return SimpleCNN2D(num_labels=num_labels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model(model_path: str, model_type: str, num_labels: int = 5, device: str = "cpu") -> nn.Module:
    """Load trained model from checkpoint."""
    model = create_model(model_type, num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model
