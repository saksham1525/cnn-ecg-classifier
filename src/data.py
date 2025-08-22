"""Simple data loading for PTB-XL ECG classification."""

from typing import Tuple, List
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import wfdb
from sklearn.model_selection import GroupKFold


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class PTBXLDataset(Dataset):
    """Simple PTB-XL ECG dataset with multi-label classification."""
    
    def __init__(self, data_root: str, indices: List[int], model_type: str = "cnn1d", 
                 is_training: bool = True, label_names: List[str] = None):
        self.data_root = Path(data_root)
        self.model_type = model_type
        self.is_training = is_training
        
        # Load metadata
        self.metadata = pd.read_csv(self.data_root / "ptbxl_database.csv", index_col="ecg_id")
        self.metadata = self.metadata.iloc[indices]
        
        # Load diagnostic mappings
        self.scp_codes = pd.read_csv(self.data_root / "scp_statements.csv", index_col=0)
        
        # Use pre-computed label names or compute from this subset
        if label_names is not None:
            self.label_names = label_names
            self.num_labels = len(self.label_names)
            self._create_binary_labels()
        else:
            # Use top 5 most common diagnostic classes for simplicity
            self._prepare_labels()
        
    def _prepare_labels(self) -> None:
        """Prepare top 5 diagnostic labels."""
        # Count diagnostic codes frequency
        diagnostic_counts = {}
        for scp_codes_str in self.metadata["scp_codes"]:
            codes = eval(scp_codes_str)
            for code in codes:
                if (code in self.scp_codes.index and 
                    self.scp_codes.loc[code, "diagnostic"] == 1.0 and
                    codes[code] >= 50):  # Only high-confidence labels
                    diagnostic_counts[code] = diagnostic_counts.get(code, 0) + 1
        
        # Take top 5 most frequent diagnostic codes
        self.label_names = sorted(diagnostic_counts.keys(), 
                                key=lambda x: diagnostic_counts[x], reverse=True)[:5]
        self.num_labels = len(self.label_names)
        
        self._create_binary_labels()
        
    def _create_binary_labels(self) -> None:
        """Create binary labels from the label names."""
        labels_list = []
        for scp_codes_str in self.metadata["scp_codes"]:
            codes = eval(scp_codes_str)
            sample_labels = [1 if code in codes and codes[code] >= 50 else 0 
                           for code in self.label_names]
            labels_list.append(sample_labels)
        
        self.labels = torch.tensor(labels_list, dtype=torch.float32)
        
    def __len__(self) -> int:
        return len(self.metadata)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[idx]
        
        # Load ECG waveform (lr for 100Hz version)
        record_path = self.data_root / row["filename_lr"]
        record = wfdb.rdrecord(str(record_path))
        signal = record.p_signal.T  # Shape: [12, time_steps]
        
        # Convert to tensor
        waveform = torch.tensor(signal, dtype=torch.float32)
        
        # Simple data augmentation during training
        if self.is_training and random.random() < 0.3:
            # Add small amount of noise
            noise = torch.normal(0, 0.01, waveform.shape)
            waveform = waveform + noise
            
        # Prepare input based on model type
        if self.model_type == "cnn1d":
            # For 1D CNN: return as [12, time_steps]
            input_tensor = waveform
        else:  # cnn2d
            # For 2D CNN: convert to simple 2D representation
            input_tensor = self._render_to_2d(waveform)
            input_tensor = input_tensor.unsqueeze(0)  # Add channel dimension [1, H, W]
            
        targets = self.labels[idx]
        return input_tensor, targets
        
    def _render_to_2d(self, waveform: torch.Tensor, height: int = 64, width: int = 256) -> torch.Tensor:
        """Simple conversion of 12-lead ECG to 2D representation."""
        num_leads, seq_len = waveform.shape
        
        # Simple approach: stack leads vertically
        leads_per_row = height // num_leads
        output = torch.zeros(height, width)
        
        for i in range(num_leads):
            # Resample each lead to target width
            signal = waveform[i]
            if seq_len != width:
                # Simple linear interpolation
                indices = torch.linspace(0, seq_len - 1, width)
                indices_floor = indices.long()
                indices_ceil = torch.clamp(indices_floor + 1, 0, seq_len - 1)
                weights = indices - indices_floor.float()
                resampled = (1 - weights) * signal[indices_floor] + weights * signal[indices_ceil]
            else:
                resampled = signal
                
            # Fill rows for this lead
            row_start = i * (height // num_leads)
            row_end = min(row_start + (height // num_leads), height)
            for row in range(row_start, row_end):
                output[row] = resampled
                
        return output


def get_global_labels(data_root: str, top_k: int = 5) -> List[str]:
    """Get globally consistent label names from full dataset."""
    metadata = pd.read_csv(Path(data_root) / "ptbxl_database.csv", index_col="ecg_id")
    scp_codes = pd.read_csv(Path(data_root) / "scp_statements.csv", index_col=0)
    
    # Count diagnostic codes frequency across full dataset
    diagnostic_counts = {}
    for scp_codes_str in metadata["scp_codes"]:
        codes = eval(scp_codes_str)
        for code in codes:
            if (code in scp_codes.index and 
                scp_codes.loc[code, "diagnostic"] == 1.0 and
                codes[code] >= 50):  # Only high-confidence labels
                diagnostic_counts[code] = diagnostic_counts.get(code, 0) + 1
    
    # Return top k most frequent diagnostic codes
    return sorted(diagnostic_counts.keys(), 
                  key=lambda x: diagnostic_counts[x], reverse=True)[:top_k]


def get_cv_splits(data_root: str, k: int = 5, seed: int = 42) -> List[Tuple[List[int], List[int]]]:
    """Get k-fold cross-validation splits by patient ID."""
    metadata = pd.read_csv(Path(data_root) / "ptbxl_database.csv", index_col="ecg_id")
    
    # Group by patient to avoid data leakage
    patient_ids = metadata["patient_id"].values
    indices = list(range(len(metadata)))
    
    gkf = GroupKFold(n_splits=k)
    gkf.random_state = seed
    
    splits = []
    for train_idx, val_idx in gkf.split(indices, groups=patient_ids):
        splits.append((train_idx.tolist(), val_idx.tolist()))
        
    return splits


def create_dataloader(dataset: Dataset, batch_size: int = 32, is_training: bool = True) -> DataLoader:
    """Create simple dataloader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=2,
        pin_memory=True
    )
