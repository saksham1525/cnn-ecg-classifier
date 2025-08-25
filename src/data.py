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


def _count_diagnostic_codes(metadata: pd.DataFrame, scp_codes: pd.DataFrame) -> dict:
    """Count diagnostic codes frequency with confidence threshold."""
    diagnostic_counts = {}
    for scp_codes_str in metadata["scp_codes"]:
        codes = eval(scp_codes_str)
        for code in codes:
            if (code in scp_codes.index and 
                scp_codes.loc[code, "diagnostic"] == 1.0 and
                codes[code] >= 50):
                diagnostic_counts[code] = diagnostic_counts.get(code, 0) + 1
    return diagnostic_counts


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
        diagnostic_counts = _count_diagnostic_codes(self.metadata, self.scp_codes)
        self.label_names = sorted(diagnostic_counts, key=diagnostic_counts.get, reverse=True)[:5]
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
        
        # Load ECG waveform and convert to tensor
        record = wfdb.rdrecord(str(self.data_root / row["filename_lr"]))
        waveform = torch.tensor(record.p_signal.T, dtype=torch.float32)
        
        # Simple data augmentation during training
        if self.is_training and random.random() < 0.3:
            # Add small amount of noise
            noise = torch.normal(0, 0.01, waveform.shape)
            waveform = waveform + noise
            
        # Prepare input based on model type
        input_tensor = (waveform if self.model_type == "cnn1d" 
                       else self._render_to_2d(waveform).unsqueeze(0))
            
        return input_tensor, self.labels[idx]
        
    def _render_to_2d(self, waveform: torch.Tensor, height: int = 64, width: int = 256) -> torch.Tensor:
        """Simple conversion of 12-lead ECG to 2D representation."""
        num_leads, seq_len = waveform.shape
        output = torch.zeros(height, width)
        
        for i in range(num_leads):
            # Resample each lead to target width
            if seq_len != width:
                indices = torch.linspace(0, seq_len - 1, width)
                indices_floor = indices.long()
                weights = indices - indices_floor.float()
                resampled = ((1 - weights) * waveform[i, indices_floor] + 
                           weights * waveform[i, torch.clamp(indices_floor + 1, 0, seq_len - 1)])
            else:
                resampled = waveform[i]
                
            # Fill rows for this lead
            row_start, row_end = i * (height // num_leads), min((i + 1) * (height // num_leads), height)
            output[row_start:row_end] = resampled
                
        return output


def get_global_labels(data_root: str, top_k: int = 5) -> List[str]:
    """Get globally consistent label names from full dataset."""
    data_root = Path(data_root)
    metadata = pd.read_csv(data_root / "ptbxl_database.csv", index_col="ecg_id")
    scp_codes = pd.read_csv(data_root / "scp_statements.csv", index_col=0)
    
    diagnostic_counts = _count_diagnostic_codes(metadata, scp_codes)
    return sorted(diagnostic_counts, key=diagnostic_counts.get, reverse=True)[:top_k]


def get_cv_splits(data_root: str, k: int = 5, seed: int = 42) -> List[Tuple[List[int], List[int]]]:
    """Get k-fold cross-validation splits by patient ID."""
    metadata = pd.read_csv(Path(data_root) / "ptbxl_database.csv", index_col="ecg_id")
    
    gkf = GroupKFold(n_splits=k)
    gkf.random_state = seed
    
    return [(train_idx.tolist(), val_idx.tolist()) 
            for train_idx, val_idx in gkf.split(range(len(metadata)), groups=metadata["patient_id"])
    ]


def create_dataloader(dataset: Dataset, batch_size: int = 32, is_training: bool = True) -> DataLoader:
    """Create simple dataloader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=2,
        pin_memory=True
    )
