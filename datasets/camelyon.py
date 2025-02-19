from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TorchDataset(Dataset):
    def __init__(
        self, data_path: str, split_path: str, split_type: str, shuffle=True
    ):
        self.data_path = data_path
        self.shuffle = shuffle
        if split_type not in ['train', 'test', 'val']:
            raise ValueError(
                f"No such dataset split type as {split_type}, \
                    select either 'train', 'test', or 'val'"
            )

        # Load entries for a given split type
        df = pd.read_csv(split_path)
        self.data = df['case_id'].loc[df['subset'] == split_type].tolist()
        self.labels = df['label'].loc[df['subset'] == split_type].tolist()

        # Remove the "normal_144" entry from the dataset
        # features for this entry are missing
        value = "normal_144"
        idx = -1
        if value in self.data:
            idx = self.data.index(value)
            self.data.pop(idx)
            self.labels.pop(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        case_id = self.data[idx]
        features = torch.load(
            Path(str(self.data_path), f"{case_id}.pt"), weights_only=True
        )
        #  Squeeze from shape [N, 1, 768] -> [N, 768]
        features = torch.squeeze(features, axis=1)
        label = self.labels[idx]

        # Shuffle instances in the bag if stated so
        if self.shuffle:
            idxs = np.arange(features.shape[0])
            np.random.shuffle(idxs)
            features = features[idxs]

        return features, label, case_id
