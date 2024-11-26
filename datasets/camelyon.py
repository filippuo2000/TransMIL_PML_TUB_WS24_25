import random
import torch

import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset


class TorchDataset(Dataset):
    def __init__(self, data_path: str, split_path: str, split_type: str, shuffle=True):
        self.data_path = data_path
        self.shuffle = shuffle
        if split_type not in ['train', 'test', 'val']:
            raise ValueError(f"No such dataset split type as {split_type}, select either 'train', 'test', or 'val'")
        
        # Load entries for a given split type
        df = pd.read_csv(split_path)
        self.data = df['case_id'].loc[df['subset'] == split_type].tolist()
        self.labels = df['label'].loc[df['subset'] == split_type].tolist()

        # Remove the "normal_144" entry from the dataset - missing features for this entry
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
        features = torch.load(Path(str(self.data_path), f"{case_id}.pt"))
        #  Squeeze from shape [N, 1, 768] -> [N, 768]
        features = torch.squeeze(features, axis=1)
        label = self.labels[idx]

        # Shuffle instances in the bag if stated so
        if self.shuffle == True:
            idxs = np.arange(features.shape[0])
            np.random.shuffle(idxs)
            features = features[idxs]
            
        return features, label

def load_data(split_file=None, num_splits=3, shuffle=True, n_train=None, n_test=None, base_dir='/home/space/datasets/camelyon16'):

    data_path = Path(base_dir, 'features', '20x', 'ctranspath_pt')

    if split_file is not None:
        if Path(split_file).is_file():
            split_path = Path(split_file)
        else:
            raise ValueError(f"File {split_file} is not a valid split file")
    else:
        split_file = 'camelyon16_tumor_85_15_orig_0.csv'
        split_path = Path(base_dir, 'splits', split_file)
    
    if not split_path.is_file():
        raise ValueError(f"Split file {split_file} does not exist")
    
    train_set = TorchDataset(data_path, split_path, split_type='train')
    test_set = TorchDataset(data_path, split_path, split_type='test')

    if num_splits==3:
        val_set = TorchDataset(data_path, split_path, split_type='val')
        return train_set, test_set, val_set
    return train, test