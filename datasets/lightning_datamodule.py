from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from datasets.camelyon import TorchDataset


class CamelyonDataset(LightningDataModule):
    def __init__(self, data_config):  # or **kwargs
        super().__init__()
        if Path(data_config.split_file).is_file():
            self.split_file = Path(data_config.split_file)
        else:
            raise ValueError(
                f"File {data_config.split_file} is not a valid split file"
            )

        self.data_path = data_config.data_path
        self.train_batch_size = data_config.Train.batch_size
        self.val_batch_size = data_config.Val.batch_size
        self.test_batch_size = data_config.Test.batch_size
        self.shuffle_patches = data_config.shuffle

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = TorchDataset(
                self.data_path,
                self.split_file,
                split_type='train',
                shuffle=self.shuffle_patches,
            )
            self.val_set = TorchDataset(
                self.data_path,
                self.split_file,
                split_type='val',
                shuffle=self.shuffle_patches,
            )
        if stage == "test" or stage is None:
            self.test_set = TorchDataset(
                self.data_path,
                self.split_file,
                split_type='test',
                shuffle=self.shuffle_patches,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            num_workers=2,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.val_batch_size,
            num_workers=2,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.test_batch_size,
            num_workers=2,
            shuffle=False,
        )
