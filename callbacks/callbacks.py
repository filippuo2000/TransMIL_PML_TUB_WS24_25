from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def get_checkpoint_callback(save_path: str):
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename="best-epoch={epoch:02d}-val_loss={val_loss:.2f}",
        save_last=True,
        save_top_k=1,
        mode='min',
        monitor="val_loss",
        verbose=True,
    )
    return checkpoint_callback


def get_early_stopping(patience_value: int = 5):
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=patience_value,
        mode='min',
        verbose=True,
    )
    return early_stop_callback
