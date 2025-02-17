import argparse
import random
import string
from pathlib import Path

import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from callbacks.callbacks import get_checkpoint_callback, get_early_stopping
from datasets.lightning_datamodule import CamelyonDataset
from models.lightning_model_module import MIL
from utils.utils import check_path, read_yaml


def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str)
    parser.add_argument(
        "--config",
        default="/home/pml16/MS3/CamelyonConfig/config.yaml",
        type=str,
    )

    parser.add_argument(
        "--run_name",
        default=''.join(
            random.choices(string.ascii_letters + string.digits, k=10)
        ),
        type=str,
    )

    args = parser.parse_args()
    return args


def main(cfg: dict):
    if cfg.General.random_seed:
        seed = random.randint(1, 1000)
        seed_everything(seed)
    else:
        seed = cfg.General.seed
        seed_everything(seed)

    # add a seed number to run_name,
    # to distinguish between runs in the same group
    run_name = f"{cfg.General.run_name}_{seed}"

    data_cfg = cfg.Data
    num_epochs = cfg.General.epochs
    num_features = cfg.Model.num_features

    ckpt_save_path = Path(
        cfg.Data.ckpt_save_path,
        f"{run_name}_lr_{cfg.Optimizer.lr}_num_feat_{num_features}",
    )
    batch_size = data_cfg.Train.batch_size

    dataset = CamelyonDataset(data_cfg)
    dataset.setup()
    trans_model = MIL(cfg)

    model_config = {
        "num_features": num_features,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr": cfg.Optimizer.lr,
        "dropout": cfg.General.dropout,
        "ppeg": cfg.Model.use_ppeg,
        "first_fc_layer": cfg.Model.use_fclayer,
    }
    print(f"Model config: \n {model_config}")

    wandb.login(key=cfg.Wandb.key, relogin=True)
    wandb_logger = WandbLogger(
        name=run_name,
        project=cfg.Wandb.project_name,
        job_type='train',
        group=cfg.General.test_group,
        config=model_config,
    )

    early_stopping_callback = get_early_stopping(cfg.General.patience)
    checkpoint_callback = get_checkpoint_callback(save_path=ckpt_save_path)

    trainer = Trainer(
        check_val_every_n_epoch=2,
        num_sanity_val_steps=1,
        log_every_n_steps=50,
        accelerator="auto",
        logger=wandb_logger,
        max_epochs=num_epochs,
        callbacks=[early_stopping_callback, checkpoint_callback],
    )
    # regular train
    trainer.fit(model=trans_model, datamodule=dataset)

    wandb.finish()


if __name__ == "__main__":
    args = make_parse()
    cfg = read_yaml(args.config)
    if args.split and check_path(args.split):
        cfg.Data.split_file = args.split
    if args.run_name:
        cfg.General.run_name = args.run_name

    main(cfg)
