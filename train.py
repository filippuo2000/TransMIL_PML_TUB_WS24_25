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
from utils.utils import read_yaml


def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", type=str, default="/home/pml16/camelyon16_mini_split.csv"
    )
    parser.add_argument(
        "--config",
        default="/home/pml16/MS2/CamelyonConfig/config.yaml",
        type=str,
    )

    parser.add_argument(
        "--run_name",
        default=''.join(
            random.choices(string.ascii_letters + string.digits, k=10)
        ),
        type=str,
    )

    parser.add_argument("--run_repeats", default=1, type=int)

    args = parser.parse_args()
    return args


def main(cfg: dict):
    seed = random.randint(1, 1000)
    seed_everything(seed)
    # seed_everything(cfg.General.seed)
    data_cfg = cfg.Data

    num_epochs = cfg.General.epochs
    num_features = cfg.Model.num_features
    # add a seed number to run_name,
    # to distinguish between runs in the same group
    run_name = f"{cfg.General.run_name}_{seed}"

    ckpt_save_path = Path(
        cfg.Data.ckpt_save_path,
        "seed_version",
        f"{run_name}_lr_{cfg.Optimizer.lr}_num_feat_{num_features}",
    )
    batch_size = data_cfg.Train.batch_size

    dataset = CamelyonDataset(data_cfg)
    dataset.setup()
    trans_model = MIL(cfg)

    wandb.login(key="", relogin=True)
    wandb_logger = WandbLogger(
        name=run_name,
        project='TransMIL_TUB_REPO_TEST',
        job_type='train',
        group=cfg.General.test_group,
        config={
            "num_features": num_features,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lr": cfg.Optimizer.lr,
            "dropout": 0.1,
            "ppeg": cfg.Model.use_ppeg,
            "first_fc_layer": cfg.Model.use_fclayer,
        },
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
    # regular test and train
    trainer.fit(model=trans_model, datamodule=dataset)
    # trainer.test(ckpt_path="best", datamodule=dataset)

    # test only

    # best_ckpt = ("./MS3/ckpts/"
    # "best-epoch=epoch=13-val_loss=val_loss=0.25.ckpt"
    # )
    # model = MIL.load_from_checkpoint(best_ckpt, cfg=cfg)
    # test_loader = dataset.test_dataloader()
    # trainer.test(model, test_loader)

    wandb.finish()


if __name__ == "__main__":
    args = make_parse()
    cfg = read_yaml(args.config)
    print(f"lr is: {cfg.Optimizer.lr}")
    if args.split:
        cfg.Data.split_file = args.split
    if args.run_name:
        cfg.General.run_name = args.run_name

    main(cfg)
