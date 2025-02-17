import argparse
import random
import string

import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

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

    parser.add_argument("--ckpt", type=str)

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

    data_cfg = cfg.Data

    dataset = CamelyonDataset(data_cfg)
    dataset.setup()

    # add a seed number to run_name,
    # to distinguish between runs in the same group
    run_name = f"{cfg.General.run_name}_{seed}"

    batch_size = data_cfg.Train.batch_size
    num_features = cfg.Model.num_features
    model_config = {
        "num_features": num_features,
        "batch_size": batch_size,
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
        job_type='test',
        group=cfg.General.test_group,
        config=model_config,
    )

    trainer = Trainer(
        num_sanity_val_steps=1,
        accelerator="auto",
        logger=wandb_logger,
        max_epochs=1,
    )

    # test only
    best_ckpt = cfg.General.ckpt_path
    model = MIL.load_from_checkpoint(best_ckpt, cfg=cfg)
    test_loader = dataset.test_dataloader()

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')
    trainer.test(model, test_loader)

    wandb.finish()


if __name__ == "__main__":
    args = make_parse()
    cfg = read_yaml(args.config)
    if args.split and check_path(args.split):
        cfg.Data.split_file = args.split
    if args.run_name:
        cfg.General.run_name = args.run_name
    if args.ckpt and check_path(args.ckpt):
        cfg.General.ckpt_path = args.ckpt

    main(cfg)
