import argparse

import matplotlib.pyplot as plt
import torch

from datasets.camelyon import TorchDataset
from models.lightning_model_module import MIL
from utils.utils import read_yaml
from xAI.attention_rollout import attention_rollout
from xAI.heatmap import heatmap
from xAI.integrated_gradients import integrated_gradients
from xAI.saliency_gradients import saliency_gradients


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", type=str, default="/home/pml16/camelyon16_mini_split.csv"
    )
    parser.add_argument(
        "--config",
        default="/home/pml16/MS2/CamelyonConfig/config.yaml",
        type=str,
    )
    parser.add_argument("--case_id", type=str, default="test_001")
    parser.add_argument(
        "--method",
        type=str,
        default="att_rollout",
        choices=["att_rollout", "integrated_grads", "saliency_grads"],
    )
    parser.add_argument("--save_dir", type=str, default="./")

    args = parser.parse_args()
    return args


def main(args):
    case_id = args.case_id
    method = args.method
    save_dir = args.save_dir

    test_set = TorchDataset(
        data_path="/mnt/features/20x/ctranspath_pt",
        split_path=args.split,
        split_type="test",
    )

    case_id = int(case_id.split("_")[1]) - 1

    features, label, case_id = test_set[case_id]
    features = torch.unsqueeze(features, dim=0)
    features.requires_grad_()

    cfg = read_yaml(args.config)
    cfg.General.run_name = "test"

    ckpt_path = (
        "./MS3/ckpts/" "best-epoch=epoch=13-val_loss=val_loss=0.25.ckpt"
    )

    model = MIL.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.eval()

    if method == "att_rollout":
        attributes = attention_rollout(model, features)
    elif method == "integrated_grads":
        attributes = integrated_gradients(model, features, label)
    elif method == "saliency_grads":
        attributes = saliency_gradients(model, features)

    colormap = plt.get_cmap('jet')
    colors = [colormap(val) for val in attributes]

    heatmap(
        colors,
        slide_id=case_id,
        patch_id_list=attributes,
        base_dir="/mnt/",
        save_dir=save_dir,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
