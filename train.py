import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.camelyon import load_data
from models.TransMILBaseline import TransMILBaseline
from utils.utils import read_yaml
from visualization import plot_stats


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
    args = parser.parse_args()
    return args


def common_compute(model, batch, criterion):
    features, labels = batch
    features, labels = features.to(DEVICE), labels.to(DEVICE)
    logits = model.forward(features)
    loss = criterion(logits, labels)
    return logits, loss, labels


def training_step(model, batch, optimizer, criterion):
    _, loss, _ = common_compute(model, batch, criterion)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


def test_step(model, batch, criterion):
    logits, loss, y = common_compute(model, batch, criterion)
    _, preds = torch.max(logits.detach(), axis=1)  # [B, 1]
    correct_preds = (preds == y).sum().item()
    return loss, y.shape[0], correct_preds


def val_step(model, batch, criterion):
    logits, loss, y = common_compute(model, batch, criterion)
    _, preds = torch.max(logits.detach(), axis=1)  # [B, 1]
    correct_preds = (preds == y).sum().item()
    return loss, y.shape[0], correct_preds


DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


def main(cfg: dict):
    num_epochs = cfg["General"]["epochs"]
    log_output = cfg["General"]["log_output"]

    train_dataset, test_dataset, val_dataset = load_data(
        split_file=cfg["Split"], base_dir="/mnt/"
    )

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    model = TransMILBaseline(
        new_num_features=384, n_heads=4, num_classes=2, device=DEVICE
    )
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    batch_size = 1
    epochs = num_epochs

    train_epoch_loss = []
    val_epoch_loss = []
    val_epoch_acc = []

    for epoch in range(epochs):
        loss_total = []
        correct = 0
        total = 0
        print(f"epoch {epoch+1} out of {epochs}")
        for batch in train_dataloader:
            model.train()
            loss = training_step(model, batch, optimizer, criterion)
            loss_total.append(loss.detach().item())
        avg_epoch_loss = sum(loss_total) / len(train_dataloader)
        train_epoch_loss.append(avg_epoch_loss)

        print(
            f"avg loss after epoch {epoch+1} / {epochs} is: {avg_epoch_loss}"
        )
        loss_total.clear()

        with torch.no_grad():
            for batch in val_dataloader:
                model.eval()
                loss, batch_size, correct_preds = val_step(
                    model, batch, criterion
                )
                loss_total.append(loss.detach().item())
                total += batch_size
                correct += correct_preds

        avg_epoch_loss = sum(loss_total) / len(val_dataloader)
        acc = correct / total * 100

        val_epoch_loss.append(avg_epoch_loss)
        val_epoch_acc.append(acc)
        print(
            f"Validation accuracy after epoch {epoch+1} / {epochs} is: {acc}"
        )

    plot_stats(
        epochs,
        train_epoch_loss,
        "Train loss",
        plot_type="loss",
        save=True,
        save_path=log_output,
    )
    plot_stats(
        epochs,
        val_epoch_loss,
        "Val loss",
        plot_type="loss",
        save=True,
        save_path=log_output,
    )
    plot_stats(
        epochs,
        val_epoch_acc,
        "Val accuracy",
        plot_type="acc",
        save=True,
        save_path=log_output,
    )

    loss_total = []
    with torch.no_grad():
        for batch in test_dataloader:
            correct, total = 0, 0
            model.eval()
            loss, batch_size, correct_preds = test_step(
                model, batch, criterion
            )
            loss_total.append(loss.detach().item())
            total += batch_size
            correct += correct_preds
        avg_test_loss = sum(loss_total) / len(test_dataloader)
        acc = correct / total * 100
        print(f"avg loss after test is: {avg_test_loss}")
        print(f"Test accuracy after {epochs} epochs training is: {acc}")


if __name__ == "__main__":
    args = make_parse()
    cfg = read_yaml(args.config)
    cfg["Split"] = args.split
    main(cfg)
