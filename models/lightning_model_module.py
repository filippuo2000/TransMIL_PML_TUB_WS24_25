import json
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from pytorch_lightning import LightningModule

# from pytorch_optimizer import Lookahead, RAdam
from torchmetrics import (
    AUROC,
    Accuracy,
    F1Score,
    MetricCollection,
    Precision,
    Recall,
    Specificity,
)

from models.trans_mil_baseline import TransMILSquaring


class MIL(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = TransMILSquaring(
            new_num_features=cfg.Model.num_features,
            num_classes=cfg.General.num_classes,
        )
        self.lr = cfg.Optimizer.lr
        self.weight_decay = cfg.Optimizer.decay
        self.num_classes = cfg.General.num_classes
        self.run_name = cfg.General.run_name
        self.current_epoch_training_loss = torch.tensor(0.0)
        self.total_samples = torch.tensor(0.0)
        self.correct_samples = torch.tensor(0.0)
        self.test_results = {
            "labels": [],
            "predictions": [],
            "predictions_prob": [],
            "case_id": [],
        }

        self.acc_dict = {
            i: {"num_correct": 0, "num_total": 0}
            for i in range(self.num_classes)
        }

        self.criterion = nn.CrossEntropyLoss()
        self.AUROC = AUROC(task="binary", num_classes=self.num_classes)
        metrics = MetricCollection(
            {
                "accuracy": Accuracy(
                    task="binary", num_classes=2, average="micro"
                ),
                "precision": Precision(
                    task="binary", num_classes=2, average="macro"
                ),
                "recall": Recall(
                    task="binary", num_classes=2, average="macro"
                ),
                "f1": F1Score(task="binary", num_classes=2, average="macro"),
                "specificity": Specificity(
                    task="binary", num_classes=2, average="macro"
                ),
            }
        )

        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x, y):  # model specific, x - network outputs
        return self.criterion(x, y)

    def common_step(self, batch, batch_idx):
        features, labels, case_id = batch
        results = self.forward(features)
        logits, y_prob = results['logits'], results['y_prob']
        loss = self.compute_loss(logits, labels)
        return logits, loss, labels, y_prob, case_id

    def common_test_valid_step(self, batch, batch_idx):
        logits, loss, labels, y_prob, case_id = self.common_step(
            batch, batch_idx
        )
        preds = torch.argmax(logits, dim=1)  # [B]

        for pred, label in zip(preds, labels):
            if pred == label:
                self.acc_dict[label.item()]["num_correct"] += 1
            self.acc_dict[label.item()]["num_total"] += 1

        num_correct_preds = (preds == labels).sum().item()
        self.total_samples += labels.shape[0]
        self.correct_samples += num_correct_preds
        return loss, y_prob, labels, preds, case_id

    def validation_step(self, batch, batch_idx):
        loss, y_prob, labels, preds, _ = self.common_test_valid_step(
            batch, batch_idx
        )
        print(
            f"preds: {preds.item()}, labels: {labels.item()}, \
            y_prob: {y_prob.tolist()}, max_prob: {torch.max(y_prob, dim=1)}\n"
        )

        self.AUROC.update(y_prob[:, 1], labels)
        self.valid_metrics.update(preds, labels)
        self.log(
            'val_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        print(
            f"num of correct preds: {self.correct_samples}, \
            num samples: {self.total_samples}"
        )
        print(
            f"positive preds: {self.acc_dict[1]['num_correct']}, \
            total positive samples: {self.acc_dict[1]['num_total']}"
        )
        print(
            f"negative preds: {self.acc_dict[0]['num_correct']}, \
            total negative samples: {self.acc_dict[0]['num_total']}"
        )
        acc = self.calc_acc()
        class_acc = self.calc_class_acc("val")

        self.log('my_acc', acc, prog_bar=True, logger=True)
        self.log('val_AUC', self.AUROC.compute(), prog_bar=True, logger=True)
        self.log_dict(class_acc, prog_bar=True, logger=True)
        self.log_dict(self.valid_metrics.compute(), prog_bar=True, logger=True)

        self.valid_metrics.reset()
        self.AUROC.reset()
        self.reset_acc_dict()

    def test_step(self, batch, batch_idx):
        loss, y_prob, labels, preds, case_id = self.common_test_valid_step(
            batch, batch_idx
        )
        self.test_results["labels"].append(labels.item())
        self.test_results["predictions"].append(preds.item())
        max_prob, _ = torch.max(y_prob, dim=1)
        self.test_results["predictions_prob"].append(max_prob.item())
        self.test_results["case_id"].append(case_id)

        self.AUROC.update(y_prob[:, 1], labels)
        self.test_metrics.update(preds, labels)
        self.log(
            'test_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {'test_loss': loss}

    def on_test_epoch_end(self):
        print(
            f"num of correct preds: {self.correct_samples}, \
            num samples: {self.total_samples}"
        )
        print(
            f"positive preds: {self.acc_dict[1]['num_correct']}, \
            total positive samples: {self.acc_dict[1]['num_total']}"
        )
        print(
            f"negative preds: {self.acc_dict[0]['num_correct']}, \
            total negative samples: {self.acc_dict[0]['num_total']}"
        )
        acc = self.calc_acc()
        class_acc = self.calc_class_acc("test")

        self.log('my_test_acc', acc, prog_bar=True, logger=True)
        self.log('test_AUC', self.AUROC.compute(), prog_bar=True, logger=True)
        self.log_dict(class_acc, prog_bar=True, logger=True)
        self.log_dict(self.test_metrics.compute(), prog_bar=True, logger=True)

        output_file = Path(
            "/home/pml16/bash_scripts/seed_runs", f"{self.run_name}.json"
        )
        with open(output_file, "w") as f:
            json.dump(self.test_results, f, indent=4)
            artifact = wandb.Artifact("test_results", type="dataset")
        artifact.add_file(output_file)
        self.logger.experiment.log_artifact(artifact)

        self.test_metrics.reset()
        self.AUROC.reset()
        self.reset_acc_dict()

    def training_step(self, batch, batch_idx):
        logits, loss, labels, _, _ = self.common_step(batch, batch_idx)
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {'loss': loss}

    def on_train_epoch_end(self):
        pass

    # SGD optimizer config
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return [optimizer]

    #     # return [optimizer], [lr_scheduler]

    # Lookahead optimizer config
    # def configure_optimizers(self):
    #     radam = RAdam(
    #         self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay
    #     )
    #     optimizer = Lookahead(radam, alpha=0.5, k=5)
    #     return [optimizer]

    def calc_acc(self):
        acc = (self.correct_samples / self.total_samples) * 100
        self.total_samples = torch.tensor(0.0)
        self.correct_samples = torch.tensor(0.0)
        return acc

    def calc_class_acc(self, prefix: str):
        class_acc = {}
        for i in range(self.num_classes):
            if self.acc_dict[i]['num_total'] > 0:
                class_acc[f"{prefix}_class_{i} accuracy"] = (
                    self.acc_dict[i]['num_correct']
                    / self.acc_dict[i]['num_total']
                )
            else:
                class_acc[f"{prefix}_class_{i} accuracy"] = torch.tensor(
                    float('nan')
                )
        return class_acc

    def reset_acc_dict(self):
        for class_idx in range(self.num_classes):
            for key in self.acc_dict[class_idx]:
                self.acc_dict[class_idx][key] = 0
