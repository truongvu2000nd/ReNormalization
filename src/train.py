import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI
from torchmetrics.functional import accuracy
from collections import OrderedDict, defaultdict

import wandb
from utils import create_mlp_model, create_resnet18_model
import plotly.express as px
import matplotlib.pyplot as plt
import yaml
import argparse


class LitModel(LightningModule):
    def __init__(self, arch, norm_type="bn", lr=0.1, weight_decay=5e-4, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.norm_type = norm_type
        if arch == "mlp":
            self.model = create_mlp_model(norm_type, **kwargs)
        elif arch == "resnet18":
            self.model = create_resnet18_model(norm_type, **kwargs)
        else:
            raise NotImplementedError

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        # if (self.global_step + 1) % self.trainer.log_every_n_steps == 0:
        #     self.log_norm_state()
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        x, _ = batch
        self.model(x)
        log_norm = self.log_norm_state(log=False)
        return log_norm

    def test_epoch_end(self, outputs) -> None:
        track_buffers = [
            "before_mean",
            "before_var",
            "after_mean",
            "after_var",
            "after_affine_mean",
            "after_affine_var",
        ]
        log_norms = OrderedDict()
        for key in outputs[0].keys():
            log_norms[key] = []

        for log_norm in outputs:
            for key, value in log_norm.items():
                log_norms[key].append(value)
        
        for key, value in log_norms.items():
            log_norms[key] = torch.cat(value, dim=0)

        for stat in track_buffers:
            boxes = torch.vstack([val for key, val in log_norms.items() if stat in key]).cpu().numpy().T
            fig = plt.figure(dpi=200)
            plt.boxplot(boxes)
            plt.xlabel("layer")
            plt.ylabel(stat)
            self.logger.experiment.log({f"boxplot/{stat}": wandb.Image(fig)})

    # Log layer state before and after normalization
    def log_norm_state(self, log=True):
        track_buffers = [
            "before_mean",
            "before_var",
            "after_mean",
            "after_var",
            "after_affine_mean",
            "after_affine_var",
        ]
        if self.norm_type == "bn":
            track_buffers += ["running_mean", "running_var"]
        log_dict = OrderedDict()

        for name, p in self.model.named_buffers():
            for n in track_buffers:
                if n in name:
                    flattened_p = torch.flatten(p).detach().float().clamp(max=1000)
                    layer_name = name[: name.rindex(".")]
                    log_dict[f"{n}/{layer_name}"] = flattened_p

                    if log:
                        self.logger.experiment.log(
                            {
                                f"{n}/{layer_name}": wandb.Histogram(
                                    flattened_p.to("cpu")
                                ),
                                "global_step": self.global_step,
                            }
                        )

        return log_dict

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3,)
        # return {"optimizer": optimizer}

        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
                                    momentum=0.9, weight_decay=self.hparams.weight_decay)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100),
            'interval': 'step'  # called after each training step
        }
        return [optimizer], [scheduler]


def main(conf):
    seed_everything(conf["seed"])
    model = LitModel(conf["arch"], conf["norm_type"], conf["lr"], conf["weight_decay"], **conf["model_kwargs"])

    # -------- CIFAR10 --------
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=conf["data_dir"], train=True, download=True, transform=train_transforms
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=conf["batch_size"], shuffle=True, 
        drop_last=False, num_workers=conf["num_workers"], pin_memory=True,
    )

    valset = torchvision.datasets.CIFAR10(
        root=conf["data_dir"], train=False, download=True, transform=test_transforms
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=200, shuffle=False, 
        num_workers=conf["num_workers"], pin_memory=True
    )

    # -------- Callbacks & Trainer --------
    name = f"{conf['arch']}-{conf['norm_type']}-seed:{conf['seed']}"
    group = conf['arch']
    tags = [conf['norm_type']]
    wandb_logger = WandbLogger(
        name, save_dir=conf["save_dir"], project="ReNormalization", log_model=True, group=group, tags=tags,
    )
    checkpoint_callback = ModelCheckpoint(save_last=True, save_top_k=0)
    callbacks = [checkpoint_callback]

    trainer = Trainer(
        max_epochs=conf["max_epochs"],
        log_every_n_steps=conf["log_every_n_steps"],
        devices=1,
        accelerator="auto",
        callbacks=callbacks,
        logger=wandb_logger,
        precision=32,
        fast_dev_run=0,
    )

    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
    trainer.test(model, valloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="config file")

    args = parser.parse_args()
    with open(args.config, "r") as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    main(conf)
