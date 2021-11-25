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
from collections import OrderedDict

import wandb
from utils import create_mlp_model, create_resnet18_model
import yaml
import argparse


class LitModel(LightningModule):
    def __init__(self, arch, norm_type="bn", **kwargs):
        super().__init__()
        self.norm_type = norm_type
        if arch == "mlp":
            self.model = create_mlp_model(norm_type, **kwargs)
        elif arch == "resnet18":
            self.model = create_resnet18_model(norm_type)
        else:
            raise NotImplementedError

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss, sync_dist=True)
        if self.global_step % self.trainer.log_every_n_steps == 0:
            self.log_norm_state()
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

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
                    flattened_p = torch.flatten(p).detach().float().clamp(max=100)
                    layer_name = name[: name.rindex(".")]
                    log_dict[f"{n}/{layer_name}"] = flattened_p

                    if log:
                        self.logger.experiment.log(
                            {
                                f"hist_{n}/{layer_name}": wandb.Histogram(
                                    flattened_p.to("cpu")
                                ),
                                "global_step": self.global_step,
                            }
                        )

        return log_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3,)
        return {"optimizer": optimizer}


def main(conf):
    seed_everything(conf["seed"])
    model = LitModel(conf["arch"], conf["norm_type"])

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
        valset, batch_size=100, shuffle=False, 
        num_workers=conf["num_workers"], pin_memory=True
    )

    # -------- Callbacks & Trainer --------
    name = f"model:{conf['arch']}-norm:{conf['norm_type']}-seed:{conf['seed']}"
    wandb_logger = WandbLogger(
        name, save_dir=conf["save_dir"], project="ReNormalization", log_model=True
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
        precision=16,
        fast_dev_run=0,
    )

    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="config file")

    args = parser.parse_args()
    with open(args.config, "r") as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    main(conf)
