import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional import accuracy
from collections import OrderedDict

import wandb

from utils import create_mlp_model, create_resnet_model
import argparse

seed_everything(42)


class LitModel(LightningModule):
    def __init__(self, model_type, norm_type="bn", **kwargs):
        super().__init__()
        self.norm_type = norm_type
        if model_type == "mlp":
            self.model = create_mlp_model(norm_type, **kwargs)
        else:
            self.model = create_resnet_model(norm_type)

    def forward(self, x):
        out = self.model(x)
        log_norm = self.log_norm_state(log=False)
        return F.log_softmax(out, dim=1), log_norm

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        self.log_norm_state()
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
        self.evaluate(batch, "test")

    def log_norm_state(self, log=True):
        track_buffers = ["before_mean", "before_var", "after_mean", "after_var", "after_affine_mean", "after_affine_var"]
        if self.norm_type == "bn":
            track_buffers += ["running_mean", "running_var"]
        track_stats = ["min", "max", "mean", "norm"]
        log_dict = OrderedDict()
        # for buffer in track_buffers:
        #     for stat in track_stats:
        #         log_dict[buffer + "/" + stat] = {} 

        for name, p in self.model.named_buffers():
            for n in track_buffers:
                if n in name:
                    flattened_p = torch.flatten(p).detach()
                    layer_name = name[:name.rindex(".")]
                    log_dict[f"{n}/{layer_name}"] = flattened_p

                    if log:
                        self.logger.experiment.log(
                            {f"hist_{n}/{layer_name}": wandb.Histogram(flattened_p.to("cpu")),
                            "global_step": self.global_step})

        return log_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
            weight_decay=5e-4,
        )
        return {"optimizer": optimizer}


def main(args):
    model = LitModel(args.model_type, args.norm_type, n_blocks=args.n_blocks)
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

    name_run = f"{args.model_type}-{args.n_blocks}-{args.norm_type}"
    wandb_logger = WandbLogger(name_run, save_dir=args.save_dir, project="norm_layer")
    checkpoint_callback = ModelCheckpoint(save_last=True, save_top_k=0)
    callbacks = [checkpoint_callback]

    wandb_logger.watch(model, log_freq=100)

    trainer = Trainer(
        max_epochs=50,
        # max_steps=50,
        log_every_n_steps=10,
        devices="auto",
        accelerator="auto",
        callbacks=callbacks,
        logger=wandb_logger
        # fast_dev_run=3,
    )

    trainer.fit(model, train_dataloaders=trainloader,
                val_dataloaders=testloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../datasets/CIFAR10", type=str, help="data directory")
    parser.add_argument("--save_dir", default="", type=str, help="root dir")
    parser.add_argument("--name_run", default=None, type=str, help="name_run")

    parser.add_argument("--model_type", default="mlp", choices=["mlp", "resnet18"], help="model")
    parser.add_argument("--n_blocks", default=5, type=int, help="num nn blocks")
    parser.add_argument("--norm_type", default="bn", type=str, help="normalization layer")

    args = parser.parse_args()
    main(args)
