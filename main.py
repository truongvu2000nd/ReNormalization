'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _NormBase
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from collections import OrderedDict
from functools import partial

from models import VGG, ResNet18, ResNet34, ResNet50
from utils import naive_lip


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--proj_name', default="", type=str, help='wandb project name')
parser.add_argument('--id', default="", type=str, help='wandb_id (if set --resume)')
parser.add_argument('--save_dir', default="", type=str, help='where to save wandb logs locally')
parser.add_argument('--dataset', default="cifar10", type=str, help='dataset')
parser.add_argument('--config', default="config.yaml", type=str, help='wandb config file')
parser.add_argument('--arch', default="ResNet18", type=str, help='model architecture')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--n_epochs', default=100, type=int, help='num epochs')
parser.add_argument('--log_norm_state_every', default=100, type=int)
parser.add_argument('--use_scheduler', action='store_true', help="use learning rate scheduler")
parser.add_argument('--watch_model', action='store_true', help="watch model gradients wandb")
parser.add_argument('--wandb_group', default="", type=str, help='wandb group')
parser.add_argument('--log_grad_norm', action='store_true', help="watch model gradients")
parser.add_argument('--log_weight_norm', action='store_true', help="watch model weights")
parser.add_argument('--compute_lip', action='store_true', help="estimate lipschitz")
parser.add_argument('--clip_grad', default=None, type=float, help="clipping gradient")
parser.add_argument('--clip_weight', default=None, type=float, help="clip weight")
parser.add_argument('--model_r', default=1.0, type=float, help="model kwargs r")
parser.add_argument('--straight_through', action='store_true', help="straight through")
parser.add_argument('--resume_from_best', action='store_true', help="resume from best")


args = parser.parse_args()

PROJECT_NAME = 'ResNet'
if args.proj_name:
    PROJECT_NAME = args.proj_name

if args.resume:
    run = wandb.init(project=PROJECT_NAME, dir=args.save_dir, resume=True, id=args.id, entity="truongvu2000")
else:
    run = wandb.init(project=PROJECT_NAME, group=args.wandb_group, 
                     dir=args.save_dir, config=args.config, entity="truongvu2000")
config = wandb.config
if not args.resume:
    config.update({"lr": args.lr, "n_epochs": args.n_epochs, "watch_model": args.watch_model, "arch": args.arch},
                   allow_val_change=True)
    if config.norm_type in ["regn", "rebn", "rebn-cpp"]:
        config.update({"model_kwargs": {"r": args.model_r, "straight_through": args.straight_through}},
                    allow_val_change=True) 
    config.use_scheduler = args.use_scheduler
    # config.log_norm_state_every = args.log_norm_state_every
    config.clip_weight = args.clip_weight
print(config)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
global_step, best_acc = 0, 0
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')


# --------------------------------------------------
# -----------------------Data-----------------------
# --------------------------------------------------
if args.dataset == "cifar10":
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


    trainset = torchvision.datasets.CIFAR10(
        root=config.data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(
        root=config.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)


elif args.dataset == "svhn":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.SVHN(
        root=config.data_dir, split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.SVHN(
        root=config.data_dir, split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)


# ---------------------------------------------------
# -----------------------Model-----------------------
# ---------------------------------------------------
print('==> Building model..')
net_dict = {
    "vgg": partial(VGG, 'VGG16'),
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
}
net = net_dict[config.arch](norm_layer=config.norm_type, **config.model_kwargs)
net = net.to(device)
if args.clip_weight is not None:
    with torch.no_grad():
        for name, p in net.named_parameters():
            if ".weight" in name:
                p.clamp_(-abs(args.clip_weight), abs(args.clip_weight))
if device == 'cuda':
    cudnn.benchmark = True

if config.watch_model:
    wandb.watch(net, log_freq=500)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=config.lr,
                      momentum=0.9, weight_decay=config.weight_decay)

if config.use_scheduler:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs, eta_min=5e-5)


if wandb.run.resumed:
    if args.resume_from_best:
        checkpoint_path = 'checkpoint/best.pth'
    else:
        checkpoint_path = 'checkpoint/last.pth'

    wandb.restore(checkpoint_path)
    checkpoint = torch.load(os.path.join(wandb.run.dir, checkpoint_path))

    net.load_state_dict(checkpoint['net_state_dict'], strict=False)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    global_step = run.step
    if 'best_acc' in checkpoint:
        best_acc = checkpoint['best_acc']
    else:
        best_acc = checkpoint['acc']
    if config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.n_epochs, last_epoch=-1, eta_min=5e-5
        )
        for _ in range(start_epoch):
            scheduler.step()
    

# ------------------------------------------------------
# -----------------------Training-----------------------
# ------------------------------------------------------
def train(epoch):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        global_step += 1
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)
        optimizer.step()

        with torch.no_grad():
            if args.clip_weight is not None:
                for name, p in net.named_parameters():
                    if ".weight" in name:
                        p.clamp_(-abs(args.clip_weight), abs(args.clip_weight))

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % config.log_every == 0:
            print('[Train]-[%d/%d]: Loss: %.3f | Acc: %.3f%% (%d/%d)' 
                    % (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            wandb.log({
                "train_loss": train_loss/(batch_idx+1), 
                "train_acc": 100.*correct/total}, step=global_step)

            if args.log_grad_norm:
                for name, p in net.named_parameters():
                    grad_norm = p.grad.detach().data.max()
                    wandb.log({f"gradient_norm/{name}": grad_norm}, step=global_step)

            if args.log_weight_norm:
                for name, p in net.named_parameters():
                    param_norm = p.detach().data.max()
                    wandb.log({f"weight_norm/{name}": param_norm}, step=global_step)
            
    print("End of epoch {} | Training time: {:.2f}s".format(epoch, time.time() - start_time))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        test_loss = test_loss / (batch_idx+1)
        print('[Test]: Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss, 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    wandb.log({"test_loss": test_loss, "test_acc": acc}, step=global_step)
    print('Saving..')
    
    if acc > best_acc:
        state = {
            'net_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': acc,
            'loss': loss,
            'epoch': epoch,
        }
        torch.save(state, 'checkpoint/best.pth')
        wandb.save('checkpoint/best.pth')
        best_acc = acc
        wandb.run.summary["best_accuracy"] = best_acc

    state = {
        'epoch': epoch,
        'net_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'acc': acc,
        'best_acc': best_acc
    }
    torch.save(state, 'checkpoint/last.pth')
    wandb.save('checkpoint/last.pth')

# --------------------------------------------------------------------------------------------
# -----------------------Log layer state before and after normalization-----------------------
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def log_norm_state():
    net.eval()
    print("\nLogging normalization layer states...")
    track_buffers = ["before_var", "after_var"]
    if config.norm_type in ["bn", "rebn"]:
        track_buffers += ["running_var"]

    if config.norm_type == "no-norm":
        track_buffers = ["before_var_bn", "before_var_gn_2", "before_var_gn_4", "before_var_gn_8", "before_var_gn_16"]

    log_dicts = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
    
        log_dict = OrderedDict()
        for name, p in net.named_buffers():
            for n in track_buffers:
                if n in name:
                    flattened_p = torch.flatten(p).detach().float()
                    layer_name = name[: name.rindex(".")]
                    log_dict[f"{n}/{layer_name}"] = flattened_p
        log_dicts.append(log_dict)

    log_norms = OrderedDict()
    for key in log_dict.keys():
        log_norms[key] = []

    for log_norm in log_dicts:
        for key, value in log_norm.items():
            log_norms[key].append(value)
    
    for key, value in log_norms.items():
        log_norms[key] = torch.cat(value, dim=0)

    for stat in track_buffers:
        boxes = np.array([val.cpu().numpy() for key, val in log_norms.items() if stat in key], dtype=object).T
        fig = plt.figure(dpi=150)
        plt.boxplot(boxes)
        plt.xlabel("layer")
        plt.ylabel(stat)
        wandb.log({f"boxplot/{stat}": wandb.Image(fig)}, step=global_step)
        plt.yscale('log')
        wandb.log({f"boxplot_log_scale/{stat}": wandb.Image(fig)}, step=global_step)
        plt.close("all")


if __name__ == '__main__':
    for epoch in range(start_epoch, config.n_epochs):
        train(epoch)
        test(epoch)
        if config.use_scheduler:
            scheduler.step()
            wandb.log({"lr": scheduler.get_last_lr()[0]}, step=global_step)

        # if (epoch + 1) % config.log_norm_state_every == 0 and epoch + 1 != config.n_epochs:
        #     log_norm_state()
    
    checkpoint = torch.load("checkpoint/best.pth")
    missing_keys, _  = net.load_state_dict(checkpoint['net_state_dict'], strict=False)
    print(missing_keys)
    net.eval()
    norm_fro_sh, norm_max_sh, norm_l2_sh, norm_fro_cw, norm_max_cw, norm_l2_cw = 0., 0., 0., 0., 0., 0.
    n_params_sh, n_params_cw = 0, 0
    for name, p in net.named_modules():
        if isinstance(p, (nn.Conv2d, nn.Linear)):
            norm_fro_sh += torch.linalg.matrix_norm(p.weight.view(p.weight.size(0), -1)).log10().item()
            norm_max_sh += p.weight.abs().max().log10().item()
            norm_l2_sh += torch.linalg.matrix_norm(p.weight.view(p.weight.size(0), -1), ord=2).log10().item()
            n_params_sh += torch.numel(p.weight)

        if isinstance(p, (nn.Conv2d, nn.Linear, _NormBase, nn.GroupNorm)):
            norm_fro_cw += torch.linalg.matrix_norm(p.weight.view(p.weight.size(0), -1)).log10().item()
            norm_max_cw += p.weight.abs().max().log10().item()
            norm_l2_cw += torch.linalg.matrix_norm(p.weight.view(p.weight.size(0), -1), ord=2).log10().item()
            n_params_cw += torch.numel(p.weight)

    wandb.run.summary["norm_fro_sh"] = norm_fro_sh
    wandb.run.summary["norm_max_sh"] = norm_max_sh
    wandb.run.summary["norm_l2_sh"] = norm_l2_sh
    wandb.run.summary["n_params_sh"] = n_params_sh

    wandb.run.summary["norm_fro_cw"] = norm_fro_cw
    wandb.run.summary["norm_max_cw"] = norm_max_cw
    wandb.run.summary["norm_l2_cw"] = norm_l2_cw
    wandb.run.summary["n_params_cw"] = n_params_cw

    if args.compute_lip:
        checkpoint = torch.load("checkpoint/best.pth")
        missing_keys, _  = net.load_state_dict(checkpoint['net_state_dict'], strict=False)
        print(missing_keys)
        net.eval()
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=1, shuffle=False, num_workers=2)
        max_norm1, max_norm2 = -1, -1
        pbar = tqdm(testloader)
        for p in net.parameters():
            p.requires_grad_(False)
        for x, _ in pbar:
            x = x.view(-1).to(device)
            x.requires_grad_(True)
            jacob = torch.autograd.functional.jacobian(lambda x: net(x.view(1, 3, 32, 32)).view(-1), x)
            grad_norm1 = torch.linalg.matrix_norm(jacob, ord=float('inf')).item()
            grad_norm2 = jacob.max().item()
            if grad_norm1 > max_norm1:
                max_norm1 = grad_norm1
            if grad_norm2 > max_norm2:
                max_norm2 = grad_norm2
            pbar.set_description("Max norm1: {:.6f} {:.6f}".format(max_norm1, max_norm2))
        wandb.run.summary["lip_max_norm"] = max_norm1
        wandb.run.summary["lip_max_max_norm"] = max_norm2
