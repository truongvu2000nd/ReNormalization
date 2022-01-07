'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
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
from collections import OrderedDict

from models import *
from utils import naive_lip


PROJECT_NAME = 'ReNorm5.3'


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--id', default="", type=str, help='wandb_id (if set --resume)')
parser.add_argument('--save_dir', default="", type=str, help='where to save wandb logs locally')
parser.add_argument('--config', default="config.yaml", type=str, help='wandb config file')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--n_epochs', default=100, type=int, help='num epochs')
parser.add_argument('--r', default=None, type=float, help='renorm param r')
parser.add_argument('--log_norm_state_every', default=100, type=int)
parser.add_argument('--use_scheduler', action='store_true', help="use learning rate scheduler")
parser.add_argument('--wandb_group', default="", type=str, help='wandb group')

args = parser.parse_args()

if args.resume:
    run = wandb.init(project=PROJECT_NAME, dir=args.save_dir, resume=True, id=args.id)
else:
    run = wandb.init(project=PROJECT_NAME, group=args.wandb_group, 
                     dir=args.save_dir, config=args.config)
config = wandb.config
if not args.resume:
    config.update({"lr": args.lr, "n_epochs": args.n_epochs, "model_kwargs": {"r": args.r}}, allow_val_change=True)
    config.use_scheduler = args.use_scheduler
    config.log_norm_state_every = args.log_norm_state_every
print(config)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
global_step, best_acc = 0, 0
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')


# --------------------------------------------------
# -----------------------Data-----------------------
# --------------------------------------------------
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
    trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=config.data_dir, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# ---------------------------------------------------
# -----------------------Model-----------------------
# ---------------------------------------------------
print('==> Building model..')
net = VGG('VGG16', norm_layer=config.norm_type, **config.model_kwargs)
# net = ResNet18(norm_layer=config.norm_type, **config.model_kwargs)
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

if config.watch_model:
    wandb.watch(net, log_freq=500)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=config.lr,
                      momentum=0.9, weight_decay=config.weight_decay)

if config.use_scheduler:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs, eta_min=1e-4)


if wandb.run.resumed:
    wandb.restore('checkpoint/last.pth')
    checkpoint = torch.load(os.path.join(wandb.run.dir, 'checkpoint/last.pth'))
    net.load_state_dict(checkpoint['net_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    global_step = run.step
    best_acc = checkpoint['best_acc']
    if config.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.n_epochs, last_epoch=start_epoch, eta_min=1e-4
        )
    

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
        optimizer.step()

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
    track_buffers = [
        "before_mean",
        "before_var",
        "after_mean",
        "after_var",
    ]
    if config.norm_type in ["bn", "rebn"]:
        track_buffers += ["running_mean", "running_var"]

    log_dicts = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
    
        log_dict = OrderedDict()
        for name, p in net.named_buffers():
            for n in track_buffers:
                if n in name:
                    flattened_p = torch.flatten(p).detach().float().clamp(max=1000)
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

        if (epoch + 1) % config.log_norm_state_every == 0 and epoch + 1 != config.n_epochs:
            log_norm_state()
    
    log_norm_state()
    lip = naive_lip(net, n_iter=100, eps=1e-7, bs=100, device=device)
    wandb.run.summary["lip"] = lip
