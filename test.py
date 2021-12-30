'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import wandb
from collections import OrderedDict
from tqdm import tqdm

from models import *


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
    root="../datasets/CIFAR10", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root="../datasets/CIFAR10", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# ---------------------------------------------------
# -----------------------Model-----------------------
# ---------------------------------------------------
print('==> Building model..')
net = VGG('VGG16', norm_layer="bn")
# net = ResNet18(norm_layer="rebn")
# net = net.to(device)
# if device == 'cuda':
#     # net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
checkpoint = torch.load("report/best_2.pth", map_location="cpu")
# print(checkpoint.keys())

missing_keys, _ = net.load_state_dict(checkpoint['net_state_dict'], strict=False)
print(missing_keys)

# def test():
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in tqdm(enumerate(testloader), total=len(testloader)):
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#         test_loss = test_loss / (batch_idx+1)
#         print('[Test]: Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                     % (test_loss, 100.*correct/total, correct, total))

@torch.no_grad()
def log_norm_state():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    print("\nLogging normalization layer states...")
    track_buffers = [
        # "before_mean",
        "before_var",
        # "after_mean",
        "after_var",
    ]

    log_dicts = []
    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader), total=len(testloader)):
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
        log_dict = OrderedDict()
        for name, p in net.named_buffers():
            for n in track_buffers:
                if n in name:
                    flattened_p = torch.flatten(p).detach().float().clamp(max=1000)
                    layer_name = name[: name.rindex(".")]
                    log_dict[f"{n}/{layer_name}"] = flattened_p
        log_dicts.append(log_dict)
    
    test_loss = test_loss / (batch_idx+1)
    print('[Test]: Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss, 100.*correct/total, correct, total))

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
        # wandb.log({f"boxplot/{stat}": wandb.Image(fig)})
        # plt.show()
        plt.yscale('log')
        # wandb.log({f"boxplot_log_scale/{stat}": wandb.Image(fig)})
        plt.show()
        plt.close("all")

log_norm_state()
