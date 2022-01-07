import wandb
import torch
import argparse
import os
import numpy as np
from models import *
from utils import naive_lip


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--id', default="", type=str, help='wandb_id (if set --resume)')
parser.add_argument('--save_dir', default="", type=str, help='where to save wandb logs locally')

args = parser.parse_args()

run = wandb.init(project="ReNorm5.3", dir=args.save_dir, resume=True, id=args.id)
config = wandb.config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = VGG('VGG16', norm_layer=config.norm_type, **config.model_kwargs)
net = net.to(device)

wandb.restore('checkpoint/last.pth')
checkpoint = torch.load(os.path.join(wandb.run.dir, 'checkpoint/last.pth'))
missing_keys, _  = net.load_state_dict(checkpoint['net_state_dict'], strict=False)
print(missing_keys)
epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# global_step = run.step
# best_acc = checkpoint['best_acc']
print(epoch)
assert epoch == 99
lips = []
for i in range(5):
    lips.append(naive_lip(net, n_iter=100, eps=1e-7, bs=100, device=device))
lips = np.array(lips)
lip_mean = np.mean(lips)
lip_std = np.std(lips)
wandb.run.summary["lip_mean"] = lip_mean
wandb.run.summary["lip_std"] = lip_std