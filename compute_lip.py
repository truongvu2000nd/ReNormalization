import wandb
import torch
import argparse
import os
import numpy as np
from models import *
from utils import naive_lip
from tqdm import tqdm


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
net.eval()
epoch = checkpoint['epoch']
print(epoch)
assert epoch == 99
lips = []
print("Computing lip...")
for i in range(5):
    lips.append(naive_lip(net, n_iter=100, eps=1e-7, bs=100, device=device))
lips = np.array(lips)
lip_mean = np.mean(lips)
lip_std = np.std(lips)
wandb.run.summary["lip_mean"] = lip_mean
wandb.run.summary["lip_std"] = lip_std


# for param in net.parameters():
#     param.requires_grad_(False)

# bs = 100
# x1 = torch.randn(bs, 3, 32, 32, device=device)
# alpha = torch.rand(bs, 3, 32, 32, device=device, requires_grad=True)

# pbar = tqdm(range(1000))
# best_lip = -1
# try:
#     for i in pbar:
#         delta = (alpha * 2 - 1) * 1e-7
#         y1, y2 = net(x1), net(x1 + delta)
#         denominator = torch.linalg.vector_norm(delta.view(bs, -1), ord=2, dim=1)
#         numerator = torch.linalg.vector_norm((y2-y1).view(bs, -1), ord=2, dim=1)
#         lip = - torch.div(numerator, denominator).max()
#         lip.backward()
#         with torch.no_grad():
#             alpha -= alpha.grad * 0.05
#             alpha.grad.zero_()
#         pbar.set_postfix({'lip': lip.item()})
#         best_lip = max(best_lip, -lip.item())
# except KeyboardInterrupt:
#     print("-" * 89)
#     print("Exiting")

#     print(best_lip)
#     print(alpha.max(), alpha.min())

# print(best_lip)
# print(alpha.max(), alpha.min())
