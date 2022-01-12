from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models import norm_layer


@torch.no_grad()
def naive_lip(model: nn.Module, n_iter: int = 100, eps=1e-3, bs=100, device="cpu"):
    lip = -1
    for i in range(n_iter):
        x1 = torch.randn(bs, 3, 32, 32, device=device)
        alpha = (torch.rand(bs, 3, 32, 32, device=device) * 2 - 1) * eps

        y1, y2 = model(x1), model(x1 + alpha)
        denominator = torch.linalg.vector_norm(alpha.view(bs, -1), ord=float('inf'), dim=1)
        numerator = torch.linalg.vector_norm((y2-y1).view(bs, -1), ord=float('inf'), dim=1)
        print(numerator.dtype)
        lip = max(lip, torch.div(numerator, denominator).max().item())
        print(lip)

    return lip


def grad_norm_lip(model: nn.Module, n_iter: int = 50):
    max_norm = -1
    for i in range(n_iter):
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        output = model(x).sum()
        output.backward()
        grad = x.grad
        # print(grad.size())
        grad_norm = grad.view(1, -1).norm(float('inf'))
        if grad_norm > max_norm:
            max_norm = grad_norm
            print(i, max_norm) 

    return max_norm


if __name__ == '__main__':
    from models import ResNet18, resnet18
    model = ResNet18(norm_layer="bn-torch")
    lip = naive_lip(model, 5, eps=1)
    print(lip)

    # lip = grad_norm_lip(model)
    # print(lip)

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # testset = torchvision.datasets.CIFAR10(
    #     root="../datasets/CIFAR10", train=False, download=False, transform=transform_test)
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=1, shuffle=False, num_workers=2)

    # max_norm = -1
    # for x in range(1000):
    #     x = torch.randn(1, 3, 32, 32, requires_grad=True)
    #     # x.requires_grad_(True)
    #     output = model(x).sum()
    #     output.backward()
    #     grad = x.grad
    #     grad_norm = grad.view(1, -1).norm(float('inf'))
    #     if grad_norm > max_norm:
    #         max_norm = grad_norm
    #         print(max_norm)
