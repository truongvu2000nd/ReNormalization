import torch
import torch.nn as nn
import torch.nn.functional as F

from models import norm_layer


@torch.no_grad()
def naive_lip(model: nn.Module, n_iter: int = 5, eps=1e-3):
    lip = -1
    for i in range(n_iter):
        x1 = torch.randn(100, 3, 32, 32)
        alpha = torch.rand(100, 3, 32, 32) * eps

        y1, y2 = model(x1), model(x1 + alpha)
        denominator = torch.linalg.vector_norm(alpha.view(100, -1), ord=2, dim=1)
        numerator = torch.linalg.vector_norm((y2-y1).view(100, -1), ord=2, dim=1)
        lip = max(lip, torch.div(numerator, denominator).max().item())

    return lip


def grad_norm_lip(model: nn.Module, c_vector: torch.Tensor, n_iter: int = 500):
    max_norm = -1
    for i in range(n_iter):
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        output = model(x).mv(c_vector).sum()
        output.backward()
        grad = x.grad
        grad_norm = grad.norm(2)
        if grad_norm > max_norm:
            max_norm = grad_norm
            print(i, max_norm) 

    return max_norm


if __name__ == '__main__':
    from models import ResNet18, resnet18
    model = ResNet18(norm_layer="bn-torch")
    # lip = naive_lip(model, 30)
    # print(lip)
    c_vector = torch.zeros(10)
    c_vector[0] = 1
    print(c_vector)
    lip = naive_lip(model, c_vector, 5)

    lip = grad_norm_lip(model, c_vector)
    print(lip)
