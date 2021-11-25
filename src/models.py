import torch
import torch.nn as nn

import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim=(3, 32, 32), hidden_dim=256, n_blocks=5, out_dim=10, norm_layer=None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.fc = nn.Linear(np.prod(input_dim), hidden_dim)

        self.layers = []
        for _ in range(n_blocks):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(),
            )

            self.layers.append(block)

        self.layers = nn.Sequential(*self.layers)
        self.last_fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.layers(x)
        x = self.last_fc(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    model = MLP(n_blocks=5)
    summary(model, (1, 3, 32, 32))
