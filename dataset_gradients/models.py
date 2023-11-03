import torch.nn as nn


class MnistMLP(nn.Module):
    """Simple MLP for MNIST classification"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(784, 500), nn.ReLU(), nn.Linear(500, 10))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
