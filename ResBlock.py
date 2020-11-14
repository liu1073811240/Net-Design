import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, padding=1, bias=False),

        )

    def forward(self, x):
        return self.layers(x) + x


if __name__ == '__main__':
    x = torch.randn(1, 16, 112, 112)
    resBlock = ResBlock()
    print(resBlock)
    y = resBlock(x)
    print(y.shape)


