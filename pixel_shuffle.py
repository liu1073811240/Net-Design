import torch

a = torch.randn(100, 16, 224, 224)

b = torch.pixel_shuffle(a, 2)
print(b.shape)  # torch.Size([100, 4, 448, 448])

c = torch.nn.PixelShuffle(2)
d = c(a)
print(d.shape)  # torch.Size([100, 4, 448, 448])


