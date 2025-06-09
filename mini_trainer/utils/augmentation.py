import torch
from math import prod

def salt_and_pepper(img : torch.Tensor, proportion : tuple[float, float]=(0, 0.5), probability : float=1):
    if torch.rand((1, )).item() > probability:
        return img
    p = torch.rand((1, )).item() * (proportion[1] - proportion[0]) + proportion[0]
    if len(img.shape) <= 3:
        m = torch.rand(img.shape[-2:], dtype=torch.float16) < p
    else:
        m = torch.rand((img.shape[0], 1, *img.shape[-2:]), dtype=torch.float16) < p
    s = torch.rand_like(m, dtype=torch.float16) <= 0.5
    img.masked_fill_(m & s, 0)
    img.masked_fill_(m & ~s, 1 if img.is_floating_point() else torch.iinfo(img.dtype).max)
    return img

class SaltAndPepper(torch.nn.Module):
    def __init__(self, proportion : tuple[float, float]=(0.05, 0.33), probability : float=0.5):
        self.proportion = proportion
        self.probability = probability

    def forward(self, img):
        return salt_and_pepper(img, proportion=self.proportion, probability=self.probability)