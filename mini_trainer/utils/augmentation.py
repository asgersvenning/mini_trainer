import os
from random import sample
from typing import Callable, Optional

import torch
from matplotlib import pyplot as plt
from mini_trainer.utils import make_convert_dtype

def debug_augmentation(
        augmentation : Callable[[torch.Tensor], torch.Tensor],
        dataset : torch.utils.data.Dataset,
        output_dir : Optional[str]=None,
        strict : bool=True
    ):
    convert2fp32 = make_convert_dtype(torch.float32)
    try:
        n = min(3, len(dataset))
        fig, axs = plt.subplots(3, n, figsize=(10, 5))

        for j, i in enumerate(sample(range(len(dataset)), n)):
            example_image : torch.Tensor = dataset[i][0].clone().cpu()
            
            axs[j, 0].imshow(example_image.permute(1,2,0))
            axs[j, 1].imshow(convert2fp32(augmentation(example_image).permute(1,2,0)))
            axs[j, 2].imshow(convert2fp32(augmentation(example_image).permute(1,2,0)))
            for ax in axs[j, :]:
                ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "example_augmentation.png") if output_dir is not None else "example_augmentation.png")
        plt.close()
    except Exception as e:
        e_msg = (
            "Error while attempting to create debug augmentation image."
            "Perhaps the supplied dataloader doesn't return items (image, label) in the expected format."
        )
        e.add_note(e_msg)
        if strict:
            raise e
        print(e_msg)
        return False
    return True

def salt_and_pepper(img : torch.Tensor, proportion : tuple[float, float]=(0, 0.5), probability : float=1):
    if torch.rand((1, )).item() > probability:
        return img
    p = torch.rand((1, )).item() * (proportion[1] - proportion[0]) + proportion[0]
    if len(img.shape) <= 3:
        m = torch.rand(img.shape[-2:], dtype=torch.float16, device=img.device) < p
    else:
        m = torch.rand((img.shape[0], 1, *img.shape[-2:]), dtype=torch.float16, device=img.device) < p
    s = torch.rand_like(m, dtype=torch.float16) <= 0.5
    img.masked_fill_(m & s, 0)
    img.masked_fill_(m & ~s, 1 if img.is_floating_point() else torch.iinfo(img.dtype).max)
    return img

class SaltAndPepper(torch.nn.Module):
    def __init__(self, proportion : tuple[float, float]=(0.05, 0.33), probability : float=0.5):
        super().__init__()
        self.proportion = proportion
        self.probability = probability

    def forward(self, img):
        return salt_and_pepper(img, proportion=self.proportion, probability=self.probability)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(proportion={self.proportion}, probability={self.probability})"