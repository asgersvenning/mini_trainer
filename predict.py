import json
import os
from argparse import ArgumentParser
from glob import glob
from math import ceil
from typing import Callable, Iterable, Union

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.functional import resize
from tqdm import tqdm as TQDM

from train import (Classifier, convert2bf16, convert2fp16, convert2fp32,
                   convert2uint8, get_model)
from utils import is_image, parse_class_index


class ImageDataset(Dataset):
    def __init__(self, func : Callable[[str], torch.Tensor], items : list[str]):
        self.func = func
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.func(self.items[i])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

class ImageLoader:
    def __init__(self, preprocessor, dtype, device):
        self.dtype, self.device = dtype, device
        self.preprocessor = preprocessor
        match self.dtype:
            case torch.float16:
                self.converter = convert2fp16
            case torch.float32:
                self.converter = convert2fp32
            case torch.bfloat16:
                self.converter = convert2bf16
            case _:
                raise ValueError("Only fp16 supported for now.")
        size = self.preprocessor.resize_size if hasattr(self.preprocessor, "resize_size") else 256
        self.shape = size if not isinstance(size, int) and len(size) == 2 else (size, size)
    
    def __call__(self, x : Union[str, Iterable]):
        if isinstance(x, str):
            proc_img : torch.Tensor = self.preprocessor(resize(convert2fp32(decode_image(x, ImageReadMode.RGB)), self.shape)).to(self.device, self.dtype)
            return proc_img
        return ImageDataset(self, x)

def find_images(root : str):
    paths = glob(os.path.join(root, "**"), recursive=True)
    return list(filter(is_image, paths))

if __name__ == "__main__":  
    parser = ArgumentParser(
        prog = "predict",
        description = "Predict with a simple classifier"
    )  
    parser.add_argument(
        "-m", "--model", type=str, default="efficientnet_v2_s", required=True,
        help="name of the model type from the torchvision model zoo (https://pytorch.org/vision/main/models.html#table-of-all-available-classification-weights). Not case-sensitive."
    )
    parser.add_argument(
        "-w", "--weights", type=str, required=True,
        help="Model weights for inference."
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="path to a directory containing a subdirectory for each class, where the name of each subdirectory should correspond to the name of the class."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="result.json", required=False,
        help='Path used to store inference results (default="result.json").'
    )
    parser.add_argument(
        "-C", "--class_index", type=str, default="class_index.json", required=True,
        help="path to a JSON file containing the class name to index mapping. If it doesn't exist, one will be created based on the directories found under `input`."
    )
    parser.add_argument(
        "--training_format", action="store_true", required=False,
        help="Are the images in `input` stored in subfolders named by their class? If so, we can calculate accuracy statistics."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, required=False,
        help="Batch size used for inference. Higher requires more VRAM."
    )
    parser.add_argument(
        "--num_workers", type=int, default=None, required=False,
        help="Number of workers used for reading/loading images for inference. Default is set to number of physical CPU cores." 
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", required=False,
        help='Device used for inference (default="cuda:0").'
    )
    parser.add_argument(
        "--dtype", type=str, default="float16", required=False,
        help="PyTorch data type used for inference (default=float16)."
    )
    ARGS = parser.parse_args()

    # Prepare state
    device = torch.device(ARGS.device)
    dtype = getattr(torch, ARGS.dtype)

    input_dir = os.path.abspath(ARGS.input)

    workers = ARGS.num_workers
    if workers is None:
        workers = os.cpu_count() - 1
        workers -= workers % 2
    batch_size = ARGS.batch_size

    CLASSES, class2idx, idx2class, num_classes = parse_class_index(ARGS.class_index)

    # Prepare model
    model, head_name, model_preprocess = get_model(ARGS.model)
    num_embeddings = getattr(model, head_name)[1].in_features
    if ARGS.weights is not None:
        model = Classifier.load(ARGS.model, ARGS.weights, device=device, dtype=dtype)
    else:
        setattr(model, head_name, Classifier(num_embeddings, num_classes))
        model.to(device, dtype)
    model.eval()
        

    # Prepare image loader
    image_loader = ImageLoader(
        model_preprocess,
        dtype,
        torch.device("cpu")
    )
    images = find_images(input_dir)
    ds = image_loader(images)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        pin_memory_device=device.type,
        shuffle=False,
        drop_last=False
    )

    # Inference
    results = {
        "path" : [],
        "pred" : [],
        "conf" : []
    }
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    with torch.no_grad():
        for batch_i, batch in TQDM(enumerate(dl), desc="Running inference...", total=ceil(len(images) / batch_size), leave=True):
            i = batch_i * batch_size
            prediction = model(batch.to(device))
            results["path"].extend(images[i:(i+len(batch))])
            results["pred"].extend([idx2class[idx] for idx in prediction.argmax(1).tolist()])
            results["conf"].extend(prediction.softmax(1).max(0).values.tolist())
    
    # Write results
    with open(ARGS.output, "w") as f:
        json.dump(results, f)

    print(f'Outputs written to {os.path.abspath(ARGS.output)}')
    end.record()
    torch.cuda.synchronize(device)
    inf_time = start.elapsed_time(end) / 1000
    print(f'Inference took {inf_time:.1f}s ({len(images)/inf_time:.1f} img/s)')

    if ARGS.training_format:
        results["gt"] = [f.split(os.sep)[-2] for f in results["path"]]

        confusion_matrix(results, idx2class)



    