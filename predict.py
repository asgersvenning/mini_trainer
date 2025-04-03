import json
import os
from argparse import ArgumentParser
from glob import glob
from typing import Union, Iterable, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.functional import resize
from tqdm import tqdm as TQDM
from train import (Classifier, convert2bf16, convert2fp16, convert2fp32,
                   convert2uint8, get_model, is_image)

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
        prog = "train",
        description = "Train a simple classifier"
    )  
    parser.add_argument(
        "--model", type=str, default="efficientnet_v2_s", required=True,
        help="name of the model type from the torchvision model zoo (https://pytorch.org/vision/main/models.html#table-of-all-available-classification-weights). Not case-sensitive."
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Model weights for inference."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="path to a directory containing a subdirectory for each class, where the name of each subdirectory should correspond to the name of the class."
    )
    parser.add_argument(
        "--output", type=str, default="result.json", required=False,
        help='Path used to store inference results (default="result.json").'
    )
    parser.add_argument(
        "--class_index", type=str, default="class_index.json", required=False,
        help="path to a JSON file containing the class name to index mapping. If it doesn't exist, one will be created based on the directories found under `input`."
    )
    parser.add_argument(
        "--training_format", action="store_true", required=False,
        help="Are the images in `input` stored in subfolders named by their class? If so, we can calculate accuracy statistics."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, required=False,
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
    workers = ARGS.num_workers
    if workers is None:
        workers = os.cpu_count()
    batch_size = ARGS.batch_size

    if not os.path.exists(ARGS.class_index):
        _class2idx = {cls : i for i, cls in enumerate(sorted([f for f in map(os.path.basename, os.listdir(ARGS.input)) if os.path.isdir(os.path.join(ARGS.input, f))]))}
        with open(ARGS.class_index, "w") as f:
            json.dump(_class2idx, f)
    with open(ARGS.class_index, "rb") as f:
        class2idx = json.load(f)
    CLASSES = list(class2idx.keys())
    idx2class = {v : k for k, v in class2idx.items()}
    num_classes = len(idx2class)

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
    images = find_images(ARGS.input)
    ds = image_loader(images)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        pin_memory_device=device.type,
        shuffle=False
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
        for batch_i, batch in TQDM(enumerate(dl), desc="Running inference...", total=len(images) // batch_size + 1):
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
        # Build confusion matrix and compute accuracies
        classes = sorted([idx2class[i] for i in range(num_classes)])
        classes_set = set(classes)
        # First, collect all unique classes from ground-truth (extracted from path) and predictions.
        for f, p in zip(results["path"], results["pred"]):
            fps = f.split(os.sep)
            gt = fps[-2]  # ground truth assumed to be the parent folder name
            classes_set.add(gt)
            classes_set.add(p)

        # Initialize confusion matrix and counters
        conf_mat = {gt: {pred: 0 for pred in classes} for gt in classes}
        total_correct = 0
        total_samples = 0
        per_class_total = {cls: 0 for cls in classes}
        per_class_correct = {cls: 0 for cls in classes}

        # Populate confusion matrix and count correct predictions
        for f, p in zip(results["path"], results["pred"]):
            fps = f.split(os.sep)
            gt = fps[-2]
            conf_mat[gt][p] += 1
            total_samples += 1
            per_class_total[gt] += 1
            if gt.lower().strip() == p.lower().strip():
                total_correct += 1
                per_class_correct[gt] += 1

        # Print the confusion matrix (numbers only, aligned)
        max_cf_n = max(val for d in conf_mat.values() for val in d.values())
        width = len(str(max_cf_n))
        for gt in classes:
            row_str = "|".join(
                "{:>{width}d}".format(conf_mat[gt][pred], width=width)
                if conf_mat[gt][pred] != 0
                else " " * width
                for pred in classes
            )
            print(row_str)

        # Compute and print per-class accuracies
        print("\nPer-class Accuracies:")
        macro_acc = 0.0
        for cls in classes:
            if per_class_total[cls] > 0:
                acc = per_class_correct[cls] / per_class_total[cls]
            else:
                acc = 0.0
            macro_acc += acc
            print(f"{cls:_<{max(map(len, classes))}}{acc:_>9.1%} ({per_class_correct[cls]}/{per_class_total[cls]})")
        macro_acc /= len(classes) if classes else 1

        # Micro accuracy: overall correct predictions / total predictions
        micro_acc = total_correct / total_samples if total_samples > 0 else 0.0

        print(f"\nMicro Accuracy: {micro_acc:.2%} ({total_correct}/{total_samples})")
        print(f"Macro Accuracy: {macro_acc:.2%}")



    