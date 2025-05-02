import os
import shutil

import numpy as np
import torch
from hierarchical.base.integration import (hierarchical_dataloader_builder,
                                           hierarchical_model_builder,
                                           hierarchical_parse_class_index)
from hierarchical.base.loss import MultiLevelCrossEntropyLoss
from hierarchical.guillaume.hierarchical import (
    DEFAULT_HIERARCHY_LEVELS, HierarchicalPathParser,
    hierarchical_create_data_index)
from hierarchical.guillaume.setup import erda_to_combinations
from PIL import Image, ImageOps
from pyremotedata.implicit_mount import IOHandler, RemotePathIterator
from torchvision.io import ImageReadMode, write_jpeg
from torchvision.transforms.v2.functional import resize, to_dtype_image
from tqdm import tqdm as TQDM

from mini_trainer.predict import main as mt_predict
from mini_trainer.train import main as mt_train

SHARE_LINK = "JlICFo26h8"
IMAGE_DIR = os.path.abspath(os.path.join("hierarchical", "gmo_traits"))

def _cast_to_uint(arr : np.ndarray, dtype : np.dtype):
    src_dtype = arr.dtype
    if src_dtype == dtype:
        return arr
    dst_max_val = np.iinfo(dtype).max
    if src_dtype.kind in ("u","i"):
        src_max_val = np.iinfo(src_dtype).max
        arr = (arr.astype(np.float32) * dst_max_val / src_max_val).round()
    elif src_dtype.kind == "f":
        src_min, src_max = np.min(arr), np.max(arr)
        if 0. <= src_min and src_max <= 1.:
            arr = (arr * dst_max_val).round()
        elif 0. <= src_min and src_max <= 255.:
            pass
        else:
            raise RuntimeError(f'Invalid image data range, found [{src_min}; {src_min}] but expected [0, 1] or [0, 255].')
    else:
        raise RuntimeError(f"Unsupported dtype {src_dtype}")
    return np.clip(arr, 0, dst_max_val).astype(dtype)

def read_image(image : str):
    img = Image.open(image, "r")
    img = ImageOps.exif_transpose(img) 
    img = img.convert("RGB")
    arr = np.asarray(img).copy()
    arr = _cast_to_uint(arr, np.uint8)
    tensor = torch.from_numpy(arr)
    match ndim := tensor.ndim:
        case 3:
            pass
        case 2:
            return tensor.unsqueeze(0).repeat(3, 1, 1)
        case _:
            raise RuntimeError(f'Invalid number of dimensions, found {ndim} when reading {image}, expected 3 (RGB) or 2 (grayscale).')
    tensor = tensor.permute(2, 0, 1)
    match n_channels := tensor.shape[0]:
        case 3:
            return tensor
        case 4:
            return tensor[:3]
        case _:
            raise RuntimeError(f'Invalid number of channels, found {n_channels} when reading {image}, expected 3 (RGB) or 4 (RGBA).')

def create_dst(src : str, dst_dir : str):
    dst = os.path.join(dst_dir, *src.split("/"))
    dst, _ = os.path.splitext(dst)
    return f'{dst}.jpg'

def image_already_saved(src : str, dst_dir : str):
    dst = create_dst(src, dst_dir)
    dst_frame0 = "{}_0{}".format(*os.path.splitext(dst))
    return os.path.exists(dst) or os.path.exists(dst_frame0)

def standardized_save(image : torch.Tensor, dst : str):
    image = to_dtype_image(image, torch.uint8)
    write_jpeg(
        resize(
            image, min(511, min(image.shape[1:])),
            max_size=512
        ),
        filename=dst,
        quality=90
    )

if __name__ == "__main__":
    # # os.system(f'sshpass -p "{SHARE_LINK}" sftp -P 2222 -o StrictHostKeyChecking=no {SHARE_LINK}@io.erda.au.dk')
    if os.path.isdir(IMAGE_DIR):
        shutil.rmtree(IMAGE_DIR)
    if os.path.exists(IMAGE_DIR):
        raise RuntimeError(f'Error while cleaning the image directory: {IMAGE_DIR}')
    os.makedirs(IMAGE_DIR, exist_ok=True)
    with IOHandler(user=SHARE_LINK, password=SHARE_LINK, remote="io.erda.au.dk", verbose=False) as io:
        ri = RemotePathIterator(io, max_queued_batches=5, batch_parallel=16)
        ri.subset([i for i, rp in enumerate(ri.remote_paths) if not image_already_saved(rp, IMAGE_DIR)])
        for lp, rp in TQDM(ri, desc="Downloading files...", smoothing=0.01):
            dst = create_dst(rp, IMAGE_DIR)
            if not os.path.exists(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst))
            try:
                image = read_image(lp, ImageReadMode.RGB)
                standardized_save(image, dst)
            except Exception as e:
                e.add_note(f'Error while trying to resize and/or save {rp} to {dst}.')
                raise e
    
    print(f'Running quality control:')
    mt_predict(
        model="efficientnet_v2_s", 
        weights="quality/efficientnet_v2_s_full_e15.pt",
        input=IMAGE_DIR,
        output="hierarchical/quality_control.json",
        class_index="quality/class_index.json"
    )
    
    hierarchical_parse_class_index(
        path="hierarchical/gmo_traits_class_index.json",
        dir=SHARE_LINK,
        dir2comb_fn=erda_to_combinations 
    )

    hierarchical_create_data_index(
        path="hierarchical/quality_control.json", 
        outpath="hierarchical/gmo_traits_data_index.json",
        split=(0.9, 0.05, 0.05), 
        max_per_cls=25,
        class_index="hierarchical/gmo_traits_class_index.json",
        levels=DEFAULT_HIERARCHY_LEVELS
    )

    mt_train(
        input="hierarchical/train",
        output="hierarchical",
        model="efficientnet_b0",
        class_index="hierarchical/gmo_traits_class_index.json",
        data_index="hierarchical/gmo_traits_data_index.json",
        name="test",
        # weights="hierarchical/efficientnet_b0_full_e2.pt",
        batch_size=16,
        learning_rate=0.001,
        warmup_epochs=0.1,
        epochs=10,
        dtype="float16",
        # device="cpu",
        spec_model_dataloader=hierarchical_parse_class_index,
        dataloader_builder=hierarchical_dataloader_builder,
        dataloader_builder_kwargs={
            "resize_size": 256, 
            # "train_proportion": 0.9,
            "path2cls2idx_builder" : HierarchicalPathParser,
            "path2cls2idx_builder_kwargs" : {
                "class_index" : "hierarchical/gmo_traits_class_index.json",
                "levels" : DEFAULT_HIERARCHY_LEVELS,
                "as_tensor" : True
            }
        },
        model_builder=hierarchical_model_builder,
        criterion_builder=MultiLevelCrossEntropyLoss,
        criterion_kwargs={"label_smoothing" : 0.01, "weights" : [0.65, 0.25, 0.1]} #  non-hierarchical: [1, 0, 0] | different hierarchical weightings: [0.1, 0.25, 0.65], [0.65, 0.25, 0.1], [1, 1, 1]
    )


