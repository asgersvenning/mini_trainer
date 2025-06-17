import os
import re
from argparse import ArgumentParser
from glob import iglob

import torch
from torchvision.io import ImageReadMode, write_jpeg
from tqdm.contrib.concurrent import process_map

from mini_trainer.builders import make_read_and_resize_fn
from mini_trainer.utils import make_convert_dtype

if __name__ == "__main__":
    parser = ArgumentParser(prog = "resizer")
    parser.add_argument(
        "-i", "--input_dir", type=str, required=True,
        help="Root directory containing images (arbitrarily nested)."
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True,
        help="Root output directory. Images will be stored in the same relative path to the root."
    )
    parser.add_argument(
        "-s", "--size", type=int, default=256, required=False,
        help="Size of the resized image."
    )
    parser.add_argument(
        "-p", "--image_pattern", type=str, default=r"\.(jpe{0,1}g|png)$",
        help="Pattern which image file names must match."
    )
    args = parser.parse_args()

    reader = make_read_and_resize_fn(ImageReadMode.RGB, (args.size, args.size), torch.device("cpu"), torch.float16)
    convert2uint8 = make_convert_dtype(torch.uint8)
    def rewrite_image(src : str, dst : str):
        if os.path.exists(dst):
            return
        write_jpeg(convert2uint8(reader(src)), dst, 95)
    def proc_one(x):
        os.makedirs(os.path.dirname(dst := os.path.join(args.output_dir, os.path.relpath(x, args.input_dir))), exist_ok=True) is None and rewrite_image(x, dst)

    pattern = re.compile(args.image_pattern, re.IGNORECASE)
    imgs = list(filter(lambda x : bool(re.search(pattern, x)), iglob(os.path.join(args.input_dir, "**", "*"))))
    process_map(
        proc_one, 
        imgs,
        max_workers = min(8, max(1, os.cpu_count() // 2)),
        chunksize = 32
    )