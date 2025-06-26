import os
import re
import time
import shutil
from uuid import uuid4
from tempfile import mkdtemp
from functools import partial

import torch
import torchvision
from torchvision.io import ImageReadMode, write_jpeg
import torchvision.transforms.functional as TF

from PIL import Image
from tqdm.contrib.concurrent import process_map

from mini_trainer.builders import make_read_and_resize_fn
from mini_trainer.utils import make_convert_dtype


def find_n_images(root: str, n: int, pattern=r"\.(jpe?g|png)$") -> list[str]:
    pat = re.compile(pattern, re.IGNORECASE)
    found = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if pat.search(f):
                found.append(os.path.join(dirpath, f))
                if len(found) >= n:
                    return found
    return found

def benchmark_read(path):
    with open(path, "rb") as f:
        return f.read()

def benchmark_write(path, data):
    with open(path, "wb") as f:
        f.write(data)

def rewrite_image_pillow(src: str, dst: str | None=None, dst_dir: str | None=None, size: int = 256):
    if dst is None:
        if dst_dir is not None:
            dst = os.path.join(dst_dir, str(uuid4()) + ".jpg")
    Image.open(src).convert("RGB").resize((size, size), Image.Resampling.NEAREST).save(dst, "JPEG", quality=95)

def rewrite_image(src: str, dst: str, reader, convert2uint8):
    write_jpeg(convert2uint8(reader(src)), dst, 95)

def rewrite_image_torch_uint8(src: str, dst: str, reader):
    img = reader(src)
    write_jpeg(img, dst, 95)

# === Setup ===
root = "/work/eu_lepi_v1/images"
n = 2500
size = 256
batch_size = 32
tmp_dir = mkdtemp()
print(f"Using temporary output dir: {tmp_dir}")

images = find_n_images(root, n)
if len(images) != n:
    raise RuntimeError(f"Found {len(images)} images but expected {n}.")
print("Benchmarking on", len(images), "images in", root)

# === Benchmark: Raw Read ===
start = time.time()
raw_data = [benchmark_read(f) for f in images]
end = time.time()
print(f"Raw reads/sec: {len(raw_data) / (end - start):.1f}")

# === Benchmark: Raw Write ===
start = time.time()
for i, data in enumerate(raw_data):
    out_path = os.path.join(tmp_dir, f"write_{i}.jpg")
    benchmark_write(out_path, data)
end = time.time()
print(f"Raw writes/sec: {len(raw_data) / (end - start):.1f}")

# === Benchmark: Pillow (single-threaded) ===
start = time.time()
for i, f in enumerate(images):
    dst = os.path.join(tmp_dir, f"pillow_{i}.jpg")
    rewrite_image_pillow(f, dst, size)
end = time.time()
print(f"Pillow resize+write/sec: {n / (end - start):.1f}")

# === Benchmark: Original Torch ===
reader = make_read_and_resize_fn(
    (size, size),
    torch.device("cpu"),
    torch.float16
)
convert2uint8 = make_convert_dtype(torch.uint8)

start = time.time()
for i, f in enumerate(images):
    dst = os.path.join(tmp_dir, f"torch_orig_{i}.jpg")
    rewrite_image(f, dst, reader, convert2uint8)
end = time.time()
print(f"Original torch resize+write/sec: {n / (end - start):.1f}")

# === Benchmark: Torch uint8 direct ===
reader_uint8 = make_read_and_resize_fn(
    (size, size),
    torch.device("cpu"),
    torch.uint8
)
start = time.time()
for i, f in enumerate(images):
    dst = os.path.join(tmp_dir, f"torch_uint8_{i}.jpg")
    rewrite_image_torch_uint8(f, dst, reader_uint8)
end = time.time()
print(f"Torch (uint8 direct) resize+write/sec: {n / (end - start):.1f}")

# === Benchmark: Multiprocessing Pillow ===
start = time.time()
process_map(
    partial(rewrite_image_pillow, dst_dir=tmp_dir, size=size),
    images,
    max_workers=min(8, os.cpu_count() // 2),
    chunksize=32
)
end = time.time()
print(f"Pillow multiprocessing resize+write/sec: {n / (end - start):.1f}")

# === Cleanup ===
shutil.rmtree(tmp_dir)
