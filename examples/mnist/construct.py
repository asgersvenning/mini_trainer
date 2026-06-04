#!/usr/bin/env python3
"""Script to download and construct the MNIST dataset for mini_trainer."""

import argparse
import array
import functools
import gzip
import operator
import os
import struct
import urllib.parse

import numpy as np
from PIL import Image
from tqdm import tqdm

from examples.utils import download_with_progress


class IdxDecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""

    pass


def parse_idx(fd):
    """Parse an IDX file, and return it as a numpy array."""
    DATA_TYPES = {
        0x08: "B",  # unsigned byte
        0x09: "b",  # signed byte
        0x0B: "h",  # short (2 bytes)
        0x0C: "i",  # int (4 bytes)
        0x0D: "f",  # float (4 bytes)
        0x0E: "d",  # double (8 bytes)
    }

    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError("Invalid IDX file, file empty or does not contain a full header.")

    zeros, data_type, num_dimensions = struct.unpack(">HBB", header)

    if zeros != 0:
        raise IdxDecodeError(f"Invalid IDX file, file must start with two zero bytes. Found 0x{zeros:02x}")

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError(f"Unknown data type 0x{data_type:02x} in IDX file")

    dimension_sizes = struct.unpack(">" + "I" * num_dimensions, fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError(f"IDX file has wrong number of items. Expected: {expected_items}. Found: {len(data)}")

    return np.array(data).reshape(dimension_sizes)


def prepare_mnist_for_minitrainer(images, labels, base_dir, split, max_count=500):
    """Extract individual digits as PNG images organized by labels."""
    label_count = {}
    for image, label in tqdm(zip(images, labels), desc=f"Writing {split} images to disk", total=len(images)):
        label_count[label] = label_count.get(label, -1) + 1
        if max_count is not None and label_count[label] >= max_count:
            continue
        image_path = os.path.join(base_dir, split, f"{label}", f"{label_count[label]}.png")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        Image.fromarray(np.broadcast_to(image[:, :, np.newaxis], (*image.shape, 3))).save(image_path)


def main():
    parser = argparse.ArgumentParser(description="Download and format MNIST dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Target directory to construct the dataset (default: directory of this script)",
    )
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_dir is None:
        base_dir = current_dir
    else:
        base_dir = os.path.abspath(args.output_dir)

    train_path = os.path.join(base_dir, "train")
    test_path = os.path.join(base_dir, "test")

    # Check if directories already exist
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Dataset already exists at {base_dir}. Skipping download.")
        return

    os.makedirs(base_dir, exist_ok=True)

    datasets_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    downloaded_paths = {}

    # Step 1: Downloading
    print("[Step 1/2] Downloading MNIST gz files...")
    for key, filename in files.items():
        dst_path = os.path.join(base_dir, filename)
        url = urllib.parse.urljoin(datasets_url, filename)
        if not os.path.exists(dst_path):
            download_with_progress(url, dst_path)
        else:
            print(f"  (Using existing archive {filename})")
        downloaded_paths[key] = dst_path

    # Step 2: Parsing and extracting PNG images
    print("\n[Step 2/2] Parsing IDX files and writing PNG images...")
    
    # Train
    with gzip.open(downloaded_paths["train_labels"], "rb") as zf:
        train_labels = parse_idx(zf)
    with gzip.open(downloaded_paths["train_images"], "rb") as zf:
        train_images = parse_idx(zf)
    prepare_mnist_for_minitrainer(train_images, train_labels, base_dir, "train", max_count=500)

    # Test
    with gzip.open(downloaded_paths["test_labels"], "rb") as zf:
        test_labels = parse_idx(zf)
    with gzip.open(downloaded_paths["test_images"], "rb") as zf:
        test_images = parse_idx(zf)
    prepare_mnist_for_minitrainer(test_images, test_labels, base_dir, "test", max_count=500)

    # Cleanup downloaded gz files
    for path in downloaded_paths.values():
        if os.path.exists(path):
            os.remove(path)

    print("\nDataset construction completed successfully!")


if __name__ == "__main__":
    main()
