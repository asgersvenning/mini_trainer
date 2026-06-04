#!/usr/bin/env python3
"""Script to download and construct the bird-species-dataset from Hugging Face."""

import argparse
import os
import shutil
import sys

from examples.utils import CleanupOnFailure, extract_tar

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: The 'huggingface_hub' package is required to download the dataset.")
    print("Please install it using: pip install huggingface_hub")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download and format bird species dataset.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Target directory to construct the dataset (default: directory of this script)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    splits = ["train", "valid", "test"]

    # Check if sentinel file exists
    sentinel_path = os.path.join(args.output_dir, ".complete")
    if os.path.exists(sentinel_path):
        print(f"Dataset already fully constructed at '{args.output_dir}'. Skipping download.")
        print_dataset_summary(args.output_dir, splits)
        return

    # If not complete, remove any existing directories to avoid partial/corrupted state
    for split in splits:
        split_dir = os.path.join(args.output_dir, split)
        if os.path.exists(split_dir):
            print(f"Removing partial/corrupt directory: {split_dir}")
            shutil.rmtree(split_dir)

    print(f"Downloading and extracting dataset to '{args.output_dir}'...")

    with CleanupOnFailure() as cleanup:
        for split in splits:
            split_dir = os.path.join(args.output_dir, split)
            cleanup.register(split_dir)

            filename = f"data/{split}.tar.gz"
            print(f"\nDownloading {filename} from Hugging Face...")
            try:
                tar_path = hf_hub_download(
                    repo_id="chriamue/bird-species-dataset",
                    repo_type="dataset",
                    filename=filename,
                )
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                sys.exit(1)

            print(f"Extracting {filename}...")
            try:
                extract_tar(tar_path, args.output_dir)
            except Exception as e:
                print(f"Error extracting {filename}: {e}")
                sys.exit(1)

        # Write sentinel file on success
        with open(sentinel_path, "w") as f:
            f.write("complete")

    print("\nDataset construction completed successfully!")
    print_dataset_summary(args.output_dir, splits)


def print_dataset_summary(base_dir, splits):
    print("\nDataset Summary:")
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            print(f"  {split}: Not found")
            continue
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        num_images = sum(
            len(files)
            for _, _, files in os.walk(split_dir)
            if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files)
        )
        print(f"  {split}: {len(classes)} classes, {num_images} images")


if __name__ == "__main__":
    main()
