#!/usr/bin/env python3
"""Script to download and construct the Blair dataset."""

import argparse
import os
import shutil

from examples.utils import download_with_progress, extract_zip


def main():
    parser = argparse.ArgumentParser(description="Download and format Blair dataset.")
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

    url = "https://github.com/Jarrett-Blair/Intro-to-CV-for-Ecologists/raw/refs/heads/main/Data/Images.zip"
    zip_path = os.path.join(base_dir, "Images.zip")

    # Step 1: Downloading
    print("[Step 1/3] Downloading Blair dataset Images.zip...")
    if not os.path.exists(zip_path):
        download_with_progress(url, zip_path)
    else:
        print("  (Using existing archive Images.zip)")

    # Step 2: Extracting
    print("\n[Step 2/3] Extracting Images.zip...")
    extract_zip(zip_path, base_dir)

    # Step 3: Formatting directories
    print("\n[Step 3/3] Formatting directories...")
    images_dir = os.path.join(base_dir, "Images")
    
    # Move training and testing folders
    shutil.move(os.path.join(images_dir, "training"), train_path)
    shutil.move(os.path.join(images_dir, "testing"), test_path)

    # Cleanup
    os.remove(zip_path)
    shutil.rmtree(images_dir)

    print("\nDataset construction completed successfully!")


if __name__ == "__main__":
    main()
