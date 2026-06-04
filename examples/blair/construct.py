#!/usr/bin/env python3
"""Script to download and construct the Blair dataset."""

import argparse
import os
import shutil

from examples.utils import CleanupOnFailure, download_with_progress, extract_zip


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

    # Check if sentinel file exists
    sentinel_path = os.path.join(base_dir, ".complete")
    if os.path.exists(sentinel_path):
        print(f"Dataset already fully constructed at {base_dir}. Skipping download.")
        return

    # Clean up any partial directories from previous runs
    for path in [train_path, test_path, os.path.join(base_dir, "Images")]:
        if os.path.exists(path):
            print(f"Removing partial/corrupt path: {path}")
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    os.makedirs(base_dir, exist_ok=True)

    url = "https://github.com/Jarrett-Blair/Intro-to-CV-for-Ecologists/raw/refs/heads/main/Data/Images.zip"
    zip_path = os.path.join(base_dir, "Images.zip")

    with CleanupOnFailure() as cleanup:
        cleanup.register(train_path)
        cleanup.register(test_path)
        cleanup.register(os.path.join(base_dir, "Images"))

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

        # Cleanup Images directory (which should be empty now)
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)

        # Cleanup zip file on success
        if os.path.exists(zip_path):
            os.remove(zip_path)

        # Write sentinel
        with open(sentinel_path, "w") as f:
            f.write("complete")

    print("\nDataset construction completed successfully!")


if __name__ == "__main__":
    main()
