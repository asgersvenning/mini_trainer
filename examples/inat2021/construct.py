#!/usr/bin/env python3
"""Script to download and construct the iNaturalist 2021 dataset (mini or full)."""

import argparse
import json
import os
import shutil
import sys

from tqdm import tqdm

from examples.utils import CleanupOnFailure, download_with_progress, extract_tar


def build_data_index(base_dir, train_dir_name, val_dir_name):
    train_dir = os.path.join(base_dir, train_dir_name)
    val_dir = os.path.join(base_dir, val_dir_name)
    
    # Check if directories exist
    if not os.path.exists(train_dir) and not os.path.exists(val_dir):
        print("Error: train or val directory does not exist.")
        return

    # Load existing taxonomy map if it exists
    taxonomy_map = {}
    taxonomy_map_path = os.path.join(base_dir, "taxonomy_map.json")
    if os.path.exists(taxonomy_map_path):
        try:
            with open(taxonomy_map_path) as f:
                taxonomy_map = json.load(f)
        except Exception:
            pass

    # Step 5/8: Scanning and mapping category directories
    print("\n[Step 5/8] Scanning and mapping category directories...")
    dir_to_species = {}
    scientific_names = set()
    
    dirs_to_scan = [d for d in [train_dir, val_dir] if os.path.exists(d)]
    
    for dir_path in dirs_to_scan:
        subdirs = sorted(os.listdir(dir_path))
        for name in tqdm(subdirs, desc=f"Scanning {os.path.basename(dir_path)}"):
            full_path = os.path.join(dir_path, name)
            if not os.path.isdir(full_path):
                continue
            parts = name.split("_")
            if len(parts) >= 8 and parts[0].isdigit():
                genus = parts[6]
                species = genus + " " + " ".join(parts[7:])
                dir_to_species[name] = species
                if species not in taxonomy_map:
                    scientific_names.add(species)
            else:
                if not name.isdigit():
                    dir_to_species[name] = name
                    if name not in taxonomy_map:
                        scientific_names.add(name)

    # Step 6/8: Resolving taxonomy via GBIF API
    print("\n[Step 6/8] Resolving taxonomy via GBIF API...")
    if scientific_names:
        print(f"Resolving taxonomy for {len(scientific_names)} species...")
        try:
            from mini_trainer.integrations import create_taxonomy, labels_from_taxonomy
            tax = create_taxonomy(list(scientific_names), levels="kingdom")
            labels = labels_from_taxonomy(tax)
            for species_name, tax_tuple in labels.items():
                taxonomy_map[species_name] = list(tax_tuple)
            # Save updated taxonomy map
            with open(taxonomy_map_path, "w") as f:
                json.dump(taxonomy_map, f, indent=2)
        except Exception as e:
            print(f"Error: Failed to resolve taxonomy via GBIF: {e}")
            print("Please make sure you have internet access and the GBIF API is available.")
            sys.exit(1)
    else:
        print("All categories already resolved.")

    # Map directory names to GBIF IDs
    dir_to_gbif_id = {}
    for name, species in dir_to_species.items():
        if species in taxonomy_map:
            gbif_id = taxonomy_map[species][0]
            dir_to_gbif_id[name] = gbif_id

    # Step 7/8: Renaming directories on disk
    print("\n[Step 7/8] Renaming directories on disk to GBIF IDs...")
    for dir_path in dirs_to_scan:
        subdirs = sorted(os.listdir(dir_path))
        for name in tqdm(subdirs, desc=f"Renaming {os.path.basename(dir_path)}"):
            full_path = os.path.join(dir_path, name)
            if not os.path.isdir(full_path):
                continue
            if name in dir_to_gbif_id:
                gbif_id = dir_to_gbif_id[name]
                new_path = os.path.join(dir_path, gbif_id)
                if not os.path.exists(new_path):
                    os.rename(full_path, new_path)
                elif full_path != new_path:
                    shutil.rmtree(full_path)

    # Build a lookup from category (GBIF key) to taxonomy list
    category_to_tax = {}
    for species, tax_list in taxonomy_map.items():
        if tax_list:
            category_to_tax[tax_list[0]] = tax_list

    # Step 8/8: Generating data_index.json
    print("\n[Step 8/8] Generating data_index.json...")
    paths = []
    splits = []
    labels = []
    
    img_exts = {".jpg", ".jpeg", ".png"}
    
    # Process train
    if os.path.exists(train_dir):
        categories = sorted(os.listdir(train_dir))
        for category in tqdm(categories, desc="Indexing train"):
            cat_dir = os.path.join(train_dir, category)
            if os.path.isdir(cat_dir):
                tax_list = category_to_tax.get(category)
                if tax_list is None:
                    tax_list = [category, category, "Unknown", "Unknown", "Unknown", "Unknown", "Unknown"]
                for f in sorted(os.listdir(cat_dir)):
                    if os.path.splitext(f)[1].lower() in img_exts:
                        paths.append(os.path.relpath(os.path.join(cat_dir, f), base_dir))
                        splits.append("train")
                        labels.append(tax_list)
                        
    # Process val
    if os.path.exists(val_dir):
        categories = sorted(os.listdir(val_dir))
        for category in tqdm(categories, desc="Indexing val"):
            cat_dir = os.path.join(val_dir, category)
            if os.path.isdir(cat_dir):
                tax_list = category_to_tax.get(category)
                if tax_list is None:
                    tax_list = [category, category, "Unknown", "Unknown", "Unknown", "Unknown", "Unknown"]
                for f in sorted(os.listdir(cat_dir)):
                    if os.path.splitext(f)[1].lower() in img_exts:
                        paths.append(os.path.relpath(os.path.join(cat_dir, f), base_dir))
                        splits.append("validation")
                        labels.append(tax_list)
                        
    index_data = {
        "path": paths,
        "split": splits,
        "label": labels
    }
    
    index_path = os.path.join(base_dir, "data_index.json")
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)
    print(f"data_index.json written to {index_path}")
    print(f"Total images: {len(paths)} (Train: {splits.count('train')}, Val: {splits.count('validation')})")


def main():
    parser = argparse.ArgumentParser(description="Download and format iNaturalist 2021 dataset.")
    parser.add_argument(
        "--type",
        choices=["mini", "full"],
        default="mini",
        help="Whether to download/construct the mini (50 images/species) or full dataset (default: mini)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Target root directory to construct the dataset (default: examples/inat2021/[type])",
    )
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_type = args.type
    
    if args.output_dir is None:
        base_dir = os.path.abspath(os.path.join(current_dir, target_type))
    else:
        base_dir = os.path.abspath(os.path.join(args.output_dir, target_type))
        
    os.makedirs(base_dir, exist_ok=True)
    
    # URLs
    urls = {
        "mini": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz",
        "full": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz",
        "val": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz"
    }
    
    train_dir_name = "train_mini" if target_type == "mini" else "train"
    val_dir_name = "val"
    
    sentinel_path = os.path.join(base_dir, ".complete")
    if os.path.exists(sentinel_path):
        print(f"Dataset already fully constructed at {base_dir}. Skipping download/construction.")
        return

    train_path = os.path.join(base_dir, train_dir_name)
    val_path = os.path.join(base_dir, val_dir_name)

    # Clean up any partial folders from previous runs
    for path in [train_path, val_path]:
        if os.path.exists(path):
            print(f"Removing partial/corrupt directory: {path}")
            shutil.rmtree(path)

    train_tar = os.path.join(base_dir, f"{train_dir_name}.tar.gz")
    val_tar = os.path.join(base_dir, "val.tar.gz")

    with CleanupOnFailure() as cleanup:
        cleanup.register(train_path)
        cleanup.register(val_path)

        # Download train
        print("[Step 1/8] Downloading train dataset...")
        if not os.path.exists(train_tar):
            download_with_progress(urls[target_type], train_tar)
        else:
            print(f"  (Using existing archive {train_tar})")
        print("[Step 2/8] Extracting train dataset...")
        extract_tar(train_tar, base_dir)
            
        # Download val
        print("[Step 3/8] Downloading validation dataset...")
        if not os.path.exists(val_tar):
            download_with_progress(urls["val"], val_tar)
        else:
            print(f"  (Using existing archive {val_tar})")
        print("[Step 4/8] Extracting validation dataset...")
        extract_tar(val_tar, base_dir)
            
        # Build data index (steps 5-8 are inside build_data_index)
        build_data_index(base_dir, train_dir_name, val_dir_name)

        # Success - clean up downloaded tar files
        for tar in [train_tar, val_tar]:
            if os.path.exists(tar):
                try:
                    os.remove(tar)
                except OSError:
                    pass

        # Write sentinel
        with open(sentinel_path, "w") as f:
            f.write("complete")

    print("\nDataset construction completed successfully!")


if __name__ == "__main__":
    main()
