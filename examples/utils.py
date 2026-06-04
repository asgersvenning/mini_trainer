#!/usr/bin/env python3
"""Shared utilities for example dataset construction."""

import concurrent.futures
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
import zipfile

from tqdm import tqdm


class CleanupOnFailure:
    """Context manager to cleanup registered directories/files if an exception occurs."""

    def __init__(self):
        self.paths_to_clean = []

    def register(self, path):
        if path not in self.paths_to_clean:
            self.paths_to_clean.append(path)

    def unregister(self, path):
        if path in self.paths_to_clean:
            self.paths_to_clean.remove(path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print("\nCleaning up partial/corrupted files due to interruption or failure...")
            for path in self.paths_to_clean:
                if os.path.exists(path):
                    try:
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        else:
                            os.remove(path)
                    except Exception as e:
                        print(f"Error deleting {path}: {e}")


def download_with_progress(url, dst, max_workers=8):
    print(f"Downloading {url} to {dst}...")

    # 1. HEAD request to check size and range support
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            total_size = int(resp.info().get("Content-Length", 0))
            accept_ranges = resp.info().get("Accept-Ranges") == "bytes"
    except Exception:
        total_size = 0
        accept_ranges = False

    # 2. Parallel download if range request is supported
    if accept_ranges and total_size > 10 * 1024 * 1024:
        print(f"Server supports range requests. Downloading in parallel using {max_workers} threads...")

        # Preallocate file
        with open(dst, "wb") as f:
            f.truncate(total_size)

        chunk_size = 32 * 1024 * 1024  # 32 MB chunks
        chunks = []
        start = 0
        while start < total_size:
            end = min(start + chunk_size - 1, total_size - 1)
            chunks.append((start, end))
            start += chunk_size

        with tqdm(total=total_size, unit="iB", unit_scale=True, desc=os.path.basename(dst)) as pbar:

            def download_chunk(start_pos, end_pos):
                req_chunk = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Range": f"bytes={start_pos}-{end_pos}",
                    },
                )
                # Retry up to 3 times on failure
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(req_chunk) as resp_chunk:
                            with open(dst, "r+b") as f:
                                f.seek(start_pos)
                                block_size = 1024 * 1024  # 1 MB blocks
                                while True:
                                    data = resp_chunk.read(block_size)
                                    if not data:
                                        break
                                    f.write(data)
                                    pbar.update(len(data))
                        return
                    except Exception as e:
                        if attempt == 2:
                            raise e
                        time.sleep(1.0)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(download_chunk, s, e) for s, e in chunks]
                concurrent.futures.wait(futures)
                # Check for exceptions
                for fut in futures:
                    if fut.exception():
                        raise fut.exception()
    else:
        # Fallback to single thread download
        print("Server does not support range requests or file is small. Downloading sequentially...")
        req_seq = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_seq) as response:
            total = int(response.info().get("Content-Length", 0))
            block_size = 1024 * 1024
            with tqdm(total=total, unit="iB", unit_scale=True, desc=os.path.basename(dst)) as pbar:
                with open(dst, "wb") as f:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        f.write(buffer)
                        pbar.update(len(buffer))

    print("Download complete.")


def extract_tar(tar_path, extract_path):
    print(f"Extracting {tar_path} to {extract_path}...")
    os.makedirs(extract_path, exist_ok=True)

    # Try using system tar command (much faster)
    if shutil.which("tar"):
        print("Using system native 'tar' for fast extraction...")
        try:
            subprocess.run(["tar", "-xf", tar_path, "-C", extract_path], check=True)
            print("Extraction complete.")
            return
        except subprocess.SubprocessError as e:
            print(f"Native 'tar' failed: {e}. Falling back to Python tarfile...")

    # Fallback to Python tarfile (with single-pass tqdm based on tar.next())
    with tarfile.open(tar_path, "r:gz") as tar:
        with tqdm(desc="Extracting", unit="file") as pbar:
            while True:
                member = tar.next()
                if member is None:
                    break
                tar.extract(member, path=extract_path, filter="data")
                pbar.update(1)
    print("Extraction complete.")


def extract_zip(zip_path, extract_path):
    print(f"Extracting {zip_path} to {extract_path}...")
    os.makedirs(extract_path, exist_ok=True)

    # Try using system unzip command
    if shutil.which("unzip"):
        print("Using system native 'unzip' for fast extraction...")
        try:
            subprocess.run(["unzip", "-q", zip_path, "-d", extract_path], check=True)
            print("Extraction complete.")
            return
        except subprocess.SubprocessError as e:
            print(f"Native 'unzip' failed: {e}. Falling back to Python zipfile...")

    # Fallback to Python zipfile
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        with tqdm(total=len(members), desc="Extracting", unit="file") as pbar:
            for member in members:
                zip_ref.extract(member, path=extract_path)
                pbar.update(1)
    print("Extraction complete.")
