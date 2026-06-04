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
        if exc_type is not None and self.paths_to_clean:
            print("\nCleaning up partial/corrupted files due to interruption or failure...")

            # Use a running count since we don't know exactly how many files made it to disk
            with tqdm(desc="Cleaning up", unit="item") as pbar:
                for path in self.paths_to_clean:
                    if not os.path.exists(path):
                        continue

                    if os.path.isdir(path):
                        # Walk the directory bottom-up so empty subdirectories can be removed
                        for root, dirs, files in os.walk(path, topdown=False):
                            # 1. Delete all files in the current folder
                            for name in files:
                                try:
                                    os.remove(os.path.join(root, name))
                                    pbar.update(1)
                                except OSError as e:
                                    print(f"\nError deleting file {name}: {e}")

                            # 2. Delete all now-empty subdirectories in the current folder
                            for name in dirs:
                                try:
                                    os.rmdir(os.path.join(root, name))
                                    pbar.update(1)
                                except OSError as e:
                                    print(f"\nError deleting directory {name}: {e}")

                        # 3. Finally, remove the top-level directory itself
                        try:
                            os.rmdir(path)
                            pbar.update(1)
                        except OSError as e:
                            print(f"\nError deleting root directory {path}: {e}")
                    else:
                        # Standard single-file deletion
                        try:
                            os.remove(path)
                            pbar.update(1)
                        except OSError as e:
                            print(f"\nError deleting file {path}: {e}")


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
    """Extracts a tar archive using native tools if available, with a Python fallback."""
    print(f"Extracting {tar_path} to {extract_path}...")
    os.makedirs(extract_path, exist_ok=True)

    # 1. Try using system native 'tar' command with real-time polling
    if shutil.which("tar"):
        print("Using system native 'tar' with real-time progress...")
        try:
            # -v forces verbose output (one line per file extracted)
            process = subprocess.Popen(
                ["tar", "-xvf", tar_path, "-C", extract_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )

            # Track progress using a running count (tar doesn't know total files upfront)
            with tqdm(desc="Extracting (Native)", unit="file") as pbar:
                for line in process.stdout:
                    pbar.update(1)

            process.wait()
            if process.returncode != 0:
                raise subprocess.SubprocessError(f"Tar failed with exit code {process.returncode}")

            print("Extraction complete.")
            return

        except Exception as e:
            print(f"Native 'tar' failed: {e}. Falling back to Python tarfile...")

    # 2. Fallback to pure Python with iterative progress
    print("Using pure Python tarfile fallback...")
    # Opening with "r" allows Python to transparently auto-detect gz, bz2, or xz compression
    with tarfile.open(tar_path, "r") as tar:
        with tqdm(desc="Extracting (Python)", unit="file") as pbar:
            for member in tar:
                # filter="data" prevents directory traversal attacks (Requires Python 3.12+)
                # Note: If you are using Python 3.11 or older, remove the filter="data" argument
                tar.extract(member, path=extract_path, filter="data")
                pbar.update(1)

    print("Extraction complete.")


def extract_zip(zip_path, extract_path):
    """Extracts a zip archive using native tools if available, with a Python fallback."""
    print(f"Extracting {zip_path} to {extract_path}...")
    os.makedirs(extract_path, exist_ok=True)

    # 1. Try using system native 'unzip' command with real-time polling
    if shutil.which("unzip"):
        print("Using system native 'unzip' with real-time progress...")
        try:
            # -o : overwrite existing files without prompting (prevents hanging)
            # -d : specify destination directory
            process = subprocess.Popen(
                ["unzip", "-o", zip_path, "-d", extract_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )

            with tqdm(desc="Extracting (Native)", unit="file") as pbar:
                for line in process.stdout:
                    # Filter out empty lines or the initial "Archive:" header
                    if line.strip() and not line.startswith("Archive:"):
                        pbar.update(1)

            process.wait()
            # unzip returns 0 for success, 1 for non-fatal warnings (like an empty zip)
            if process.returncode not in (0, 1):
                raise subprocess.SubprocessError(f"Unzip failed with exit code {process.returncode}")

            print("Extraction complete.")
            return

        except Exception as e:
            print(f"Native 'unzip' failed: {e}. Falling back to Python zipfile...")

    # 2. Fallback to pure Python with total % progress
    print("Using pure Python zipfile fallback...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.infolist()
        # Because zips have a central directory, we can provide a 'total' for a 0-100% bar
        with tqdm(total=len(members), desc="Extracting (Python)", unit="file") as pbar:
            for member in members:
                zip_ref.extract(member, path=extract_path)
                pbar.update(1)

    print("Extraction complete.")
