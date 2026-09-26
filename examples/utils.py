#!/usr/bin/env python3
"""Shared utilities for example dataset construction."""

import concurrent.futures
import os
import shutil
import subprocess
import tarfile
import tempfile
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
        if exc_type is None or not self.paths_to_clean:
            return
        print("\nCleaning up partial/corrupted files due to interruption or failure...")

        def report_error(function, path, error):
            print(f"\nError deleting {path}: {error}")

        for path in tqdm(self.paths_to_clean, desc="Cleaning up", unit="path"):
            try:
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path, onexc=report_error)
                elif os.path.lexists(path):
                    os.remove(path)
            except OSError as error:
                report_error(None, path, error)


def download_with_progress(url, dst, max_workers=8):
    """Publish a downloaded archive only after its transfer succeeds."""
    print(f"Downloading {url} to {dst}...")
    destination = os.path.abspath(dst)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(destination), prefix=".download-") as staging:
        partial = os.path.join(staging, os.path.basename(destination))
        _download_to_file(url, partial, max_workers)
        os.replace(partial, destination)
    print("Download complete.")


def _download_to_file(url, dst, max_workers):
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


def _extract_native(command, success_codes=(0,)):
    """Run an available extractor; let the caller use Python on failure."""
    if not shutil.which(command[0]):
        return False
    print(f"Using system native '{command[0]}' with real-time progress...")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        with tqdm(desc="Extracting (Native)", unit="file") as pbar:
            for line in process.stdout:
                if line.strip() and not line.startswith("Archive:"):
                    pbar.update(1)
        process.wait()
        if process.returncode not in success_codes:
            raise subprocess.SubprocessError(f"{command[0]} failed with exit code {process.returncode}")
    except Exception as error:
        print(f"Native '{command[0]}' failed: {error}. Falling back to Python...")
        return False
    return True


def extract_tar(tar_path, extract_path):
    """Extract TAR using native tools when available, otherwise Python's data filter."""
    print(f"Extracting {tar_path} to {extract_path}...")
    os.makedirs(extract_path, exist_ok=True)
    if not _extract_native(["tar", "-xvf", tar_path, "-C", extract_path]):
        print("Using pure Python tarfile fallback...")
        with tarfile.open(tar_path, "r") as tar:
            for member in tqdm(tar, desc="Extracting (Python)", unit="file"):
                tar.extract(member, path=extract_path, filter="data")
    print("Extraction complete.")


def extract_zip(zip_path, extract_path):
    """Extract ZIP using native tools when available, with a Python fallback."""
    print(f"Extracting {zip_path} to {extract_path}...")
    os.makedirs(extract_path, exist_ok=True)
    # unzip's status 1 is a warning; -o permits overwriting without a prompt.
    if not _extract_native(["unzip", "-o", zip_path, "-d", extract_path], success_codes=(0, 1)):
        print("Using pure Python zipfile fallback...")
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in tqdm(archive.infolist(), desc="Extracting (Python)", unit="file"):
                archive.extract(member, path=extract_path)
    print("Extraction complete.")
