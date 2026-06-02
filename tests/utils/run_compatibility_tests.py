#!/usr/bin/env python
"""Compatibility testing utility for model backbones in mini_trainer.

Runs a fast training integration test (1 epoch on a small synthetic dataset)
across all backbones to verify compatibility, manages the blacklist automatically,
and stores results in a persistent JSON cache.
"""

import argparse
import json
import os
import shutil
import sys
import traceback
from typing import Any

import torch
import torchvision.transforms as tt
from torch.utils.data import DataLoader, TensorDataset

import mini_trainer.modeling.architectures.core as mt_core
from mini_trainer.builders import BaseBuilder
from mini_trainer.logging import configure_loggers
from mini_trainer.modeling.architectures.load import list_supported_backbones, save_blacklist
from mini_trainer.train import main as mt_main

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "compatibility_results.json")
IMG_SIZE = 128


class TestSyntheticBuilder(BaseBuilder):
    """Builder for a small synthetic dataset used in compatibility tests."""

    @staticmethod
    def class_spec(*args, **kwargs):
        return {"num_classes": 2, "cls2idx": {"class_a": 0, "class_b": 1}}

    @staticmethod
    def build_dataloader(batch_size, device, dtype, **kwargs):
        n = 8
        c, h, w = 3, IMG_SIZE, IMG_SIZE
        data_0 = torch.randn(n // 2, c, h, w)
        data_1 = torch.randn(n // 2, c, h, w) + 1.0
        data = torch.cat([data_0, data_1])
        labels = torch.cat([torch.zeros(n // 2, dtype=torch.long), torch.ones(n // 2, dtype=torch.long)])

        dataset = TensorDataset(data, labels)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return labels.numpy(), train_loader, val_loader

    @staticmethod
    def build_augmentation(dtype):
        return tt.Compose([])

    @staticmethod
    def build_regularizer(*args, **kwargs):
        return lambda x: torch.tensor(0.0)


def load_results() -> dict[str, Any]:
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_results(results: dict[str, Any]) -> None:
    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error saving results: {e}", file=sys.stderr)


def run_integration_test(model_name: str, online: bool, temp_dir: str) -> tuple[str, str]:
    """Runs a single epoch of training using the actual main training loop."""
    output_dir = os.path.join(temp_dir, "output")
    input_dir = os.path.join(temp_dir, "data")
    os.makedirs(os.path.join(input_dir, "class_a"), exist_ok=True)
    os.makedirs(os.path.join(input_dir, "class_b"), exist_ok=True)

    # Configure model_args
    model_args: dict[str, Any] = {"pretrained": False}
    if not online:
        model_args["local_files_only"] = True

    try:
        args = {
            "input": input_dir,
            "output": output_dir,
            "epochs": 1,
            "size": IMG_SIZE,
            "device": "cuda",
            "name": "compat_run",
            "builder": TestSyntheticBuilder,
            "model_builder_kwargs": {
                "model_type": model_name,
                "fine_tune": False,
                "hidden": False,
                "droprate": 0.1,
                "normalized": True,
                "model_args": model_args,
            },
            "dataloader_builder_kwargs": {"batch_size": 4, "train_proportion": 0.9, "resample": False, "cache": "cuda"},
            "logger_builder_kwargs": {
                "logger_cls": configure_loggers(),
            },
            "ema": False,
            "seed": 42,
        }

        # Redirect standard output/error to silence train prints
        sys.stdout.flush()
        sys.stderr.flush()

        mt_main(**args)

        return "PASS", "Training integration test completed successfully."
    except Exception as e:
        err_msg = str(e)
        tb = traceback.format_exc()

        # Detect offline/download errors
        is_offline_err = any(
            x in err_msg.lower() or x in tb.lower()
            for x in ["offline", "local_files_only", "connectionerror", "requestsexception", "huggingface_hub.utils"]
        )
        if is_offline_err and not online:
            return "OFFLINE_SKIPPED", f"Skipped (Offline cache miss): {err_msg}"

        return "FAIL", f"{type(e).__name__}: {err_msg}\n{tb}"
    finally:
        # Clean up temp run directory after test
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass


def main_cli():
    parser = argparse.ArgumentParser(description="Run backbone compatibility integration tests and sync blacklist.")
    parser.add_argument(
        "--backend",
        choices=["torchvision", "timm", "transformers", "bioclip", "all"],
        default="all",
        help="Filter backbones by backend.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of models tested.")
    parser.add_argument("--model", type=str, default=None, help="Test a single specific model name.")
    parser.add_argument("--online", action="store_true", help="Allow online downloads during tests.")
    parser.add_argument("--clear", action="store_true", help="Clear previous test results cache before running.")
    args = parser.parse_args()

    # 1. Results caching initialization
    if args.clear:
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)
        print("Cleared compatibility results cache.")

    results = load_results()

    # 2. Discover models before clearing the blacklist, so BackboneInfo has the original status
    all_backbones = list_supported_backbones()

    # Filter backbones to test
    to_test = []
    for b in all_backbones:
        if args.backend != "all" and b.backend != args.backend:
            continue
        if args.model and b.model != args.model:
            continue
        if not b.availability:
            continue
        to_test.append(b)

    if args.limit:
        to_test = to_test[: args.limit]

    if not to_test:
        print("No models matching the selection criteria found to test.")
        return

    # Keep a deep copy of original blacklist to compute updates later
    original_blacklist = {k: list(v) for k, v in mt_core.BLACKLIST.items()}

    # Clear blacklist in-place in memory so we can load/import/test all models
    for k in mt_core.BLACKLIST:
        mt_core.BLACKLIST[k] = []

    print(f"Starting compatibility integration tests for {len(to_test)} models...")
    print("Press Ctrl+C to stop. Progress will be saved and the blacklist will be synchronized.")
    print("-" * 60)

    # Base temp dir in workspace
    workspace_temp = os.path.join(os.path.dirname(__file__), ".temp_compat_runs")
    os.makedirs(workspace_temp, exist_ok=True)

    passed = 0
    failed = 0
    skipped = 0

    run_outcomes = {}

    try:
        for idx, b in enumerate(to_test):
            # Check cache
            cached = results.get(b.model)
            if cached and cached.get("status") in ("PASS", "FAIL") and not args.model:
                status = cached.get("status")
                message = cached.get("message", "")
                print(f"[{idx + 1}/{len(to_test)}] {b.model} ({b.backend}) -> {status} (cached)")
                if status == "PASS":
                    passed += 1
                else:
                    failed += 1
                run_outcomes[b.model] = status
                continue

            print(f"[{idx + 1}/{len(to_test)}] Testing {b.model} ({b.backend})... ", end="", flush=True)

            temp_run_dir = os.path.join(workspace_temp, f"run_{idx}")
            status, message = run_integration_test(b.model, args.online, temp_run_dir)

            # Save immediately
            results[b.model] = {"backend": b.backend, "status": status, "message": message}
            save_results(results)

            run_outcomes[b.model] = status

            if status == "PASS":
                print("PASS")
                passed += 1
            elif status == "OFFLINE_SKIPPED":
                print("SKIPPED (Offline)")
                skipped += 1
            else:
                print("FAIL")
                print(f"  Error details: {message.splitlines()[0] if message else 'Unknown error'}")
                failed += 1
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user. Saving current progress...")
    finally:
        # Clean up global temp dir
        if os.path.exists(workspace_temp):
            try:
                shutil.rmtree(workspace_temp)
            except Exception:
                pass

        # Update blacklist on disk/core using test results
        new_blacklist = {k: list(v) for k, v in original_blacklist.items()}
        added_to_blacklist = []
        removed_from_blacklist = []

        for b in to_test:
            status = run_outcomes.get(b.model)
            if not status or status == "OFFLINE_SKIPPED":
                continue

            clean_name = b.model
            if clean_name.startswith("timm:"):
                clean_name = clean_name.split(":", 1)[1]
            elif clean_name.startswith("hf-hub:"):
                clean_name = clean_name.split(":", 1)[1]

            backend_list = new_blacklist.get(b.backend, [])

            if status == "FAIL":
                if clean_name not in backend_list:
                    backend_list.append(clean_name)
                    added_to_blacklist.append(b.model)
            elif status == "PASS":
                if clean_name in backend_list:
                    backend_list.remove(clean_name)
                    removed_from_blacklist.append(b.model)

        save_blacklist(new_blacklist)

        # Print statistics
        print("-" * 60)
        print("Compatibility Testing Summary:")
        print(f"  Total models evaluated: {passed + failed + skipped}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Skipped (Offline): {skipped}")
        print("-" * 60)

        # Print blacklist updates
        if added_to_blacklist or removed_from_blacklist:
            print("Blacklist Updates:")
            if added_to_blacklist:
                print("  Added to blacklist:")
                for m in sorted(added_to_blacklist):
                    print(f"    - {m}")
            if removed_from_blacklist:
                print("  Removed from blacklist:")
                for m in sorted(removed_from_blacklist):
                    print(f"    - {m}")
        else:
            print("No blacklist changes detected.")
        print("-" * 60)

        # Save results once more to be safe
        save_results(results)


if __name__ == "__main__":
    main_cli()
