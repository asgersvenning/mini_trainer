"""Collect the model-card comparison on UCloud; no training or calibration."""

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import CSV_COLUMNS, write_json
from dev.releases.mambo_v3.metrics import REVISION, finite_json, require_pinned_metrics
from dev.releases.mambo_v3.package_download_metadata import CARD_COUNT

HERE = Path(__file__).parent
API = "https://api.inaturalist.org/v1"
MODELS = ("nemo", "meghan", "inaturalist")


def read(path):
    return json.loads(path.read_text())


def selected_records(args):
    """Use a deterministic prefix without modifying the frozen source manifest."""
    return read(args.output / "samples.json")["records"][: args.count]


def session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    client = requests.Session()
    client.headers["User-Agent"] = "Nemo-model-card-comparison/0.3.1"
    client.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))
    return client


def cached_json(client, url, path, **params):
    if not path.exists():
        time.sleep(1)  # Respect the public API; never parallelize scoring requests.
        response = client.get(url, params=params, timeout=60)
        response.raise_for_status()
        write_json(path, response.json())
    return read(path)


def species_taxon(client, taxon, cache):
    if taxon["rank"] == "species":
        return taxon
    result = cached_json(client, f"{API}/taxa/{taxon['id']}", cache / f"taxon-{taxon['id']}.json")["results"][0]
    return next((t for t in result.get("ancestors", []) if t["rank"] == "species"), None)


def gbif_label(client, taxon, cache):
    """Only accept unambiguous exact species matches; retain unresolved taxa explicitly."""
    result = cached_json(
        client,
        "https://api.gbif.org/v1/species/match",
        cache / f"gbif-{taxon['id']}.json",
        name=taxon["name"],
        rank="SPECIES",
        kingdom="Animalia",
        order="Lepidoptera",
        strict="true",
    )
    if result.get("matchType") == "EXACT" and result.get("rank") == "SPECIES" and result.get("order") == "Lepidoptera":
        return str(result.get("acceptedUsageKey", result["usageKey"]))
    return f"inat:{taxon['id']}"


def fetch(args):
    root = args.output
    cache = root / "responses"
    cache.mkdir(exist_ok=True)
    images = root / "images"
    images.mkdir(exist_ok=True)
    client = session()
    selection = root / "selection.json"
    if not selection.exists():
        write_json(selection, {"count": args.count, "cutoff": datetime.now(UTC).isoformat()})
    spec = read(selection)
    if args.count > spec["count"]:
        raise ValueError("Use a new output directory when increasing sample count")
    manifest = root / "samples.json"
    if manifest.exists():
        return
    records, seen = [], set()
    page = 1
    while len(records) < args.count:
        response = cached_json(
            client,
            f"{API}/observations",
            cache / f"observations-{page}.json",
            taxon_id=47157,
            quality_grade="research",
            photos="true",
            lrank="subspecies",
            hrank="species",
            order_by="created_at",
            order="desc",
            created_d2=spec["cutoff"],
            per_page=200,
            page=page,
        )
        if not response["results"]:
            raise ValueError("Not enough eligible observations")
        for observation in response["results"]:
            if len(records) == args.count:
                break
            if observation["id"] in seen:
                continue
            seen.add(observation["id"])
            photo = observation["photos"][0]
            taxon = species_taxon(client, observation["taxon"], cache)
            if taxon is None:
                raise ValueError(f"No species ancestor for observation {observation['id']}")
            url = photo["url"].replace("/square.", "/medium.")
            path = images / f"{photo['id']}.jpg"
            if not path.exists():
                response = client.get(url, timeout=60)
                response.raise_for_status()
                path.write_bytes(response.content)
            records.append(
                {
                    "id": observation["id"],
                    "photo_id": photo["id"],
                    "created_at": observation["created_at"],
                    "path": str(path.relative_to(root)),
                    "url": url,
                    "sha256": file_hash(path),
                    "taxon": taxon,
                    "label": gbif_label(client, taxon, cache),
                }
            )
        print(f"selected {len(records)}/{args.count}", flush=True)
        page += 1
    write_json(
        manifest,
        {
            **spec,
            "count": len(records),
            "sampling": "first photo per observation; latest created_at; research grade; species/subspecies",
            "records": records,
        },
    )


def cv_prediction(response):
    results = response.get("results", [])
    if not results or any("vision_score" not in r for r in results):
        raise ValueError("CV response lacks visual scores; refusing to substitute community identifications")
    return max(results, key=lambda r: float(r["vision_score"]))


def score_image(client, path, token):
    from urllib3.util.retry import Retry

    for attempt in range(5):
        with path.open("rb") as stream:
            start = time.perf_counter()
            response = client.post(
                f"{API}/computervision/score_image",
                headers={"Authorization": f"Bearer {token}"},
                files={"image": (path.name, stream, "image/jpeg")},
                data={"taxon_id": "47157", "skip_frequencies": "true"},
                timeout=60,
            )
            elapsed = time.perf_counter() - start
        if response.status_code != 429 or attempt == 4:
            response.raise_for_status()
            return response, elapsed
        delay = max(Retry().get_retry_after(response) or 0, 60 * 2**attempt)
        response.close()
        print(f"iNaturalist rate limited; waiting {delay:.0f}s before retry {attempt + 1}/4", flush=True)
        time.sleep(delay)


def inaturalist(args):
    token = os.environ.get("INAT_API_TOKEN")
    if not token:
        raise ValueError("Set INAT_API_TOKEN on UCloud to an authorized iNaturalist API JWT; never put it in arguments")
    client = session()
    cache = args.output / "responses"
    samples = selected_records(args)
    for i, record in enumerate(samples[:1] if args.stage == "check" else samples):
        target = args.output / "inaturalist" / f"{record['id']}.json"
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            continue
        path = args.output / record["path"]
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"Image changed: {path}")
        time.sleep(1)
        response, elapsed = score_image(client, path, token)
        payload = response.json()
        prediction = cv_prediction(payload)
        taxon = species_taxon(client, prediction["taxon"], cache)
        label = gbif_label(client, taxon, cache) if taxon else f"inat:{prediction['taxon']['id']}"
        write_json(target, {"label": label, "seconds": elapsed, "response": payload, "time": datetime.now(UTC).isoformat()})
        if i % 100 == 0:
            print(f"iNaturalist {i + 1}/{len(samples)}", flush=True)


def runtime_environment(model):
    return {
        "platform": platform.platform(),
        "cpu": Path("/proc/cpuinfo").read_text().split("model name", 1)[-1].splitlines()[0].lstrip("\t :"),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (("torch", "numpy", "onnxruntime") if model == "nemo" else ("torch", "numpy", "open-clip-torch"))
        },
    }


def start_runtime(folder, provenance, environment):
    path = folder / "runtime.json"
    if path.exists():
        previous = read(path)
        if previous["identity"] != provenance or any(previous.get(key) != value for key, value in environment.items()):
            raise ValueError("Cached predictions use a different source, model, CPU or runtime; use a new output")
    write_json(path, {"identity": provenance, **environment, "status": "running"})


def check_timing_environments(runtimes):
    first, second = runtimes
    if any(r["status"] != "complete" for r in runtimes):
        raise ValueError("Finish both local prediction stages before reporting timings")
    for key in ("cpu", "platform"):
        if first[key] != second[key]:
            raise ValueError(f"Local timings use different {key}; rerun on the same node")
    if first["identity"]["threads"] != second["identity"]["threads"]:
        raise ValueError("Local timings use different thread counts")
    for name in first["versions"].keys() & second["versions"].keys():
        if first["versions"][name] != second["versions"][name]:
            raise ValueError(f"Local timings use different {name} versions")


def local(args):
    import torch

    from deployment.mambo_deploy import Predictor
    from deployment.mambo_deploy.download import fetch_file
    from mini_trainer.deploy import Predictor as NativePredictor

    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    if args.stage == "nemo":
        predictor = Predictor(backend="onnx", device="cpu", model="full", threads=args.threads).load()
        identity = {"bundle": predictor.bundle.manifest, "backend": "onnx"}
    else:
        item = next(
            a
            for a in tomllib.loads((HERE / "inventory.toml").read_text())["artifacts"]
            if a["path"] == "MAMBO/hierarchical_bioclip2_ft_v1.pt"
        )
        weights = args.output / "meghan.pt"
        fetch_file(item["url"], weights, size=item["size"], sha256=item["sha256"])
        predictor = NativePredictor(device="cpu", weights=weights, precision="fp32")
        from dev.releases.mambo_v3.legacy_evaluation import verify_backbone

        identity = {"checkpoint_sha256": item["sha256"], "backbone": verify_backbone(), "backend": "torch"}
    load_seconds = time.perf_counter() - start
    vocabulary = set(predictor.classes[0])
    records = selected_records(args)
    folder = args.output / args.stage
    folder.mkdir(exist_ok=True)
    provenance = {
        "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "threads": args.threads,
        "script_sha256": file_hash(__file__),
        "samples_sha256": file_hash(args.output / "samples.json"),
        **identity,
    }
    environment = runtime_environment(args.stage)
    start_runtime(folder, provenance, environment)
    for _ in range(3):
        predictor.predict(args.output / records[0]["path"])
    for i, record in enumerate(records):
        target = folder / f"{record['id']}.json"
        if target.exists():
            continue
        path = args.output / record["path"]
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"Image changed: {path}")
        start = time.perf_counter()
        prediction = predictor.predict(path)[0]
        elapsed = time.perf_counter() - start
        write_json(target, {"label": str(prediction.label[0]), "seconds": elapsed, "known": record["label"] in vocabulary})
        if i % 100 == 0:
            print(f"{args.stage} {i + 1}/{len(records)}", flush=True)
    write_json(
        folder / "runtime.json",
        {
            "identity": provenance,
            "status": "complete",
            "load_seconds": load_seconds,
            **environment,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },
    )


def summarize(args):
    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import evaluate_file

    require_pinned_metrics()
    manifest = read(args.output / "samples.json")
    records = selected_records(args)
    result = {
        "count": len(records),
        "species_count": len({r["label"] for r in records}),
        "observation_ids": [r["id"] for r in records],
        "cutoff": manifest["cutoff"],
        "samples_sha256": file_hash(args.output / "samples.json"),
        "mini_metrics_revision": REVISION,
        "unmapped_truth": sum(r["label"].startswith("inat:") for r in records),
        "models": {},
    }
    for model in MODELS:
        predictions = [read(args.output / model / f"{r['id']}.json") for r in records]
        csv_path = args.output / f"{model}.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_COLUMNS)
            for record, prediction in zip(records, predictions, strict=True):
                writer.writerow(
                    (
                        record["id"],
                        record["path"],
                        0,
                        record["label"],
                        prediction["label"],
                        1,
                        0,
                        int(prediction.get("known", True)),
                        1,
                        1 if record["label"] == prediction["label"] else -1,
                    )
                )
        scores = finite_json(
            evaluate_file(
                MetricDF.from_source(csv_path),
                threshold=0,
                optimal=False,
                known_only=False,
                simple=True,
                hierarchical=False,
                pattern=r"^(accuracy|f1)$",
                verbose=0,
            )
        )
        result["models"][model] = {
            "macro_accuracy": scores["accuracy"]["0"],
            "macro_f1": scores["f1"]["0"],
            "median_ms": float(np.median([p["seconds"] for p in predictions]) * 1000),
            "csv_sha256": file_hash(csv_path),
            "vocabulary_coverage": None if model == "inaturalist" else sum(p["known"] for p in predictions) / len(records),
        }
        if model != "inaturalist":
            result["models"][model]["runtime"] = read(args.output / model / "runtime.json")
    check_timing_environments([result["models"][name]["runtime"] for name in MODELS[:2]])
    charts(args.output, result)
    result["figures"] = {name: file_hash(args.output / name) for name in ("quality.png", "speed.png")}
    write_json(args.output / "summary.json", result)


def charts(output, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Nemo", "Meghan", "iNaturalist CV"]
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    x = np.arange(3)
    for offset, metric, label in [(-0.18, "macro_accuracy", "Macro-Accuracy"), (0.18, "macro_f1", "Macro-F1")]:
        values = [result["models"][m][metric] * 100 for m in MODELS]
        bars = ax.bar(x + offset, values, 0.36, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=9)
    ax.set(
        xticks=x, xticklabels=names, ylim=(0, 105), ylabel="Percent", title=f"{result['count']:,} recent Research Grade Lepidoptera images"
    )
    ax.legend(loc="upper right")
    fig.savefig(output / "quality.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), layout="constrained")
    for ax, models, labels, title in [
        (axes[0], MODELS[:2], names[:2], "Local CPU · batch 1"),
        (axes[1], MODELS[2:], names[2:], "Remote API · includes network"),
    ]:
        bars = ax.bar(labels, [result["models"][m]["median_ms"] for m in models])
        ax.bar_label(bars, fmt="%.1f")
        ax.set(ylabel="Median milliseconds / image", title=title)
        ax.margins(y=0.2)
    fig.savefig(output / "speed.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("fetch", "check", *MODELS, "report", "card"), required=True)
    parser.add_argument(
        "--count", type=int, default=CARD_COUNT, help="first N observations from the frozen manifest (default: %(default)s)"
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.count < 1 or args.threads < 1:
        parser.error("count and threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "fetch":
        fetch(args)
    elif args.stage in ("check", "inaturalist"):
        inaturalist(args)
    elif args.stage == "report":
        summarize(args)
    elif args.stage == "card":
        from dev.releases.mambo_v3.package_download_metadata import card_performance

        card_performance(args.output, required=True)
        destination = HERE / "card-performance"
        destination.mkdir(exist_ok=True)
        for name in ("summary.json", "quality.png", "speed.png"):
            shutil.copyfile(args.output / name, destination / name)
    else:
        local(args)


if __name__ == "__main__":
    main()
