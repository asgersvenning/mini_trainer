"""Collect the model-card comparison on UCloud; threshold selection belongs to mini_metrics."""

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
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dev.benchmarks.inference.onnx_inference import file_hash
from dev.releases.mambo_v3.evaluation_data import CSV_COLUMNS, write_json
from dev.releases.mambo_v3.metrics import REVISION, finite_json, require_pinned_metrics
from dev.releases.mambo_v3.package_download_metadata import CARD_COUNT

HERE = Path(__file__).parent
API = "https://api.inaturalist.org/v1"
MODELS = ("nemo", "nemo-tta", "meghan", "inaturalist")


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
            for name in (("torch", "numpy", "onnxruntime") if model != "meghan" else ("torch", "numpy", "open-clip-torch"))
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
    first = runtimes[0]
    if any(r["status"] != "complete" for r in runtimes):
        raise ValueError("Finish all local prediction stages before reporting timings")
    for second in runtimes[1:]:
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
    if args.stage in ("nemo", "nemo-tta"):
        predictor = Predictor(
            backend="onnx",
            device="cpu",
            model="full",
            threads=args.threads,
            tta=args.stage == "nemo-tta",
        ).load()
        identity = {"bundle": predictor.bundle.manifest, "backend": "onnx", "tta": predictor.tta.name if predictor.tta else None}
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
        write_json(
            target,
            {
                "label": str(prediction.label[0]),
                "confidence": float(prediction.confidence[0]),
                "seconds": elapsed,
                "known": record["label"] in vocabulary,
            },
        )
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


def prediction_confidence(prediction, model):
    if model == "inaturalist":
        # The API's visual score is a percentage; retained responses need no new requests.
        value = float(cv_prediction(prediction["response"])["vision_score"]) / 100
    elif "confidence" in prediction:
        value = float(prediction["confidence"])
    else:
        raise ValueError(f"{model} cache lacks confidence; rerun the local variants in a new output directory")
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid {model} confidence: {value}")
    return value


def summarize(args):
    from mini_metrics.data import MetricDF
    from mini_metrics.metrics import MacroAccuracy, MacroF1, evaluate_file

    require_pinned_metrics()
    manifest = read(args.output / "samples.json")
    records = selected_records(args)
    predictions_by_model = {model: [read(args.output / model / f"{r['id']}.json") for r in records] for model in MODELS}
    confidences = {model: [prediction_confidence(p, model) for p in predictions_by_model[model]] for model in MODELS}
    check_timing_environments([read(args.output / model / "runtime.json") for model in MODELS[:-1]])
    destination = args.report_output.resolve()
    source = args.output.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Report output must be outside the prediction directory")
    destination.mkdir(parents=True, exist_ok=False)
    reporting_data = {}
    result = {
        "threshold_selection": "mini_metrics.evaluate_file(optimal=True, seed=42, opt_crit=MacroF1)",
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
        predictions = predictions_by_model[model]
        csv_path = destination / f"{model}.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_COLUMNS)
            for record, prediction, confidence in zip(records, predictions, confidences[model], strict=True):
                writer.writerow(
                    (
                        record["id"],
                        record["path"],
                        0,
                        record["label"],
                        prediction["label"],
                        confidence,
                        0,
                        int(prediction.get("known", True)),
                        1,
                        1 if record["label"] == prediction["label"] else -1,
                    )
                )
        data = MetricDF.from_source(csv_path)
        scores = finite_json(
            evaluate_file(
                data,
                optimal=True,
                seed=42,
                opt_crit=MacroF1,
                eps=0.01,
                use_quantiles=True,
                known_only=False,
                simple=True,
                hierarchical=False,
                pattern=r"^(accuracy|f1|coverage|optimal_confidence_threshold)$",
                verbose=0,
            )
        )
        threshold = scores["optimal_confidence_threshold"]["0"]
        if threshold is None:
            raise ValueError("mini_metrics returned no finite threshold; retain this report for upstream diagnosis")
        reporting, calibration = data.split((0.9, 0.1), strata=("label",), seed=42)
        ids = {"reporting_ids": reporting.instance_id.tolist(), "calibration_ids": calibration.instance_id.tolist()}
        if model == MODELS[0]:
            result.update(ids)
            result.update(report_count=len(reporting), calibration_count=len(calibration))
        elif any(result[key] != value for key, value in ids.items()):
            raise ValueError("mini_metrics selected different image partitions across models")
        reporting_data[model] = reporting.with_threshold(threshold)
        result["models"][model] = {
            "threshold": threshold,
            "coverage": scores["coverage"]["0"],
            "macro_accuracy": scores["accuracy"]["0"],
            "macro_f1": scores["f1"]["0"],
            "median_ms": float(np.median([p["seconds"] for p in predictions]) * 1000),
            "csv_sha256": file_hash(csv_path),
            "vocabulary_coverage": None if model == "inaturalist" else sum(p["known"] for p in predictions) / len(records),
        }
        if model != "inaturalist":
            result["models"][model]["runtime"] = read(args.output / model / "runtime.json")
    # Same support policy as the retained release figures; keep every row's FP/FN.
    domains = []
    for data in reporting_data.values():
        truth = Counter(map(str, data.label))
        predicted = Counter(map(str, data.prediction[data.prediction_made]))
        domains.append({label for label in truth if truth[label] > 5 and predicted[label] > 5})
    shared = set.intersection(*domains)
    result["shared_species_count"] = len(shared)
    for model, data in reporting_data.items():
        for name, metric in (("macro_accuracy", MacroAccuracy()), ("macro_f1", MacroF1())):
            groups = metric(data, aggregate=False, verbose=0)[0]
            selected = {k: v for k, v in groups.items() if str(k) in shared}
            result["models"][model][name + "_shared"] = float(metric._aggregate_groups(selected)) if selected else None
    charts(destination, result)
    result["figures"] = {name: file_hash(destination / name) for name in ("quality.png", "speed.png")}
    write_json(destination / "summary.json", finite_json(result))
    print(f"Report: {destination} ({result['report_count']} reporting, {result['calibration_count']} calibration images)")


def charts(output, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Nemo", "Nemo + TTA", "Meghan", "iNaturalist CV"]
    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    x = np.arange(len(MODELS))
    for offset, metric, label in [(-0.18, "macro_accuracy", "Macro-Accuracy"), (0.18, "macro_f1", "Macro-F1")]:
        values = [result["models"][m][metric] * 100 for m in MODELS]
        bars = ax.bar(x + offset, values, 0.36, label=label)
        ax.bar_label(bars, fmt="%.1f", fontsize=9, label_type="center", color="white")
        for position, value, model in zip(x + offset, values, MODELS, strict=True):
            shared = result["models"][model][metric + "_shared"]
            if shared is not None:
                top = shared * 100
                ax.bar(position, max(0, top - value), 0.36, bottom=value, color=bars[0].get_facecolor(), alpha=0.35)
                if top < value:
                    ax.plot([position - 0.18, position + 0.18], [top, top], color="white", linestyle="--")
                ax.text(position, max(top, value) + 1, f"{top:.1f}", ha="center", fontsize=9)
    ax.set(
        xticks=x,
        xticklabels=[f"{name}\n{result['models'][model]['coverage']:.1%} accepted" for name, model in zip(names, MODELS, strict=True)],
        ylim=(0, 112),
        ylabel="Percent",
        title=f"Calibrated scores · {result['report_count']:,} reporting images",
    )
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
    fig.supxlabel(
        "Solid: full support · lighter extensions: shared support >5"
        if result["shared_species_count"]
        else "Full support · no species meet shared support >5",
        fontsize=9,
    )
    fig.savefig(output / "quality.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for ax, models, labels, title in [
        (axes[0], MODELS[:-1], names[:-1], "Local CPU · batch 1"),
        (axes[1], MODELS[-1:], names[-1:], "Remote API · includes network"),
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
    parser.add_argument("--report-output", type=Path, help="new report directory outside --output; required for report")
    args = parser.parse_args()
    if args.count < 1 or args.threads < 1:
        parser.error("count and threads must be positive")
    if args.stage == "report" and args.report_output is None:
        parser.error("report requires --report-output outside the prediction directory")
    if args.stage != "report":
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
