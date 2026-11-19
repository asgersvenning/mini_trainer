"""Turn a verified release folder into a Hugging Face model repository, locally. Nothing is uploaded.

The card is generated from the release itself (files, sizes, preprocessing, heads, verification) and
from a small JSON file the model's owner writes: title, summary, licence, training, results with their
sources, limitations, citation. The licence is never guessed: without one the folder is reported as
not ready. :func:`package_hf` ends with the one command that would upload the folder.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from .build import write_config

REQUIRED = ("title", "summary", "license")


def model_card(config: dict, verification: dict, card: dict) -> str:
    prep, trunk = config["preprocessing"], config["trunk"]
    heads = config["heads"]
    default = heads[config["default_head"]]
    sizes = {path: entry["size"] / 1e6 for path, entry in config["files"].items()}
    front = ["---", f"license: {card.get('license', 'TODO-OWNER-DECISION')}"]
    front += [f"{key}: {card[key]}" for key in ("license_name", "license_link", "base_model") if key in card]
    front += ["library_name: onnx", "pipeline_tag: image-classification", "tags:"]
    front += [f"  - {tag}" for tag in card.get("tags", [])] + ["  - onnx", "  - safetensors", "  - mini_trainer", "---", ""]
    head_rows = [
        f"| `{name}` | {', '.join(f'{rank} ({n:,})' for rank, n in zip(h['ranks'], h['sizes']))} | `{h['file']}`, `{h['labels']}` |"
        for name, h in heads.items()
    ]
    file_rows = [f"| `{path}` | {mb:,.1f} MB |" for path, mb in sizes.items()]
    checks = []
    for name, art in verification.get("artifacts", {}).items():
        if "ranks" in art:
            worst = min(r["top1_agreement"] for r in art["ranks"].values())
            dmax = max(r["max_abs_logit_diff"] for r in art["ranks"].values())
            checks.append(f"| `{name}` | {worst:.2%} | {dmax:.1e} | {art['embedding_cosine_min']:.5f} |")
    results = [f"- {r['claim']} (source: {r['source']})" for r in card.get("results", [])]
    lines = front + [
        f"# {card.get('title', 'TODO: title')}",
        "",
        card.get("summary", "TODO: summary"),
        "",
        "| | |",
        "|---|---|",
        f"| **Trunk** | `{trunk['backbone']}`, {trunk['features']}-d features and the `{config['default_head']}` head's embedding |",
        f"| **Default head** | {', '.join(f'{rank} ({n:,})' for rank, n in zip(default['ranks'], default['sizes']))} |",
        f"| **Input** | one RGB image; the largest centred square, resized to {prep['read_size']} px (see Preprocessing) |",
        f"| **Licence** | {card.get('license_text', card.get('license', '**not decided**'))} |",
        "",
        "## Quick start",
        "",
        "```bash",
        "pip install onnxruntime numpy pillow safetensors",
        "python predict.py photo.jpg --top 3",
        "```",
        "",
        "```python",
        "from predict import Release, subset",
        "release = Release('.')                               # trunk.onnx + the default head",
        "print(release.predict(['photo.jpg'], top=3))         # per rank: [(label, probability), ...]",
        "features, embedding = release.embed(['photo.jpg'])   # embedding: unit length, cosine = dot product",
        "local = subset(release.head, rows=[0, 5, 9])         # a species list is a subset of rows",
        "print(release.predict(['photo.jpg'], head=local))",
        "```",
        "",
        "## How the model is split",
        "",
        "The **trunk** (`trunk.onnx`, `trunk_fp16.onnx`, `trunk.safetensors`) maps an image to `features`, the",
        "classifier's input, and to `embedding`, the default head's L2-normalised embedding (useful for retrieval and",
        "few-shot use on its own). A **head** (`heads/<name>.safetensors`) holds an optional hidden layer, one unit",
        "prototype per class with a learned magnitude `g` and bias `b`, and each class's parent at every rank.",
        "`predict.py` scores it with numpy:",
        "",
        "```",
        "logit_c = sqrt(D - 2) * asin(clamp(g_c * cos(embedding, prototype_c), -1 + 1e-7, 1 - 1e-7)) + b_c",
        "logit(parent) = logsumexp(logit(children))      # so softmax at any rank is that rank's marginal",
        "```",
        "",
        "Heads trained on the same frozen trunk share it; `config.json` records the trunk's hash, so a head can only",
        "be added to the trunk it was trained on. For a single file, `model.safetensors` holds the trunk and the default",
        "head (`trunk.*`, `head.*`) and `model.onnx` is them as one graph (outputs `embedding` and `logits_<rank>`).",
        "safetensors is the format to fine-tune or load in PyTorch; ONNX is the format to deploy.",
        "",
        "| head | ranks (classes) | files |",
        "|---|---|---|",
        *head_rows,
        "",
        "## Preprocessing",
        "",
        f"Outside the graph: {prep['reader']} (`read_square` in `predict.py`). Inside the graph: {prep['in_graph']}",
        f"(model size {prep['model_size']} px). Feed `uint8` `[N, 3, {prep['read_size']}, {prep['read_size']}]`; do not normalise.",
        "",
        "## Verification",
        "",
        f"Every file was checked against mini_trainer's own evaluation of the checkpoint on {verification.get('images', 0)}",
        "real images, through `predict.py`'s own reader (identical pixels for "
        f"{verification.get('preprocessing', {}).get('identical_images', 0):.1%} of images). This is agreement with the",
        "training framework, not accuracy. The largest logit differences are for classes whose cosine sits at the clamp",
        "(asin is vertical at -1 and 1), far below the top class; they do not change rankings.",
        "",
        "| file | top-1 agreement, worst rank | max logit difference | min embedding cosine |",
        "|---|---|---|---|",
        *checks,
        "",
        "## Files",
        "",
        "| file | size |",
        "|---|---|",
        *file_rows,
        "",
        "## Results",
        "",
        *(results or ["No evaluation results are claimed in this card yet."]),
        "",
        "## Training",
        "",
        card.get("training", "TODO: training data and recipe."),
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in card.get("limitations", ["TODO: limitations."])],
        "",
        "## Licence",
        "",
        card.get("license_section", "**Not decided.** The model's owner must set the licence before release."),
        "",
    ]
    if "citation" in card:
        lines += ["## Citation", "", "```bibtex", card["citation"].strip(), "```", ""]
    return "\n".join(lines)


def check_predict_runs(folder: Path, image: str) -> str | None:
    """Run ``predict.py`` in a fresh, isolated interpreter; it must not import torch or mini_trainer."""
    code = (
        "import runpy, sys; image, folder = sys.argv[1:]; sys.argv = ['predict.py', image, '--release', folder];"
        "runpy.run_path(folder + '/predict.py', run_name='__main__');"
        "bad = sorted(m for m in ('torch', 'mini_trainer', 'timm') if m in sys.modules);"
        "sys.exit('imported ' + ', '.join(bad) if bad else 0)"
    )
    done = subprocess.run([sys.executable, "-I", "-c", code, image, str(folder)], capture_output=True, text=True)
    return None if done.returncode == 0 else (done.stderr or done.stdout).strip()[-500:]


def package_hf(release, repo_id: str, card: dict, example_image: str | None = None) -> list[str]:
    """Write ``README.md`` and ``predict.py`` into the release, check it, print the upload command.

    Returns the problems that block a release (an empty list means ready).
    """
    folder = Path(release)
    config = json.loads((folder / "config.json").read_text())
    verification_path = folder / "verification.json"
    verification = json.loads(verification_path.read_text()) if verification_path.exists() else {}
    for name in card.get("license_files", []):
        shutil.copyfile(name, folder / Path(name).name)
    shutil.copyfile(Path(__file__).with_name("standalone.py"), folder / "predict.py")
    write_config(folder, config)  # file hashes now include predict.py and any licence files
    (folder / "README.md").write_text(model_card(config, verification, card))
    write_config(folder, config)

    problems = [f"card field missing: {key}" for key in REQUIRED if key not in card]
    if not verification:
        problems.append("no verification.json: run `mt_release verify` first")
    elif not verification.get("pass"):
        problems.append("verification failed")
    missing = [path for path in config["files"] if not (folder / path).is_file()]
    problems += [f"missing file: {path}" for path in missing]
    image = example_image or next((p for p in verification.get("example_images", []) if Path(p).is_file()), None)
    failure = check_predict_runs(folder, image) if image else "no example image to run predict.py on"
    if failure:
        problems.append(f"predict.py did not run standalone: {failure}")
    print(f"\nTo upload (the owner runs this; nothing has been uploaded):\n  hf upload {repo_id} {folder} . --repo-type model")
    print("Ready." if not problems else "Not ready:\n  - " + "\n  - ".join(problems))
    return problems
