"""Qualify exact installed candidate wheels in an isolated CPU environment."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from dev.releases.mambo_v3.prepare_candidate import HERE, ROOT, digest
from dev.releases.mambo_v3.publication_assets import seal, verify


def qualify(candidate, dataset=None):
    manifest = verify(candidate, qualified=False)
    reports = candidate / "qualification"
    reports.mkdir(exist_ok=False)
    deployment = next((candidate / "dist").glob("mambo_v3-*.whl"))
    training = next((candidate / "dist").glob("minitrainer-*.whl"))
    with tempfile.TemporaryDirectory(prefix="mambo-installed-") as directory:
        work = Path(directory)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "GRADIO_ANALYTICS_ENABLED": "False", "MAMBO_CACHE": str(work / "cache")}
        env.pop("PYTHONPATH", None)
        env.pop("MAMBO_BUNDLE", None)
        env.pop("MAMBO_OFFLINE", None)
        subprocess.run(["uv", "venv", "--python", "3.13", str(work / "env")], check=True)
        python = work / "env/bin/python"

        def run(*args):
            subprocess.run([str(python), "-I", *map(str, args)], cwd=work, env=env, check=True)

        subprocess.run(["uv", "pip", "install", "--python", str(python), f"{deployment}[onnx]"], check=True)
        run(
            "-c",
            "import importlib.util; assert importlib.util.find_spec('torch') is None; "
            "assert importlib.util.find_spec('mini_trainer') is None",
        )
        if dataset is None:
            dataset = work / "images"
            run(
                "-c",
                f"from pathlib import Path; from PIL import Image; p=Path({str(dataset)!r}); "
                "[(p/str(i)).mkdir(parents=True) for i in range(4)]; "
                "[Image.new('RGB',(96+i*7,80+i*11),(30+i*45,100,150)).save(p/str(i)/'fixture.jpg') for i in range(4)]",
            )
            fixture_kind = "synthetic integration fixtures; no accuracy claim"
        else:
            dataset = dataset.resolve()
            fixture_kind = "four retained real images; no new accuracy benchmark"
        image = next(dataset.glob("*/*.jpg"))
        run(HERE / "check_download_install.py", image, reports / "download.json")
        env["MAMBO_OFFLINE"] = "1"
        subprocess.run(
            [str(python.parent / "mambo_predict"), "-i", str(image), "-o", str(work / "cli"), "--name", "smoke", "--tta", "--embeddings"],
            cwd=work,
            env=env,
            check=True,
        )
        run(
            "-c",
            f"import json,numpy as np; from pathlib import Path; p=Path({str(work / 'cli/smoke')!r}); "
            "assert len(json.loads((p/'predictions.json').read_text())['results'])==1; "
            "assert np.load(p/'embeddings.npy').shape==(1,1280); assert (p/'mini_metric.csv').is_file()",
        )
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "--torch-backend", "cpu", str(training), f"{deployment}[torch]"], check=True
        )
        for tta in ("none", "rotation30_pad25_3"):
            run(HERE / "qualify_bundle.py", candidate / "mambo-v3-bundle", dataset, "--tta", tta, "--output", reports / f"cpu-{tta}.json")
        subprocess.run(["uv", "pip", "install", "--python", str(python), "gradio==6.28.0"], check=True)
        env["MAMBO_BUNDLE"] = str(candidate / "mambo-v3-bundle")
        run(HERE / "qualify_demo.py", ROOT / "deployment/demo/app.py", image, reports / "demo.json")
        subprocess.run(["uv", "pip", "check", "--python", str(python)], check=True)
        packages = subprocess.check_output(["uv", "pip", "freeze", "--python", str(python)], text=True)
        (reports / "environment.txt").write_text(packages)
        (reports / "validation.json").write_text(
            json.dumps(
                {
                    "source_commit": manifest["source_commit"],
                    "fixture_kind": fixture_kind,
                    "wheel_sha256": {p.name: digest(p) for p in (deployment, training)},
                    "onnx_without_torch": True,
                    "automatic_download_and_offline_cache": True,
                    "cli_tta_embeddings": True,
                    "cpu_backends_presets_custom_tta_embeddings": True,
                },
                indent=2,
            )
            + "\n"
        )
    manifest["qualification"] = "passed"
    (candidate / "release-candidate.json").write_text(json.dumps(manifest, indent=2) + "\n")
    seal(candidate)
    verify(candidate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--dataset", type=Path, help="Optional retained image directory; otherwise use synthetic fixtures")
    args = parser.parse_args()
    qualify(args.candidate.resolve(), args.dataset)
