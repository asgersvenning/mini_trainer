"""Package a verified embedding ONNX export for the static browser explorer."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory


def package_bundle(export, output, runtime, weights):
    """Package the supported RGB nearest-square/ImageNet evaluation contract.

    Runtime is the unpacked, pinned onnxruntime-web npm package. This command
    copies assets; it never downloads code or changes a Python environment.
    """
    export, output, runtime = Path(export), Path(output), Path(runtime)
    manifest = json.loads((export / "manifest.json").read_text())
    if manifest["output_semantics"] != "predictions_and_preclassification_embedding":
        raise ValueError("Export with include_embeddings=True first.")
    if len(manifest["classifiers"]) != 1:
        raise ValueError("Browser bundle requires one classifier.")
    metadata = manifest["classifiers"][0]["metadata"]
    mapping = metadata["cls2idx"]
    levels = [mapping[str(i)] for i in range(len(mapping))] if isinstance(next(iter(mapping.values())), dict) else [mapping]
    classes = [[str(name) for name, _ in sorted(level.items(), key=lambda item: item[1])] for level in levels]
    shape = manifest["input"]["shape"]
    recipe = manifest["preprocessing"]["recipe"]
    if (
        manifest["input"]["dtype"] != "float32"
        or shape[1] != 3
        or shape[2] != shape[3]
        or not recipe
        or recipe.get("contract") != "nearest-square-uint8-bilinear-center-imagenet-v1"
        or recipe["size"] != shape[2]
        or recipe["resize"] < shape[2]
    ):
        raise ValueError("Bundle requires an explicitly verified nearest-square RGB and ImageNet preprocessing recipe.")
    from .explore import load_prototypes

    prototypes, names, _, provenance = load_prototypes(Path(weights))
    if provenance["checkpoint_sha256"] != manifest["source"]["checkpoint_sha256"] or names != classes[0]:
        raise ValueError("Prototype checkpoint differs from the ONNX export.")
    version = json.loads((runtime / "package.json").read_text())["version"]
    if version != "1.24.3":
        raise ValueError("This browser bundle supports onnxruntime-web 1.24.3.")
    if output.exists():
        raise FileExistsError(output)
    # Check every supplied export artifact before creating a distribution.
    for name, digest in manifest["artifacts"].items():
        if Path(name).name != name or hashlib.sha256((export / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Invalid export artifact: {name}")
    destination = output
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".browser-bundle-", dir=destination.parent) as temporary:
        output = Path(temporary) / "bundle"
        output.mkdir()
        for name in manifest["artifacts"]:
            shutil.copy2(export / name, output / name)
        (output / "prototypes.f32").write_bytes(prototypes.numpy().astype("<f4").tobytes())
        assets = output / "runtime"
        assets.mkdir()
        for name in ("ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"):
            shutil.copy2(runtime / "dist" / name, assets / name.replace(".mjs", ".js"))
        shutil.copy2(runtime / "LICENSE.txt", assets / "LICENSE")
        shutil.copy2(Path(__file__).with_name("inference_worker.js"), output / "inference_worker.js")
        shutil.copy2(Path(__file__).with_name("insertion.js"), output / "insertion.js")
        bundle = {
            "prototypes": "prototypes.f32",
            "embedding_dimensions": prototypes.shape[1],
            "schema": "mini-trainer-browser-v1",
            "checkpoint_sha256": manifest["source"]["checkpoint_sha256"],
            "model": "model.onnx",
            "external_data": [name for name in manifest["artifacts"] if name != "model.onnx"],
            "outputs": [item["name"] for item in manifest["outputs"] if item["name"] != "embedding"],
            "embedding": "embedding",
            "classes": classes,
            "size": recipe["size"],
            "resize": recipe["resize"],
            "preprocessing": recipe["contract"],
            "runtime_version": version,
            "provider": "wasm",
            "artifacts": {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*") if p.is_file()},
        }
        (output / "manifest.json").write_text(json.dumps(bundle, indent=2) + "\n")
        output.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="Matching source checkpoint for prototype directions")
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path, help="Unpacked onnxruntime-web 1.24.3 npm package")
    args = parser.parse_args()
    print(package_bundle(args.export, args.output, args.runtime, args.weights))


if __name__ == "__main__":
    main()
