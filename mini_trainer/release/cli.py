"""``mt_release``: checkpoint -> verified release folder -> local Hugging Face repository.

    mt_release export     --weights last.pt --output release/ --ranks species,genus,family --images test.txt
    mt_release add-head   --weights head.pt --release release/ --name moths --images moths.txt
    mt_release verify     --weights last.pt --release release/ --images test.txt
    mt_release package-hf --release release/ --repo-id user/model --card card.json

``export`` and ``add-head`` verify on ``--images`` (a folder, a text file of paths, or a JSON list) unless
``--no-verify``; ``package-hf`` refuses nothing but lists what blocks a release and prints the upload
command for the owner to run.
"""

import json
import random
from argparse import ArgumentParser
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def image_list(source: str, limit: int | None) -> list[str]:
    """Image paths from a folder, a JSON list (or ``{"path": [...]}``) or a text file; a seeded sample of ``limit``."""
    path = Path(source)
    if path.is_dir():
        paths = sorted(str(p) for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
        paths = data["path"] if isinstance(data, dict) else data
    else:
        paths = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if limit and len(paths) > limit:
        paths = random.Random(0).sample(paths, limit)
    return paths


def _verify(args, release, head=None):
    from .verify import summary, verify_release

    if args.no_verify:
        return
    if not args.images:
        print("WARNING: no --images; the release is NOT verified on real images. Run `mt_release verify`.")
        return
    report = verify_release(release, args.weights, image_list(args.images, args.limit), head=head, threads=args.threads)
    print(summary(report))
    if not report["pass"]:
        raise SystemExit(1)


def run():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, weights=True):
        if weights:
            p.add_argument("--weights", required=True, help="mini_trainer checkpoint (.pt)")
        p.add_argument("--images", help="verification images: folder, text file of paths, or JSON list")
        p.add_argument("--limit", type=int, default=300, help="sample this many verification images (default 300)")
        p.add_argument("--threads", type=int, default=4, help="ONNX Runtime and torch CPU threads (default 4)")
        p.add_argument("--no-verify", action="store_true")

    e = sub.add_parser("export", help="checkpoint -> release folder (trunk, default head, ONNX graphs)")
    common(e)
    e.add_argument("--output", required=True, help="new release folder")
    e.add_argument("--ranks", help="rank names, finest first, e.g. species,genus,family")
    e.add_argument("--head-name", default="main", help="name of the checkpoint's head in the release (default main)")
    e.add_argument("--no-fp16", action="store_true", help="skip trunk_fp16.onnx")
    e.add_argument("--no-combined", action="store_true", help="skip model.safetensors and model.onnx (trunk and head in one file)")

    a = sub.add_parser("add-head", help="add a head trained on this release's frozen trunk")
    common(a)
    a.add_argument("--release", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--ranks")

    v = sub.add_parser("verify", help="check every artifact against mini_trainer on real images")
    common(v)
    v.add_argument("--release", required=True)
    v.add_argument("--head", help="head to verify (default: the release's default head)")

    p = sub.add_parser("package-hf", help="write README.md and predict.py, check, print the upload command")
    p.add_argument("--release", required=True)
    p.add_argument("--repo-id", required=True, help="Hugging Face repository, e.g. user/model")
    p.add_argument("--card", required=True, help="JSON: title, summary, license, training, results, limitations, citation")
    p.add_argument("--example-image", help="image predict.py is test-run on (default: one from verification.json)")

    args = parser.parse_args()
    ranks = args.ranks.split(",") if getattr(args, "ranks", None) else None
    if args.command == "export":
        from .build import export_release

        release = export_release(args.weights, args.output, ranks, args.head_name, fp16=not args.no_fp16, combined=not args.no_combined)
        _verify(args, release)
    elif args.command == "add-head":
        from .build import add_head

        _verify(args, add_head(args.weights, args.release, args.name, ranks), head=args.name)
    elif args.command == "verify":
        args.no_verify = False
        _verify(args, args.release, head=args.head)
    else:
        from .card import package_hf

        problems = package_hf(args.release, args.repo_id, json.loads(Path(args.card).read_text()), args.example_image)
        raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    run()
