"""Measure SVG/PNG decoding and forced rasterization in a disposable Chromium page."""

import argparse
import base64
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New artifact directory")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise SystemExit("Install playwright and its Chromium browser in a disposable environment; see dendrogram.md") from error

    sources = {}
    for name, path in (("before", args.before), ("after", args.after)):
        if path.suffix.lower() not in (".svg", ".png"):
            parser.error("Input files must be SVG or PNG")
        mime = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
        sources[name] = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    args.output.mkdir(parents=True, exist_ok=False)
    samples = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        version = browser.version
        for size in (1200, 4800):
            for repeat in range(args.repeats):
                for name in ("before", "after") if repeat % 2 == 0 else ("after", "before"):
                    context = browser.new_context()
                    try:
                        page = context.new_page()
                        result = page.evaluate(
                            """async ({svg, size}) => {
                                const image = new Image(); const start = performance.now();
                                image.src = svg;
                                await image.decode(); const decoded = performance.now();
                                const canvas = document.createElement('canvas');
                                canvas.width = canvas.height = size;
                                const ctx = canvas.getContext('2d');
                                ctx.drawImage(image, 0, 0, size, size);
                                ctx.getImageData(0, 0, size, size);
                                const rastered = performance.now();
                                return {decode_ms: decoded-start, raster_ms: rastered-decoded,
                                    total_ms: rastered-start, png: canvas.toDataURL()};
                            }""",
                            {"svg": sources[name], "size": size},
                        )
                        png = result.pop("png")
                        if repeat == 0:
                            (args.output / f"{name}-{size}.png").write_bytes(base64.b64decode(png.split(",", 1)[1]))
                        samples.setdefault(f"{name}-{size}", []).append(result)
                    finally:
                        context.close()
        browser.close()
    medians = {key: {metric: statistics.median(v[metric] for v in values) for metric in values[0]} for key, values in samples.items()}
    report = {"chromium": version, "medians": medians, "samples": samples}
    (args.output / "measurement.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(medians, indent=2))


if __name__ == "__main__":
    main()
