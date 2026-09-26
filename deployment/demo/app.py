"""Single-image Space using the published MAMBO deployment API."""

import json
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
from mambo_deploy import Predictor  # noqa: E402

# Serial requests keep cached runtimes bounded and avoid competing CPU inference.
LOCK = threading.Lock()
NAMES_PATH = Path(__file__).with_name("taxon-names.json")
NAMES = json.loads(NAMES_PATH.read_text()) if NAMES_PATH.exists() else {}


@lru_cache(maxsize=2)
def predictor_for(backend):
    if backend == "torch":
        import torch

        torch.set_num_threads(2)
    return Predictor(backend=backend, batch_size=1, threads=2)


def classify(image, backend, preset, custom, tta, topk):
    if image is None:
        raise gr.Error("Upload one moth or butterfly image first.")
    custom = custom or ""
    labels = tuple(dict.fromkeys(custom.replace(",", " ").split())) if custom.strip() else ()
    try:
        with LOCK:
            predictor = predictor_for(backend)
            predictor.configure(class_list=labels, tta=tta) if labels else predictor.configure(model=preset, tta=tta)
            # The API returns one common K for all retained ranks.
            available = min(map(len, predictor.hierarchy_plan(predictor.selected).labels))
            effective_k = min(int(topk), available)
            started = time.perf_counter()
            result = predictor.predict(image, topk=effective_k)
            elapsed = time.perf_counter() - started
            tables = []
            for rank in range(3):
                tables.append(
                    [
                        [
                            k + 1,
                            NAMES.get(result.labels[0][k][rank], ""),
                            result.labels[0][k][rank],
                            round(float(result.confidence[0, k, rank]) * 100, 2),
                        ]
                        for k in range(effective_k)
                    ]
                )
            scope = "custom class list" if labels else preset
            details = (
                f"**{backend} · CPU · {scope} · TTA {'on' if tta else 'off'}**  \n"
                f"Prediction: **{elapsed:.2f} s** (includes model loading on first use; not a throughput benchmark)."
            )
            if effective_k != int(topk):
                details += f" Showing top-{effective_k}: the smallest retained rank has {available} classes."
            return *tables, details
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        raise gr.Error(str(error)) from error


def build_app():
    presets = Predictor().available_presets()
    with gr.Blocks(title="MAMBO V3", delete_cache=(300, 300)) as app:
        gr.Markdown(
            "# MAMBO V3\nIdentify moths and butterflies at species, genus and family level. "
            "Images are processed on this server. Uploads are temporary and are not used for training. "
            "This classifies one individual; it does not detect animals or reliably reject unknown species."
        )
        with gr.Row():
            image = gr.Image(type="pil", sources=["upload"], label="One individual", image_mode="RGB")
            with gr.Column():
                backend = gr.Radio(["onnx", "torch"], value="onnx", label="Runtime")
                preset = gr.Dropdown(list(presets), value="full", label="Geographic scope")
                scope = gr.Markdown(presets["full"]["scope"])
                preset.change(lambda name: presets[name]["scope"], preset, scope, queue=False)
                tta = gr.Checkbox(value=False, label="TTA — recommended recipe, about 3× more inference work")
                topk = gr.Slider(1, 10, value=5, step=1, label="Candidates per rank")
                with gr.Accordion("Custom species list", open=False):
                    custom = gr.Textbox(value="", label="GBIF species IDs, separated by spaces, commas or newlines", lines=3)
                    gr.Markdown("When supplied, this replaces the geographic preset.")
                run = gr.Button("Classify", variant="primary")
        status = gr.Markdown()
        tables = [
            gr.Dataframe(
                headers=["Rank", "Name", "GBIF taxon ID", "Confidence (%)"],
                datatype=["number", "str", "str", "number"],
                interactive=False,
                label=rank,
            )
            for rank in ("Species", "Genus", "Family")
        ]
        gr.Markdown(
            "Ranks are predicted independently; the winning IDs need not form one ancestral path. "
            "TTA helps the evaluated monitoring crops but can hurt on other image domains. "
            "Weights: **CC BY-NC-SA 4.0**; adapter: MIT. "
            "[Integration, evidence and limitations](https://github.com/asgersvenning/mini_trainer/"
            "blob/MAMBO_v3/deployment/README.md)."
        )
        run.click(classify, [image, backend, preset, custom, tta, topk], [*tables, status], concurrency_limit=1, api_name=False)
    return app.queue(max_size=8, default_concurrency_limit=1)


if __name__ == "__main__":
    build_app().launch(share=False, max_file_size="20mb", enable_monitoring=False, run_history=False)
