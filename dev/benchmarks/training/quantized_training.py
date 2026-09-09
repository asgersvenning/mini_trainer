"""CUDA QT kernel probe using the opt-in model backend.

Uses INT8 stored weights (no floating-point master copy), INT8 saved linear
inputs, and scaled INT8 forward/dgrad/wgrad GEMMs. Gradients and SGD update math
remain floating point. TorchAO stochastic rounding writes updates back to INT8.
SGD and AdamW are available; do not infer support for other optimizers.
"""

import gc
import json
import math
import time
from argparse import ArgumentParser

import torch

from mini_trainer.modeling.quantized_training import IntegerLinear as IntegerLinear


def dependencies():
    from mini_trainer.modeling._quantized_training import TrainingWeight, quantize_int8_rowwise, scaled_int8_mm

    return TrainingWeight, quantize_int8_rowwise, scaled_int8_mm


class Layer(torch.nn.Module):
    def __init__(self, width, quantized, dtype):
        super().__init__()
        self.quantized = quantized
        weights = torch.randn(width, width, device="cuda", dtype=dtype) / width**0.5
        self.weight = torch.nn.Parameter(dependencies()[0].from_float(weights) if quantized else weights)

    def forward(self, inputs):
        return torch.nn.functional.linear(inputs, self.weight)


def run(
    width=4096,
    batch_size=2048,
    layers=4,
    steps=10,
    dtype="float16",
    compiled=True,
    optimizer_name="sgd",
    weight_decay=0.0,
    momentum=0.0,
    epsilon=1e-4,
):
    if not torch.cuda.is_available():
        raise RuntimeError("The QT probe requires an accessible CUDA GPU.")
    results = {}
    dependencies()
    for quantized in (False, True):
        torch.manual_seed(42)
        gc.collect()
        torch.cuda.empty_cache()
        model = torch.nn.Sequential(*[Layer(width, quantized, getattr(torch, dtype)) for _ in range(layers)])
        if compiled:
            model = torch.compile(model, fullgraph=True)
        inputs = torch.randn(batch_size, width, device="cuda", dtype=getattr(torch, dtype))
        optimizer_cls = {"sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}[optimizer_name]
        optimizer_options = {"momentum": momentum} if optimizer_name == "sgd" else {"eps": epsilon}
        optimizer = optimizer_cls(model.parameters(), lr=1e-3, weight_decay=weight_decay, foreach=False, **optimizer_options)
        if compiled:
            optimizer.step = torch.compile(optimizer.step, fullgraph=False)

        def step(model=model, inputs=inputs, optimizer=optimizer):
            optimizer.zero_grad(set_to_none=True)
            loss = model(inputs).float().square().mean()
            loss.backward()
            optimizer.step()
            return loss

        warmup_losses = []
        for _ in range(3):
            warmup_losses.append(float(step().detach()))
            # This random linear-stack MSE must train every weight. AOT can
            # otherwise report fast steps while silently dropping gradients
            # for the experimental tensor subclass.
            if any(parameter.grad is None or not bool(torch.count_nonzero(parameter.grad)) for parameter in model.parameters()):
                raise RuntimeError(
                    "Missing or zero weight gradients in the QT probe; compiled tensor-subclass training is not verified. Try --eager."
                )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(steps):
            loss = step()
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - started) / steps
        results["int8" if quantized else dtype] = {
            "seconds_per_step": seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "loss": float(loss.detach()),
            "warmup_losses": warmup_losses,
            "weight_bytes": sum(
                parameter.int_data.numel() + parameter.scale.numel() * parameter.scale.element_size()
                if quantized
                else parameter.numel() * parameter.element_size()
                for parameter in model.parameters()
            ),
        }
        if not math.isfinite(results["int8" if quantized else dtype]["loss"]):
            raise RuntimeError("Nonfinite training loss; check optimizer precision, epsilon and learning rate.")
        del step, model, inputs, optimizer, loss
    return {
        "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "width": width,
        "batch_size": batch_size,
        "layers": layers,
        "steps": steps,
        "compiled": compiled,
        "seed": 42,
        "scope": "synthetic square linear stack and MSE; warmed forward/backward/optimizer; excludes compilation and loading",
        "optimizer": {
            "name": optimizer_name,
            "lr": 1e-3,
            "weight_decay": weight_decay,
            **({"momentum": momentum} if optimizer_name == "sgd" else {"eps": epsilon}),
        },
        "results": results,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--optimizer-name", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=1e-4, help="AdamW epsilon; 1e-4 avoids FP16 underflow")
    args = vars(parser.parse_args())
    args["compiled"] = not args.pop("eager")
    if min(args[key] for key in ("width", "batch_size", "layers", "steps")) < 1:
        parser.error("Dimensions, layers and steps must be positive")
    if args["optimizer_name"] != "sgd" and args["momentum"]:
        parser.error("--momentum applies only to SGD")
    torch.set_num_threads(1)
    print(json.dumps(run(**args), indent=2))


if __name__ == "__main__":
    main()
