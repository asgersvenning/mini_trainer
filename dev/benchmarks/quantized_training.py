"""Experimental CUDA QT kernel probe; not yet integrated with mt_train.

Uses INT8 stored weights (no floating-point master copy), INT8 saved linear
inputs, and scaled INT8 forward/dgrad/wgrad GEMMs. Gradients and SGD update math
remain floating point. TorchAO stochastic rounding writes updates back to INT8.
Only zero-decay, zero-momentum SGD is exercised; do not infer AdamW/Muon support.
"""

import gc
import json
import time
from argparse import ArgumentParser

import torch


def dependencies():
    from torchao.prototype.quantized_training.int8 import Int8QuantizedTrainingLinearWeight, quantize_int8_rowwise
    from torchao.prototype.quantized_training.int8_mm import scaled_int8_mm

    return Int8QuantizedTrainingLinearWeight, quantize_int8_rowwise, scaled_int8_mm


class IntegerLinear(torch.autograd.Function):
    """Row-scaled INT8 GEMMs, including approximate input and weight gradients."""

    @staticmethod
    def forward(ctx, inputs, weight):
        _, quantize, mm = dependencies()
        quantized, scale = quantize(inputs)
        ctx.save_for_backward(quantized, scale, weight.int_data, weight.scale)
        return mm(quantized.contiguous(), weight.int_data.T, scale.contiguous(), weight.scale.contiguous())

    @staticmethod
    def backward(ctx, grad_output):
        _, quantize, mm = dependencies()
        inputs, input_scale, weight, weight_scale = ctx.saved_tensors
        ones = torch.ones(weight.shape[1], device=grad_output.device, dtype=grad_output.dtype)
        grad_input = None
        if ctx.needs_input_grad[0]:
            # Weight scales lie along the contraction axis: absorb them into
            # dY before its row quantization, not into the result columns.
            quantized_grad, scale = quantize(grad_output * weight_scale)
            grad_input = mm(quantized_grad.contiguous(), weight.contiguous(), scale.contiguous(), ones)
        grad_weight = None
        if ctx.needs_input_grad[1]:
            # Similarly absorb saved activation scales into dY.T for dW.
            quantized_grad, scale = quantize(grad_output.T * input_scale)
            grad_weight = mm(quantized_grad.contiguous(), inputs.contiguous(), scale.contiguous(), ones)
        return grad_input, grad_weight


class Layer(torch.nn.Module):
    def __init__(self, width, quantized, dtype):
        super().__init__()
        self.quantized = quantized
        weights = torch.randn(width, width, device="cuda", dtype=dtype) / width**0.5
        self.weight = torch.nn.Parameter(dependencies()[0].from_float(weights) if quantized else weights)

    def forward(self, inputs):
        return IntegerLinear.apply(inputs, self.weight) if self.quantized else torch.nn.functional.linear(inputs, self.weight)


def run(width=4096, batch_size=2048, layers=4, steps=10, dtype="float16", compiled=True):
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
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, foreach=False)
        if compiled:
            optimizer.step = torch.compile(optimizer.step, fullgraph=False)

        def step(model=model, inputs=inputs, optimizer=optimizer):
            optimizer.zero_grad(set_to_none=True)
            loss = model(inputs).float().square().mean()
            loss.backward()
            optimizer.step()
            return loss

        for _ in range(3):
            step()
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
            "weight_bytes": sum(
                parameter.int_data.numel() + parameter.scale.numel() * parameter.scale.element_size()
                if quantized
                else parameter.numel() * parameter.element_size()
                for parameter in model.parameters()
            ),
        }
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
        "scope": "synthetic square linear stack and MSE; warmed forward/backward/SGD; excludes compilation and loading",
        "optimizer": {"name": "SGD", "lr": 1e-3, "momentum": 0, "weight_decay": 0},
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
    args = vars(parser.parse_args())
    args["compiled"] = not args.pop("eager")
    if min(args[key] for key in ("width", "batch_size", "layers", "steps")) < 1:
        parser.error("Dimensions, layers and steps must be positive")
    torch.set_num_threads(1)
    print(json.dumps(run(**args), indent=2))


if __name__ == "__main__":
    main()
