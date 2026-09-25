"""Two-slot device staging; runtime imports remain optional and lazy."""

import math
import time
from concurrent.futures import ThreadPoolExecutor


def pinned_factory(device, *, compact=False):
    import torch

    def allocate(shape):
        with torch.cuda.device(device):
            return torch.empty(shape, dtype=torch.uint8 if compact else torch.float32, pin_memory=True).numpy()

    return allocate


def download_tensors(values, stats=None, *, torch, defer=False, stream=None):
    """Return CPU arrays, or a completion callable retaining buffers until D2H finishes."""
    if values[0].device.type == "cpu":
        arrays = [value.float().numpy() for value in values]
        return (lambda: arrays) if defer else arrays
    packed = torch.cat([value.float().reshape(-1) for value in values])
    shapes = [tuple(value.shape) for value in values]
    host = torch.empty(packed.shape, dtype=torch.float32, pin_memory=True)
    current = torch.cuda.current_stream(packed.device)
    stream = stream or current
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        begin, done = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record(stream)
        host.copy_(packed, non_blocking=True)
        done.record(stream)
        packed.record_stream(stream)

    def finish():
        start = time.perf_counter()
        done.synchronize()  # Only the result consumer waits; no CPU access before completion.
        if stats is not None:
            stats["d2h_device_seconds"] = stats.get("d2h_device_seconds", 0.0) + begin.elapsed_time(done) / 1000
            stats["output_completion_wait_seconds"] = stats.get("output_completion_wait_seconds", 0.0) + time.perf_counter() - start
        array = host.numpy()
        result, offset = [], 0
        for shape in shapes:
            end = offset + math.prod(shape)
            result.append(array[offset:end].reshape(shape))
            offset = end
        return result

    return finish if defer else finish()


def device_batches(source, backend, device, stats):
    """Stage N+1 while N executes; yielded buffers are leased until the next iteration."""
    device_id = int(device.split(":")[-1]) if ":" in device else 0
    slots = [{"buffers": [], "used": None}, {"buffers": [], "used": None}]
    counter = 0
    if backend == "torch":
        import torch

        copy_stream = torch.cuda.Stream(device=device)
    else:
        import onnxruntime as ort

        ort.preload_dlls() if hasattr(ort, "preload_dlls") else None
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mambo-transfer")

    def stage():
        nonlocal counter
        try:
            offset, views = next(source)
        except StopIteration:
            return None
        slot = slots[counter % 2]
        counter += 1
        start = time.perf_counter()
        if backend == "torch":
            with torch.cuda.device(device), torch.cuda.stream(copy_stream):
                if slot["used"] is not None:
                    copy_stream.wait_event(slot["used"])
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(copy_stream)
                staged = []
                for index, view in enumerate(views):
                    source_tensor = torch.from_numpy(view)
                    if index == len(slot["buffers"]):
                        slot["buffers"].append(torch.empty(view.shape, dtype=source_tensor.dtype, device=device))
                        stats["device_buffer_allocations"] = stats.get("device_buffer_allocations", 0) + 1
                    target = slot["buffers"][index][: len(view)]
                    target.copy_(source_tensor, non_blocking=True)
                    staged.append(target)
                end.record(copy_stream)
                end.synchronize()  # The source may recycle its pinned host buffers on next().
                stats["h2d_device_seconds"] = stats.get("h2d_device_seconds", 0.0) + begin.elapsed_time(end) / 1000
        else:
            staged = []
            for index, view in enumerate(views):
                if index == len(slot["buffers"]) or tuple(slot["buffers"][index].shape()) != view.shape:
                    value = ort.OrtValue.ortvalue_from_numpy(view, "cuda", device_id)
                    if index == len(slot["buffers"]):
                        slot["buffers"].append(value)
                    else:
                        slot["buffers"][index] = value
                    stats["device_buffer_allocations"] = stats.get("device_buffer_allocations", 0) + 1
                else:
                    slot["buffers"][index].update_inplace(view)
                staged.append(slot["buffers"][index])
        stats["transfer_worker_seconds"] = stats.get("transfer_worker_seconds", 0.0) + time.perf_counter() - start
        return offset, tuple(staged), slot, len(views[0])

    future = pool.submit(stage)
    try:
        while True:
            result = future.result()
            if result is None:
                break
            offset, views, slot, count = result
            future = pool.submit(stage)
            try:
                yield offset, views, count
            finally:
                if backend == "torch":
                    with torch.cuda.device(device):
                        slot["used"] = torch.cuda.Event()
                        slot["used"].record(torch.cuda.current_stream(device))
    finally:
        future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        source.close()
        if backend == "torch":
            copy_stream.synchronize()
