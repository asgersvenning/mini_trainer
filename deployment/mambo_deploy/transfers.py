"""Two-slot device staging; runtime imports remain optional and lazy."""

import time
from concurrent.futures import ThreadPoolExecutor


def pinned_factory(device):
    import torch

    def allocate(shape):
        with torch.cuda.device(device):
            return torch.empty(shape, dtype=torch.float32, pin_memory=True).numpy()

    return allocate


def download_tensors(values, stats=None):
    """One completed D2H copy for all ranks/embeddings, with owning NumPy views."""
    import torch

    if values[0].device.type == "cpu":
        return [value.float().numpy() for value in values]
    start = time.perf_counter()
    packed = torch.cat([value.float().reshape(-1) for value in values])
    host = torch.empty(packed.shape, dtype=torch.float32, pin_memory=True)
    begin = torch.cuda.Event(enable_timing=True)
    begin.record(torch.cuda.current_stream(packed.device))
    host.copy_(packed, non_blocking=True)
    done = torch.cuda.Event(enable_timing=True)
    done.record(torch.cuda.current_stream(packed.device))
    done.synchronize()  # CPU results must be complete before the result worker reads them.
    if stats is not None:
        stats["d2h_device_seconds"] = stats.get("d2h_device_seconds", 0.0) + begin.elapsed_time(done) / 1000
        stats["download_host_seconds"] = stats.get("download_host_seconds", 0.0) + time.perf_counter() - start
    array = host.numpy()
    result, offset = [], 0
    for value in values:
        end = offset + value.numel()
        result.append(array[offset:end].reshape(tuple(value.shape)))
        offset = end
    return result


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
                    if index == len(slot["buffers"]):
                        slot["buffers"].append(torch.empty(view.shape, dtype=torch.float32, device=device))
                        stats["device_buffer_allocations"] = stats.get("device_buffer_allocations", 0) + 1
                    target = slot["buffers"][index][: len(view)]
                    target.copy_(torch.from_numpy(view), non_blocking=True)
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
