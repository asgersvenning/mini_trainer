"""One-batch CUDA transfer lookahead, independent of model precision."""

import torch
from torch.utils.data import DataLoader


def _map_tensors(batch, operation):
    if isinstance(batch, torch.Tensor):
        return operation(batch)
    if isinstance(batch, dict):
        return {key: _map_tensors(value, operation) for key, value in batch.items()}
    if isinstance(batch, tuple) and hasattr(batch, "_fields"):
        return type(batch)(*(_map_tensors(value, operation) for value in batch))
    if isinstance(batch, (tuple, list)):
        return type(batch)(_map_tensors(value, operation) for value in batch)
    return batch


class CUDAPrefetchLoader(DataLoader):
    """DataLoader whose opt-in iterator yields tensors on the requested CUDA device.

    The sampler and worker lifecycle remain those of DataLoader. Only one extra
    device batch is staged. Preprocessing and augmentation stay on the caller's
    compute stream. CPU hooks may run one batch earlier than without lookahead.
    """

    def __init__(self, *args, device, **kwargs):
        self.transfer_device = torch.device(device)
        if self.transfer_device.type != "cuda":
            raise ValueError("CUDA batch prefetch requires a CUDA target device.")
        super().__init__(*args, **kwargs)

    def __iter__(self):
        return self._prefetch(super().__iter__())

    def _prefetch(self, source):
        stream = torch.cuda.Stream(device=self.transfer_device)
        stream.wait_stream(torch.cuda.current_stream(self.transfer_device))
        sentinel = object()

        def preload():
            batch = next(source, sentinel)
            if batch is sentinel:
                return sentinel
            producer = torch.cuda.current_stream(self.transfer_device)

            def transfer(tensor):
                if tensor.device.type == "cuda":
                    if tensor.device != stream.device:
                        raise ValueError("CUDA prefetch cannot stage tensors from another CUDA device.")
                    stream.wait_stream(producer)
                    tensor.record_stream(stream)
                return tensor.to(self.transfer_device, non_blocking=True)

            with torch.cuda.stream(stream):
                return _map_tensors(batch, transfer)

        pending = preload()
        while pending is not sentinel:
            current = torch.cuda.current_stream(self.transfer_device)
            # Wait only for this batch, before queueing the next copy.
            current.wait_stream(stream)

            def record(tensor):
                tensor.record_stream(current)
                return tensor

            batch = _map_tensors(pending, record)
            error = None
            try:
                pending = preload()
            except Exception as caught:
                # Deliver the already loaded batch before surfacing a later
                # reader failure, as ordinary sequential iteration would.
                error = caught
            yield batch
            if error is not None:
                raise error
