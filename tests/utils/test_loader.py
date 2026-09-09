import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from mini_trainer.data import _workers
from mini_trainer.data import io as data_io
from mini_trainer.data import loader as data_loader
from mini_trainer.data.loader import PathLabelProcessor, get_dataset_dataloader, get_inference_dataloader


def _assert_shared_worker_batch(samples):
    assert torch.utils.data.get_worker_info() is not None
    assert not torch.cuda.is_initialized()
    values = samples.data if isinstance(samples.data, (tuple, list)) else (samples.data,)
    # Check before the multiprocessing queue can copy ordinary storage into
    # shared memory; checking only the parent's received batch would miss it.
    assert all(value.is_shared() for value in values)
    return data_loader._collate_batch(samples)


def test_nearest_coordinates_match_legacy_float_resize():
    sources = [*range(1, 64), 127, 255, 257, 1023, 2049]
    targets = [*range(1, 64), 127, 224, 257, 4096]
    for source in sources:
        values = torch.arange(source, dtype=torch.float32).reshape(1, 1, source, 1)
        for target in targets:
            expected = torch.nn.functional.interpolate(values, size=(target, 1), mode="nearest").flatten().long()
            torch.testing.assert_close(data_io._nearest_indices(source, target), expected, rtol=0, atol=0)


@pytest.mark.parametrize("source", [(3, 7), (17, 31), (128, 256)])
@pytest.mark.parametrize("target", [(1, 1), (1, 19), (11, 1), (11, 19), (224, 224)])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_reader_nearest_resize_matches_pixels_layout_and_conversion(monkeypatch, source, target, dtype):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transforms

    image = torch.randint(0, 256, (*source, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(127)).permute(2, 0, 1)
    monkeypatch.setattr(data_io, "decode_image", lambda *args, **kwargs: image)
    reader = data_io.make_read_and_resize_fn((target[1], target[0]), torch.device("cpu"), dtype)
    expected = transforms.resize(image, list(target), interpolation=InterpolationMode.NEAREST)
    if dtype != torch.uint8:
        expected = data_io.make_convert_dtype(dtype)(expected)
    torch.testing.assert_close(reader("test-image"), expected, rtol=0, atol=0, check_stride=True)


def test_reader_identity_and_non_nearest_paths_remain_unchanged(monkeypatch):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as transforms

    image = torch.randint(0, 256, (7, 11, 3), dtype=torch.uint8).permute(2, 0, 1)
    monkeypatch.setattr(data_io, "decode_image", lambda *args, **kwargs: image)
    identity = data_io.make_read_and_resize_fn((11, 7), torch.device("cpu"), torch.uint8)
    assert identity("test-image") is image
    bilinear = data_io.make_read_and_resize_fn((19, 5), torch.device("cpu"), torch.uint8, interpolation=Image.Resampling.BILINEAR)
    expected = transforms.resize(image, [5, 19], interpolation=InterpolationMode.BILINEAR)
    torch.testing.assert_close(bilinear("test-image"), expected, rtol=0, atol=0, check_stride=True)
    wide = data_io.make_read_and_resize_fn((4097, 1), torch.device("cpu"), torch.uint8)
    expected = transforms.resize(image, [1, 4097], interpolation=InterpolationMode.NEAREST)
    torch.testing.assert_close(wide("test-image"), expected, rtol=0, atol=0, check_stride=True)


@pytest.fixture
def metadata(tmp_path):
    paths = []
    for index in range(5):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (8, 8), (index, index, index)).save(path)
        paths.append(str(path))
    return {"path": paths, "class": list(range(5))}


@pytest.mark.parametrize("size", [4, (4, 6), [4, 6]])
def test_loader_shapes_sampling_and_subsampling(metadata, size):
    datasets, loaders = get_dataset_dataloader(metadata, metadata, resize_size=size, batch_size=2, num_workers=0, cache="none", subsample=2)
    train, val = loaders
    assert isinstance(train.batch_sampler.sampler, RandomSampler)
    assert isinstance(val.batch_sampler.sampler, SequentialSampler)
    assert train.batch_sampler.drop_last and not val.batch_sampler.drop_last
    assert len(train) == 1 and len(val) == 2
    assert not train.persistent_workers
    images, labels = next(iter(val))
    # Existing resize tuples are (width, height).
    assert images.shape == (2, 3, 4 if isinstance(size, int) else 6, 4)
    assert images.dtype == torch.uint8
    assert labels.dtype == torch.long and labels.device.type == "cpu"
    assert labels.tolist() == [0, 2]
    assert len(datasets[0]) == 3
    dataset, inference = get_inference_dataloader(metadata["path"], resize_size=size, batch_size=2, num_workers=0, subsample=2)
    assert len(dataset) == 3
    assert not inference.pin_memory and not inference.persistent_workers
    assert isinstance(inference.batch_sampler.sampler, SequentialSampler)
    torch.testing.assert_close(next(iter(inference)), images)


@pytest.mark.parametrize("size", [None, 1.5, "4", (4,), (4, "6"), (4, 6, 8)])
def test_invalid_resize_errors(size):
    message = f"Invalid resize size passed, found {size}, but expected an integer or a tuple of two integers"
    with pytest.raises(TypeError) as error:
        get_dataset_dataloader(resize_size=size)
    assert str(error.value) == message + "."
    with pytest.raises(TypeError) as error:
        get_inference_dataloader([], resize_size=size)
    assert str(error.value) == message


@pytest.mark.parametrize("label", [2, [2, 3], (2, 3), np.array([2, 3]), torch.tensor([2, 3])])
@pytest.mark.parametrize("multilabel", [False, True])
def test_label_processing_and_hook(label, multilabel):
    image = torch.zeros(3, 4, 4, dtype=torch.uint8)
    processor = PathLabelProcessor(lambda _: image, lambda value: value + 1, multilabel)
    result, target = processor(("unused", label))
    expected = torch.as_tensor(label).long()
    if not multilabel and expected.numel() > 1:
        expected = expected[0]
    torch.testing.assert_close(target, expected)
    assert target.device.type == "cpu"
    torch.testing.assert_close(result, image + 1)
    if isinstance(label, torch.Tensor):
        target.fill_(99)
        assert label.tolist() == [2, 3]


@pytest.mark.parametrize("available,expected", [(1, 0), (4, 0), (7, 2), (8, 4), (64, 16)])
def test_automatic_training_workers_respect_affinity(metadata, monkeypatch, available, expected):
    monkeypatch.setattr(_workers.os, "cpu_count", lambda: 256)
    monkeypatch.setattr(_workers.os, "process_cpu_count", lambda: 256, raising=False)
    monkeypatch.setattr(_workers.os, "sched_getaffinity", lambda _: set(range(available)), raising=False)
    _, loaders = get_dataset_dataloader(metadata, resize_size=4, modes=("train",), cache="none")
    assert loaders[0].num_workers == expected
    assert loaders[0].persistent_workers == (expected > 0)


@pytest.mark.parametrize("process_count,affinity_count,expected", [(6, 128, 2), (128, 8, 4), (80, 80, 32)])
def test_inference_uses_smaller_process_limit(metadata, monkeypatch, process_count, affinity_count, expected):
    monkeypatch.setattr(_workers.os, "process_cpu_count", lambda: process_count, raising=False)
    monkeypatch.setattr(_workers.os, "sched_getaffinity", lambda _: set(range(affinity_count)), raising=False)
    _, loader = get_inference_dataloader(metadata["path"], resize_size=4)
    assert loader.num_workers == expected


@pytest.mark.parametrize("host_count", [None, 1, 8])
def test_worker_detection_fallbacks(monkeypatch, host_count):
    monkeypatch.delattr(_workers.os, "process_cpu_count", raising=False)
    monkeypatch.delattr(_workers.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(_workers.os, "cpu_count", lambda: host_count)
    assert _workers._available_cpu_count() == (host_count or 0)


def test_worker_detection_failed_affinity_and_unknown_process_count(monkeypatch):
    def unavailable(_):
        raise OSError("Affinity unavailable")

    monkeypatch.setattr(_workers.os, "process_cpu_count", lambda: None, raising=False)
    monkeypatch.setattr(_workers.os, "sched_getaffinity", unavailable, raising=False)
    monkeypatch.setattr(_workers.os, "cpu_count", lambda: 8)
    assert _workers._available_cpu_count() == 8


@pytest.mark.parametrize("workers", [0, 1, 3])
def test_explicit_worker_counts_are_preserved(metadata, monkeypatch, workers):
    def unexpected_detection():
        pytest.fail("Explicit worker selection must not trigger automatic detection")

    monkeypatch.setattr(_workers, "_available_cpu_count", unexpected_detection)
    _, loaders = get_dataset_dataloader(metadata, resize_size=4, modes=("train",), cache="none", num_workers=workers)
    _, inference = get_inference_dataloader(metadata["path"], resize_size=4, num_workers=workers)
    assert loaders[0].num_workers == inference.num_workers == workers


@pytest.mark.parametrize("workers", [None, 3])
def test_cuda_cache_still_disables_loader_workers(metadata, monkeypatch, workers):
    # Exercise loader configuration without allocating GPU storage.
    monkeypatch.setattr(data_loader, "LazyDataset", lambda **_: torch.utils.data.TensorDataset(torch.zeros(5, 1)))
    _, loaders = get_dataset_dataloader(metadata, resize_size=4, modes=("train",), cache="cuda", num_workers=workers)
    assert loaders[0].num_workers == 0
    assert not loaders[0].pin_memory
    assert not loaders[0].persistent_workers


@pytest.mark.parametrize("available,expected", [(0, 0), (1, 0), (2, 0), (4, 0), (8, 4), (256, 16)])
def test_ram_cache_thread_budget_and_contents(monkeypatch, available, expected):
    monkeypatch.setattr(_workers, "_available_cpu_count", lambda: available)
    executor = data_io.ThreadPoolExecutor
    selected = []

    def capture_executor(**kwargs):
        selected.append(kwargs["max_workers"])
        return executor(**kwargs)

    monkeypatch.setattr(data_io, "ThreadPoolExecutor", capture_executor)
    dataset = data_io.LazyDataset(lambda item: torch.tensor([item[0]]), (list(range(33)),), cache="cpu")
    assert selected == ([expected] if expected else [])
    torch.testing.assert_close(dataset[:], torch.arange(33).reshape(33, 1))


def test_distributed_loader_retains_spawn_and_sampler(monkeypatch):
    monkeypatch.setattr(data_loader, "is_dist_avail_and_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    dataset = torch.utils.data.TensorDataset(torch.arange(8))
    loader = data_loader.get_dataloader(dataset, "train", 2, 1, False, torch.device("cpu"))
    assert loader.multiprocessing_context.get_start_method() == "spawn"
    assert isinstance(loader.batch_sampler.sampler, torch.utils.data.DistributedSampler)
    assert loader.batch_sampler.sampler.num_replicas == 2
    assert loader.batch_sampler.drop_last


@pytest.mark.parametrize("cache", ["none", "cpu"])
@pytest.mark.parametrize("workers", [0, 1])
def test_batched_fetch_matches_default_collation(metadata, cache, workers):
    dataset = data_io.LazyDataset(
        PathLabelProcessor(data_io.make_read_and_resize_fn((4, 4), torch.device("cpu"), torch.uint8), None, False),
        (metadata["path"], metadata["class"]),
        cache=cache,
    )
    reference = torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=workers, multiprocessing_context="spawn" if workers else None
    )
    optimized = data_loader.get_dataloader(
        dataset, "val", 2, workers, False, torch.device("cpu"), multiprocessing_context="spawn", prefetch_factor=1
    )
    for expected, actual in zip(reference, optimized, strict=True):
        assert isinstance(actual, list) and len(actual) == 2
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if workers == 0:
        batch = dataset.__getitems__([4, 1, 1])
        direct = data_loader._collate_batch(batch)
        assert direct[0] is batch.data[0]
        assert direct[1].tolist() == [4, 1, 1]


def test_inference_batched_fetch_keeps_tensor_output(metadata):
    dataset, loader = get_inference_dataloader(metadata["path"], resize_size=4, batch_size=2, num_workers=0)
    batches = list(loader)
    assert all(isinstance(batch, torch.Tensor) for batch in batches)
    torch.testing.assert_close(torch.cat(batches), dataset[list(range(5))], rtol=0, atol=0)
    external = torch.utils.data.DataLoader(dataset, batch_size=2)
    torch.testing.assert_close(torch.cat(list(external)), torch.cat(batches), rtol=0, atol=0)


@pytest.mark.parametrize("cache", ["none", "cpu"])
def test_cuda_transfer_batches_are_pinned(metadata, cache):
    import os

    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 to validate pinned CUDA transfer batches")
    if not torch.cuda.is_available():
        pytest.fail("CUDA checks requested but no CUDA device is accessible")
    device = torch.device("cuda:0")
    _, loaders = get_dataset_dataloader(metadata, resize_size=4, modes=("val",), cache=cache, batch_size=2, num_workers=0, device=device)
    images, labels = next(iter(loaders[0]))
    assert images.is_pinned() and labels.is_pinned()
    torch.testing.assert_close(images.to(device, non_blocking=True).cpu(), images)
    _, inference = get_inference_dataloader(metadata["path"], resize_size=4, batch_size=2, num_workers=0, device=device)
    assert next(iter(inference)).is_pinned()


def test_cache_worker_override_reaches_dataset(metadata, monkeypatch):
    def unexpected_pool(*args, **kwargs):
        pytest.fail("cache_workers=0 must bypass the reader thread pool")

    monkeypatch.setattr(data_io, "ThreadPoolExecutor", unexpected_pool)
    _, loaders = get_dataset_dataloader(metadata, resize_size=4, modes=("val",), cache="cpu", cache_workers=0, num_workers=0, batch_size=5)
    _, labels = next(iter(loaders[0]))
    assert labels.tolist() == list(range(5))


def test_bounded_cuda_cache_matches_cpu_cache(metadata):
    import os

    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 to validate CUDA cache construction")
    if not torch.cuda.is_available():
        pytest.fail("CUDA checks requested but no CUDA device is accessible")
    device = torch.device("cuda:0")
    _, cpu_loaders = get_dataset_dataloader(
        metadata, resize_size=4, modes=("val",), cache="cpu", cache_workers=0, num_workers=0, batch_size=5
    )
    with pytest.warns(UserWarning, match="CUDA caching"):
        _, gpu_loaders = get_dataset_dataloader(
            metadata, resize_size=4, modes=("val",), cache="cuda", cache_workers=2, num_workers=3, batch_size=5, device=device
        )
    assert gpu_loaders[0].num_workers == 0
    expected = next(iter(cpu_loaders[0]))
    actual = next(iter(gpu_loaders[0]))
    for cpu, gpu in zip(expected, actual, strict=True):
        assert gpu.device == device
        torch.testing.assert_close(gpu.cpu(), cpu, rtol=0, atol=0)


def test_cuda_prefetch_rejects_cpu_target(metadata):
    with pytest.raises(ValueError, match="CUDA target"):
        get_inference_dataloader(metadata["path"], resize_size=4, num_workers=0, cuda_prefetch=True)


def _require_prefetch_cuda():
    import os

    if os.environ.get("RUN_CUDA_TESTS") != "1":
        pytest.skip("Set RUN_CUDA_TESTS=1 to verify CUDA transfer streams")
    assert torch.cuda.is_available()


@pytest.mark.parametrize("workers", [0, 1])
def test_cuda_prefetch_order_epochs_tail_and_lifetime(metadata, workers):
    _require_prefetch_cuda()
    from torch.utils.data import DataLoader

    datasets, loaders = get_dataset_dataloader(
        metadata,
        modes=("val",),
        resize_size=4,
        batch_size=2,
        num_workers=workers,
        device="cuda:0",
        cache="CPU",
        cache_workers=0,
        cuda_prefetch=True,
        multiprocessing_context="spawn" if workers else None,
    )
    loader = loaders[0]
    assert isinstance(loader, DataLoader)
    assert loader.dataset is datasets[0] and len(loader) == 3
    consumer = torch.cuda.Stream()
    for _ in range(2):
        outputs = []
        with torch.cuda.stream(consumer):
            for images, labels in loader:
                assert images.device == labels.device == torch.device("cuda:0")
                assert images.dtype == torch.uint8 and labels.dtype == torch.long
                # Queue use then release the batch while the allocator may reuse
                # its copy-stream storage for subsequent batches.
                outputs.append((images.float().mean((1, 2, 3)), labels + 0))
                del images, labels
        consumer.synchronize()
        assert torch.cat([result[1] for result in outputs]).tolist() == list(range(5))
        torch.testing.assert_close(torch.cat([result[0] for result in outputs]).cpu(), torch.arange(5).float())
    iterator = iter(loader)
    next(iterator)
    del iterator
    assert sum(len(images) for images, _ in loader) == 5


def test_cuda_prefetch_inference_empty_and_failure(metadata):
    _require_prefetch_cuda()
    from mini_trainer.data._prefetch import CUDAPrefetchLoader

    _, loader = get_inference_dataloader(
        metadata["path"],
        resize_size=4,
        batch_size=2,
        num_workers=0,
        device="cuda:0",
        cuda_prefetch=True,
    )
    result = torch.cat(list(loader))
    torch.testing.assert_close(result.float().mean((1, 2, 3)).cpu(), torch.arange(5).float())
    empty = CUDAPrefetchLoader([], device="cuda:0", batch_size=2)
    assert list(empty) == []

    class Broken(torch.utils.data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            if index == 1:
                raise RuntimeError("broken sample")
            return torch.tensor(index)

    iterator = iter(CUDAPrefetchLoader(Broken(), device="cuda:0", batch_size=1, pin_memory=True))
    assert next(iterator).item() == 0
    with pytest.raises(RuntimeError, match="broken sample"):
        next(iterator)


def test_cuda_prefetch_nested_cuda_source():
    _require_prefetch_cuda()
    from mini_trainer.data._prefetch import CUDAPrefetchLoader

    class Mixed(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {"value": torch.ones(1024, device="cuda:0") * index, "label": index, "name": str(index)}

    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        outputs = list(CUDAPrefetchLoader(Mixed(), device="cuda:0", batch_size=2))
    consumer.synchronize()
    for index, output in enumerate(outputs):
        assert output["name"] == [str(2 * index), str(2 * index + 1)]
        torch.testing.assert_close(output["value"].mean(1), output["label"].float())


def test_pinned_cache_gather_storage_and_indices(metadata):
    _require_prefetch_cuda()
    datasets, loaders = get_dataset_dataloader(
        metadata,
        modes=("val",),
        resize_size=4,
        batch_size=2,
        num_workers=0,
        device="cuda:0",
        cache="CPU",
        cache_workers=0,
    )
    dataset = datasets[0]
    assert dataset._pin_batches
    fetched = dataset.__getitems__([4, 1, 1, -1])
    images, labels = data_loader._collate_batch(fetched)
    assert images.is_pinned() and labels.is_pinned()
    assert images.pin_memory().data_ptr() == images.data_ptr()
    assert labels.tolist() == [4, 1, 1, 4]
    before = dataset[1][0].clone()
    images[1].zero_()
    torch.testing.assert_close(dataset[1][0], before)
    retained = next(iter(loaders[0]))[0]
    copied = retained.clone()
    list(loaders[0])
    torch.testing.assert_close(retained, copied)


def test_pinned_cache_gather_never_pins_inside_worker(monkeypatch):
    dataset = data_io.LazyDataset(lambda item: torch.tensor(item[0]), ([0, 1, 2],), cache="CPU", cache_workers=0, pin_batches=True)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
    # CPU-only execution also proves this branch does not need a pin allocator.
    result = data_loader._collate_batch(dataset.__getitems__([2, 0]))
    assert not result.is_pinned()
    assert result.tolist() == [2, 0]


@pytest.mark.parametrize("labels", [False, True])
def test_cached_worker_batches_are_built_in_shared_storage(metadata, labels):
    if labels:
        datasets, loaders = get_dataset_dataloader(
            metadata, resize_size=4, modes=("val",), cache="CPU", cache_workers=0, batch_size=2, num_workers=0
        )
        dataset, base = datasets[0], loaders[0]
    else:
        uncached, _ = get_inference_dataloader(metadata["path"], resize_size=4, batch_size=2, num_workers=0)
        dataset = data_io.LazyDataset(uncached.func, uncached.items, cache="CPU", cache_workers=0)
        base = data_loader.get_dataloader(dataset, "val", 2, 0, False, torch.device("cpu"))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=base.batch_sampler,
        collate_fn=_assert_shared_worker_batch,
        num_workers=1,
        multiprocessing_context="spawn",
    )
    batches = list(loader)
    expected = [data_loader._collate_batch(dataset.__getitems__(indices)) for indices in base.batch_sampler]
    torch.testing.assert_close(batches, expected, rtol=0, atol=0)
    first = batches[0][0] if labels else batches[0]
    preserved = first.clone()
    last = batches[-1][0] if labels else batches[-1]
    last.zero_()
    torch.testing.assert_close(first, preserved, rtol=0, atol=0)


def test_external_collation_does_not_allocate_an_extra_shared_batch(monkeypatch):
    dataset = data_io.LazyDataset(lambda item: torch.tensor(item[0]), ([0, 1],), cache="CPU", cache_workers=0)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
    ordinary = dataset.__getitems__([0, 1]).data
    direct = dataset.__getitems__(data_io._DirectBatchIndices([0, 1])).data
    assert not ordinary.is_shared() and direct.is_shared()
    torch.testing.assert_close(ordinary, direct, rtol=0, atol=0)


def test_cached_repository_loader_does_not_unpack_sample_views():
    from torch.utils._python_dispatch import TorchDispatchMode

    images = torch.arange(96).reshape(8, 3, 2, 2)
    dataset = data_io.LazyDataset(lambda item: (images[item[0]], torch.tensor(item[0])), (list(range(8)),), cache="cpu", cache_workers=0)
    loader = data_loader.get_dataloader(dataset, "val", 4, 0, False, torch.device("cpu"))
    unpacked = []

    class ObserveUnpacking(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.unbind.int, torch.ops.aten.select.int):
                unpacked.append(func)
            return func(*args, **(kwargs or {}))

    with ObserveUnpacking():
        actual = list(loader)
    assert not unpacked
    assert loader.dataset is dataset
    torch.testing.assert_close(torch.cat([batch[0] for batch in actual]), images)
    assert torch.cat([batch[1] for batch in actual]).tolist() == list(range(8))


def test_direct_batches_preserve_shuffled_sampling_and_rng():
    dataset = data_io.LazyDataset(lambda item: torch.tensor(item[0]), (list(range(19)),), cache="cpu", cache_workers=0)
    optimized = data_loader.get_dataloader(dataset, "train", 4, 0, False, torch.device("cpu"))
    reference = torch.utils.data.DataLoader(
        dataset, batch_sampler=torch.utils.data.BatchSampler(RandomSampler(dataset), batch_size=4, drop_last=True)
    )
    torch.manual_seed(31)
    for _ in range(3):
        rng = torch.get_rng_state()
        expected = list(reference)
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        actual = list(optimized)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), expected_rng)


@pytest.mark.parametrize("cache", ["none", "cpu"])
@pytest.mark.parametrize("workers", [0, 1])
@pytest.mark.parametrize("labels", [False, True])
def test_default_collator_can_reuse_repository_batch_sampler(metadata, cache, workers, labels):
    if labels:
        datasets, loaders = get_dataset_dataloader(
            metadata, resize_size=4, modes=("val",), cache=cache, cache_workers=0, batch_size=2, num_workers=0
        )
        dataset, optimized = datasets[0], loaders[0]
    else:
        dataset, optimized = get_inference_dataloader(
            metadata["path"], resize_size=4, cache=cache, cache_workers=0, batch_size=2, num_workers=0
        )
    external = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=optimized.batch_sampler,
        num_workers=workers,
        multiprocessing_context="spawn" if workers else None,
    )
    for expected, actual in zip(optimized, external, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if not labels:
        # Ordinary direct batched fetches remain real Tensor lists, including
        # torch.stack's C-level sequence handling (which bypasses list methods).
        direct = dataset.__getitems__([4, 1, 1])
        torch.testing.assert_close(torch.stack(direct), dataset[[4, 1, 1]])
