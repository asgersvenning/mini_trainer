import os

import numpy as np
import pytest
import torch
from PIL import Image

from mini_trainer.data import LazyDataset


def create_dummy_images(root_dir, n=10, size=(32, 32)):
    os.makedirs(root_dir, exist_ok=True)
    paths: list[str] = []
    for i in range(n):
        path = os.path.join(root_dir, f"img_{i}.png")
        # Create random image
        img_np = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
        img = Image.fromarray(img_np)
        img.save(path)
        paths.append(path)
    return (paths,)


def dummy_loader(path: tuple[str]):
    assert len(path) == 1
    img = Image.open(path[0]).convert("RGB")
    return torch.from_numpy(np.array(img)).permute(2, 0, 1)


def failing_loader(path: tuple[str]):
    assert len(path) == 1
    if "fail" in path[0]:
        raise ValueError("Simulated read failure")
    return torch.randn(3, 32, 32)


def dummy_loader_tuple(path: tuple[str]):
    assert len(path) == 1
    img = Image.open(path[0]).convert("RGB")
    t_img = torch.from_numpy(np.array(img)).permute(2, 0, 1)
    label = torch.tensor([1])
    return t_img, label


class TestLazyDatasetIntegration:
    @pytest.fixture
    def image_paths(self, tmp_path):
        data_dir = tmp_path / "data"
        return create_dummy_images(str(data_dir))

    def test_lazy_dataset_none_cache(self, image_paths):
        ds = LazyDataset(dummy_loader, image_paths, cache=None)
        assert len(ds) == len(image_paths[0])
        item = ds[0]
        assert torch.is_tensor(item)
        assert item.shape == (3, 32, 32)

        items = ds[0:2]
        assert torch.is_tensor(items)
        assert items.shape == (2, 3, 32, 32)

    def test_lazy_dataset_cpu_cache(self, image_paths):
        ds = LazyDataset(dummy_loader, image_paths, cache="cpu")
        assert len(ds) == len(image_paths[0])

        assert hasattr(ds, "_ram_cache")
        assert len(ds._ram_cache) == len(image_paths[0])

        item = ds[0]
        assert torch.is_tensor(item)
        assert item.shape == (3, 32, 32)

    def test_lazy_dataset_tuple_return(self, image_paths):
        ds = LazyDataset(dummy_loader_tuple, image_paths, cache="cpu")
        item = ds[0]
        assert isinstance(item, (list, tuple))
        assert len(item) == 2

    def test_lazy_dataset_picklable(self, image_paths):
        import pickle

        from mini_trainer.data import get_dataset_dataloader

        metadata = {
            "path": image_paths,
            "class": [0 if i % 2 == 0 else 1 for i in range(len(image_paths))],
            "split": ["train" for _ in image_paths],
        }

        datasets, loaders = get_dataset_dataloader(
            metadata,
            resize_size=16,
            modes=("train",),
            batch_size=2,
            num_workers=0,
            cache=None,
        )
        ds = datasets[0]

        pickled = pickle.dumps(ds)
        unpickled = pickle.loads(pickled)

        assert len(unpickled) == len(ds)
        item = unpickled[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_lazy_dataset_caching_exception_propagation(self):
        paths = ["ok1.png", "fail.png", "ok2.png"]
        with pytest.raises(ValueError, match="Simulated read failure"):
            LazyDataset(failing_loader, (paths,), cache="cpu")


@pytest.mark.parametrize("workers", [0, 1, 3])
@pytest.mark.parametrize("structured", [False, True])
def test_cache_decodes_once_and_preserves_order(workers, structured):
    from collections import Counter
    from threading import Lock

    calls, lock = Counter(), Lock()

    def reader(item):
        index = int(item[0])
        with lock:
            calls[index] += 1
        image = torch.full((2, 3), index, dtype=torch.uint8)
        return (image, torch.tensor(index)) if structured else image

    dataset = LazyDataset(reader, (range(37),), cache="cpu", cache_workers=workers)
    assert calls == Counter(range(37))
    for index in range(37):
        result = dataset[index]
        image = result[0] if structured else result
        assert torch.equal(image, torch.full((2, 3), index, dtype=torch.uint8))
        if structured:
            assert result[1].item() == index


def test_cache_read_ahead_is_bounded(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Thread

    from mini_trainer.data import io

    release, window_ready = Event(), Event()
    submitted, outcome = [], []

    class RecordingPool(ThreadPoolExecutor):
        def submit(self, func, item):
            submitted.append(int(item[0]))
            result = super().submit(func, item)
            if len(submitted) == 4:
                window_ready.set()
            return result

    monkeypatch.setattr(io, "ThreadPoolExecutor", RecordingPool)

    def reader(item):
        if item[0] == 1:
            assert release.wait(10), "Test failed to release slow reader"
        return torch.tensor(item[0])

    def construct():
        try:
            outcome.append(LazyDataset(reader, (range(100),), cache="cpu", cache_workers=2))
        except Exception as error:
            outcome.append(error)

    constructor = Thread(target=construct)
    constructor.start()
    try:
        assert window_ready.wait(10)
        assert submitted == [1, 2, 3, 4]
    finally:
        release.set()
        constructor.join(10)
    assert not constructor.is_alive()
    assert len(outcome) == 1 and isinstance(outcome[0], LazyDataset)
    assert torch.equal(outcome[0][:], torch.arange(100))


@pytest.mark.parametrize("failure", ["read", "shape", "arity"])
@pytest.mark.parametrize("workers", [0, 2])
def test_cache_failure_shuts_down_readers(failure, workers):
    import threading

    before = set(threading.enumerate())

    def reader(item):
        if item[0] == 3:
            if failure == "read":
                raise ValueError("broken reader")
            if failure == "shape":
                return (torch.zeros(5),)
            return (torch.zeros(2), torch.zeros(2))
        return (torch.zeros(2),)

    with pytest.raises((ValueError, RuntimeError)):
        LazyDataset(reader, (range(100),), cache="cpu", cache_workers=workers)
    assert not [thread for thread in threading.enumerate() if thread not in before and thread.name.startswith("mini-trainer-cache")]


def test_cache_automatic_workers_respect_small_cpu_budget(monkeypatch):
    from mini_trainer.data import _workers, io

    monkeypatch.setattr(_workers, "_available_cpu_count", lambda: 4)

    def unexpected_pool(*args, **kwargs):
        pytest.fail("Four available CPUs should not start cache reader threads")

    monkeypatch.setattr(io, "ThreadPoolExecutor", unexpected_pool)
    dataset = LazyDataset(lambda item: torch.tensor(item[0]), (range(10),), cache="cpu")
    assert torch.equal(dataset[:], torch.arange(10))


@pytest.mark.parametrize("workers", [-1, True, 1.5])
def test_cache_invalid_worker_count(workers):
    with pytest.raises(ValueError, match="cache_workers"):
        LazyDataset(lambda item: torch.tensor(0), ([0],), cache="cpu", cache_workers=workers)


def test_cache_rejects_incomplete_input_columns():
    with pytest.raises(ValueError, match="equal lengths"):
        LazyDataset(lambda item: torch.tensor(item[0]), (range(3), range(2)), cache="cpu")
