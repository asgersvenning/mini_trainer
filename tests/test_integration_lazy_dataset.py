import os

import numpy as np
import pytest
import torch
from PIL import Image

from mini_trainer.data import LazyDataset


def create_dummy_images(root_dir, n=10, size=(32, 32)):
    os.makedirs(root_dir, exist_ok=True)
    paths = []
    for i in range(n):
        path = os.path.join(root_dir, f"img_{i}.png")
        # Create random image
        img_np = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
        img = Image.fromarray(img_np)
        img.save(path)
        paths.append(path)
    return paths


def dummy_loader(path):
    img = Image.open(path).convert("RGB")
    return torch.from_numpy(np.array(img)).permute(2, 0, 1)


def failing_loader(path):
    if "fail" in path:
        raise ValueError("Simulated read failure")
    return torch.randn(3, 32, 32)


def dummy_loader_tuple(path):
    img = Image.open(path).convert("RGB")
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
        assert len(ds) == len(image_paths)
        item = ds[0]
        assert torch.is_tensor(item)
        assert item.shape == (3, 32, 32)

        items = ds[0:2]
        assert torch.is_tensor(items)
        assert items.shape == (2, 3, 32, 32)

    def test_lazy_dataset_cpu_cache(self, image_paths):
        ds = LazyDataset(dummy_loader, image_paths, cache="cpu")
        assert len(ds) == len(image_paths)

        assert hasattr(ds, "_ram_cache")
        assert len(ds._ram_cache) == len(image_paths)

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
            LazyDataset(failing_loader, paths, cache="cpu")
