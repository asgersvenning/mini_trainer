
import os
import shutil
import torch
import numpy as np
from PIL import Image
from mini_trainer.utils.io import LazyDataset, CACHE_MODE
import pytest

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
    # Retrieve the image directly using PIL to verify content
    img = Image.open(path).convert("RGB")
    return torch.from_numpy(np.array(img)).permute(2, 0, 1)

def dummy_loader_tuple(path):
    # Returns (img, label_dummy_tensor)
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
        # Cache = None
        ds = LazyDataset(dummy_loader, image_paths, cache=None)
        assert len(ds) == len(image_paths)
        item = ds[0]
        # Wait, LazyDataset.__getitem__ returns func(item) if caching is None and index is int.
        # Check source: case CACHE_MODE.NONE: if isinstance(i, int): return self.func(...)
        
        # dummy_loader returns tensor.
        assert torch.is_tensor(item)
        assert item.shape == (3, 32, 32)
        
        # Test slice
        items = ds[0:2]
        # Slice return: torch.stack(elements) if elements are tensors (our fix)
        # elements = [tensor, tensor] -> stack -> stacked_tensor
        assert torch.is_tensor(items)
        assert items.shape == (2, 3, 32, 32)

    def test_lazy_dataset_cpu_cache(self, image_paths):
        # Cache = "cpu"
        ds = LazyDataset(dummy_loader, image_paths, cache="cpu")
        assert len(ds) == len(image_paths)
        
        # _ram_cache is initialized
        assert hasattr(ds, "_ram_cache")
        assert len(ds._ram_cache) == len(image_paths)
        
        item = ds[0]
        # In CPU/CUDA mode: return data[0] if _ram_was_single_tensor else data
        # dummy_loader returns tensor -> _ram_was_single_tensor = True
        assert torch.is_tensor(item)
        assert item.shape == (3, 32, 32)
        
    def test_lazy_dataset_disk_cache(self, image_paths, tmp_path):
        # Cache = "disk"
        # Since LazyDataset uses hashlib for unique cache path, we should verify it creates files.
        # However, LazyDataset hardcodes cache dir to os.path.join(gettempdir(), ".mini_trainer")
        # We might want to mock gettempdir or just check if it works.
        
        ds = LazyDataset(dummy_loader, image_paths, cache="disk")
        assert len(ds) == len(image_paths)
        
        item = ds[0]
        # Zarr loading
        assert torch.is_tensor(item)
        assert item.shape == (3, 32, 32)

        # Force reload from disk to verify persistence
        # (Re-instantiating with same items should hit cache)
        # Note: LazyDataset computes hash from items.
        ds2 = LazyDataset(dummy_loader, image_paths, cache="disk")
        item2 = ds2[0]
        assert torch.equal(item, item2)
        
    def test_lazy_dataset_tuple_return(self, image_paths):
        # Test with loader returning tuple
        ds = LazyDataset(dummy_loader_tuple, image_paths, cache="cpu")
        item = ds[0]
        assert isinstance(item, (list, tuple))
        assert len(item) == 2


