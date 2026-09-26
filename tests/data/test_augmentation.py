import pytest
import torch
from matplotlib import pyplot as plt

from mini_trainer.data import SaltAndPepper, debug_augmentation, salt_and_pepper


@pytest.mark.parametrize("count", [1, 2, 5])
def test_debug_augmentation(tmp_path, count):
    dataset = torch.utils.data.TensorDataset(torch.zeros(count, 3, 10, 10), torch.arange(count))
    caller_figure = plt.figure()
    before = plt.get_fignums()
    try:
        assert debug_augmentation(lambda image: image, dataset, output_dir=str(tmp_path)) is True
        assert (tmp_path / "example_augmentation.png").exists()
        assert plt.get_fignums() == before
    finally:
        plt.close(caller_figure)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("failure", ["empty", "augmentation"])
def test_debug_augmentation_failure_preserves_caller_figure(tmp_path, strict, failure):
    dataset = torch.utils.data.TensorDataset(torch.zeros(0 if failure == "empty" else 3, 3, 10, 10))

    def broken(image):
        raise ValueError("broken augmentation")

    caller_figure = plt.figure()
    before = plt.get_fignums()
    try:
        if strict:
            with pytest.raises(ValueError):
                debug_augmentation(broken, dataset, str(tmp_path), strict=True)
        else:
            with pytest.warns(UserWarning, match="debug augmentation"):
                assert debug_augmentation(broken, dataset, str(tmp_path), strict=False) is False
        assert plt.get_fignums() == before
        assert not (tmp_path / "example_augmentation.png").exists()
    finally:
        plt.close(caller_figure)


def test_salt_and_pepper():
    # Create a small image
    img = torch.zeros((3, 10, 10), dtype=torch.float32)

    # Apply salt and pepper with high probability and proportion range
    aug_img = salt_and_pepper(img, proportion=(0.5, 0.5), probability=1.0)

    # Check that some pixels have changed
    assert not torch.allclose(img, aug_img)

    unique_vals = torch.unique(aug_img)
    # Ideally should be only 0.0 and 1.0, but original image was 0.0 so we might just see 0 and 1.
    for val in unique_vals:
        assert val.item() in [0.0, 1.0]


def test_salt_and_pepper_module():
    mod = SaltAndPepper(proportion=(0.5, 0.5), probability=1.0)
    img = torch.zeros((3, 10, 10), dtype=torch.float32)
    aug_img = mod(img)
    assert not torch.allclose(img, aug_img)
    assert "SaltAndPepper" in repr(mod)
