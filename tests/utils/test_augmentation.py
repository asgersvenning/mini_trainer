import torch

from mini_trainer.utils.augmentation import SaltAndPepper, salt_and_pepper


def test_debug_augmentation(tmp_path):
    from mini_trainer.utils.augmentation import debug_augmentation

    # Mock dataset
    class MockDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 5

        def __getitem__(self, idx):
            return torch.zeros((3, 10, 10)), 0

    ds = MockDataset()

    def aug(x):
        return x

    ret = debug_augmentation(aug, ds, output_dir=str(tmp_path), strict=True)
    assert ret is True
    assert (tmp_path / "example_augmentation.png").exists()


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
