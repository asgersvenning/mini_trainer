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


def test_random_affine_resize_draws_per_image_within_fastai_ranges():
    from mini_trainer.data import RandomAffineResize
    from mini_trainer.utils import affine_resample

    torch.manual_seed(0)
    image = torch.randint(1, 256, (1, 3, 20, 20), dtype=torch.uint8)
    batch = image.expand(16, 3, 20, 20).contiguous()
    out = RandomAffineResize(size=12, max_rotate=15.0)(batch)
    assert out.shape == (16, 3, 12, 12) and out.dtype == torch.float32
    assert len({o.numpy().tobytes() for o in out}) > 4  # identical inputs, independent transforms
    assert (out > 0).all() and out.max() <= 1  # reflection padding never samples zeros
    assert RandomAffineResize(size=12)(image[0]).shape == (3, 12, 12)
    # With nothing drawn the augmentation is the plain resize used for evaluation.
    plain = RandomAffineResize(size=12, flip_p=0, max_rotate=0, max_zoom=1.0)(batch)
    torch.testing.assert_close(plain, affine_resample(batch.float() / 255, None, 12), rtol=0, atol=0)
    matrices = RandomAffineResize(size=12, max_rotate=15.0).matrices(20000, torch.device("cpu"))
    singular = torch.linalg.svdvals(matrices[:, :2, :2])  # flips and rotations keep 1; zoom scales by 1/z, z in [1, 1.1]
    assert singular.min() >= 1 / 1.1 - 1e-6 and singular.max() <= 1 + 1e-6
    identity = ((matrices - torch.eye(3)).abs().amax((1, 2)) < 1e-6).float().mean().item()
    assert abs(identity - 0.5625 * 0.25 * 0.25) < 0.006  # dihedral identity (0.5 + 0.5 / 8), no rotation, no zoom


def test_domain_augmentation_presets_and_batch_semantics():
    from mini_trainer.data import RandomJPEGish, RandomLowLight, RandomMotionBlur, domain_augmentation

    assert domain_augmentation(None) == []
    assert [(type(t), t.p) for t in domain_augmentation("trap")] == [(RandomMotionBlur, 0.25), (RandomLowLight, 0.25), (RandomJPEGish, 0.2)]
    with pytest.raises(ValueError, match="Unknown domain augmentation"):
        domain_augmentation("fog")

    torch.manual_seed(0)
    batch = torch.rand(4, 3, 16, 16)
    for transform in (RandomMotionBlur, RandomLowLight, RandomJPEGish):
        assert transform(p=0)(batch) is batch  # skipped draws leave the batch untouched
        out = transform(p=1)(batch)
        assert out.shape == batch.shape and out.dtype == batch.dtype and not torch.equal(out, batch)
        assert out.min() >= batch.min() - 1e-6 and out.max() <= batch.max() + 1e-6
        assert transform(p=1)(batch[0]).shape == batch[0].shape
    # One draw per batch, as fastai's RandTransform: identical images stay identical (noise aside).
    same = batch[:1].expand(4, 3, 16, 16)
    for out in (RandomMotionBlur(p=1)(same), RandomJPEGish(p=1)(same)):
        assert (out == out[:1]).all()
    # A line blur keeps a constant image's interior and averages along one axis only.
    flat = torch.full((1, 1, 16, 16), 0.5)
    assert torch.allclose(RandomMotionBlur(p=1, max_kernel=3)(flat)[..., 2:-2, 2:-2], flat[..., 2:-2, 2:-2])
    levels = RandomJPEGish(p=1, min_levels=12)(torch.linspace(0, 1, 1000).view(1, 1, 10, 100)).unique().numel()
    assert 13 <= levels <= 64
