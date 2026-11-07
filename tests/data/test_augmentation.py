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


def _rotate_pad(module, images, degrees, pad):
    """RandomRotatePad's resampling at fixed angles and pads (its forward draws them)."""
    n, _, height, width = images.shape
    grid = module.grid(torch.full((n,), float(degrees)), torch.full((n,), float(pad)), height, width)
    fill = torch.tensor(module.fill, dtype=torch.float32).view(1, 3, 1, 1) / 255
    return (
        torch.nn.functional.grid_sample(images.float() / 255 - fill, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        + fill
    )


def test_random_rotate_pad_matches_pil_rotation_on_an_expanded_canvas_and_edge_padding():
    import numpy as np
    from PIL import Image

    from mini_trainer.data import RandomRotatePad

    module = RandomRotatePad(fill=(255, 0, 0))
    torch.manual_seed(0)
    images = torch.randint(0, 200, (2, 3, 10, 10), dtype=torch.uint8)
    images[:, 0] = 0  # no red in the images, so any red output pixel is fill
    torch.testing.assert_close(_rotate_pad(module, images, 0, 0), images.float() / 255, atol=1e-6, rtol=0)
    # A quarter turn of a square: PIL's counter-clockwise rotation with expand=True, exactly.
    pil = np.stack([np.asarray(Image.fromarray(im.permute(1, 2, 0).numpy()).rotate(90, expand=True)) for im in images])
    torch.testing.assert_close(
        _rotate_pad(module, images, 90, 0), torch.from_numpy(pil).permute(0, 3, 1, 2).float() / 255, atol=1e-5, rtol=0
    )
    # Edge padding repeats the border (no fill colour) and shrinks the image to the centre.
    padded = _rotate_pad(module, images, 0, 0.25)
    assert padded[:, 0].max() < 1e-6
    torch.testing.assert_close(padded[..., 0, 0], images[..., 0, 0].float() / 255, atol=1e-6, rtol=0)
    # At 45 degrees the expanded canvas's corners are fill.
    rotated = _rotate_pad(module, images, 45, 0)
    assert torch.allclose(rotated[:, :, 0, 0], torch.tensor([1.0, 0.0, 0.0]).expand(2, 3), atol=1e-6)


def test_random_rotate_pad_draws_per_image_and_keeps_shape():
    from mini_trainer.data import RandomRotatePad

    torch.manual_seed(0)
    image = torch.randint(0, 256, (3, 16, 12), dtype=torch.uint8)
    batch = image.expand(6, -1, -1, -1).clone()
    out = RandomRotatePad()(batch)
    assert out.shape == batch.shape and out.dtype == torch.float32 and 0 <= out.min() and out.max() <= 1
    assert len({tuple(o.flatten()[:64].tolist()) for o in out}) == 6  # every image its own angle and pad
    assert RandomRotatePad()(image).shape == image.shape  # single images, as debug_augmentation draws
    torch.testing.assert_close(RandomRotatePad(p=0)(batch), batch.float() / 255, atol=1e-6, rtol=0)
    with pytest.raises(ValueError):
        RandomRotatePad(max_degrees=-1)


def test_trivial_augment_wide_sample_augmentation_and_the_augmentation_spec():
    from mini_trainer.data import (
        RandomJPEGish,
        RandomLowLight,
        RandomMotionBlur,
        RandomRotatePad,
        sample_augmentation,
        split_domain_augmentation,
    )

    assert sample_augmentation(None) is None
    transform = sample_augmentation("trivialwide")
    torch.manual_seed(0)
    image = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)
    outs = [transform(image) for _ in range(8)]
    assert all(o.shape == image.shape and o.dtype == torch.uint8 for o in outs)
    assert len({o.numpy().tobytes() for o in outs}) > 1  # a new draw on every read
    with pytest.raises(ValueError, match="Unknown sample augmentation"):
        sample_augmentation("nope")

    assert split_domain_augmentation(None) == ([], [])
    before, after = split_domain_augmentation("geometric+trap")
    assert [type(t) for t in before] == [RandomRotatePad]
    assert [type(t) for t in after] == [RandomMotionBlur, RandomLowLight, RandomJPEGish]
    with pytest.raises(ValueError, match="Unknown domain augmentation"):
        split_domain_augmentation("geometric+nope")
