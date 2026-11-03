import numpy as np
import pytest
import torch
from PIL import Image

from mini_trainer.hierarchical.model import IndependentClassifier
from mini_trainer.modeling import Classifier
from mini_trainer.modeling.architectures import ResampleNormalize
from mini_trainer.utils import affine_resample


def _scale_and_bias(head):
    layers = getattr(head, "layers", [head.linear])
    return [p for layer in layers for p in (layer.bias, layer.parametrizations.weight.original0)]


def test_trainable_scale_and_bias_are_opt_in_for_every_level():
    masks = [torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([0, 0, 1])]
    default = IndependentClassifier(in_features=8, out_features=6, sparse_masks=masks)
    assert not any(p.requires_grad for p in _scale_and_bias(default))
    assert "trainable_scale" not in default.metadata and "trainable_bias" not in default.metadata

    torch.manual_seed(0)
    head = IndependentClassifier(in_features=8, out_features=6, sparse_masks=masks, trainable_scale=True, trainable_bias=True)
    params = _scale_and_bias(head)
    assert len(params) == 6 and all(p.requires_grad for p in params)
    assert head.metadata["trainable_scale"] is True and head.metadata["trainable_bias"] is True
    before = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=0.1)
    targets = torch.stack([torch.randint(0, n, (16,)) for n in (6, 3, 2)], 1)
    logits = head(torch.randn(16, 8))
    sum(torch.nn.functional.cross_entropy(lvl, targets[:, i]) for i, lvl in enumerate(logits)).backward()
    optimizer.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, params))


def test_float32_head_runs_outside_autocast():
    torch.manual_seed(0)
    head = Classifier(in_features=8, out_features=5, float32_head=True).eval()
    x = torch.randn(4, 8).bfloat16()
    with torch.no_grad():
        reference = head(x.float())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            guarded = head(x)
            head.float32_head = False
            unguarded = head(x)
    assert guarded.dtype == torch.float32
    torch.testing.assert_close(guarded, reference, rtol=0, atol=0)
    assert not torch.equal(unguarded.float(), reference)  # autocast was active for the unguarded call


def test_legacy_spherical_init_reproduces_the_original_step():
    def original(num_classes, features, seed, lr=0.5):  # Classifier.init_spherical_repulsion before d1a9cc9
        torch.manual_seed(seed)
        w = torch.nn.init.normal_(torch.empty(num_classes, features))
        for _ in range(100):
            w.div_(w.norm(dim=1, keepdim=True).clamp(min=1e-9))
            grad = w @ w.t() @ w
            proj = (grad * w).sum(dim=1, keepdim=True) * w
            w.sub_(lr * (grad - proj))
        return w.div_(w.norm(dim=1, keepdim=True).clamp(min=1e-9))

    layer = torch.nn.Linear(8, 6)
    torch.manual_seed(1)
    Classifier.init_spherical_repulsion(layer, per_class_step=False)
    torch.testing.assert_close(layer.weight.detach(), original(6, 8, 1), rtol=0, atol=0)
    torch.manual_seed(1)
    Classifier.init_spherical_repulsion(layer)
    assert not torch.equal(layer.weight.detach(), original(6, 8, 1))


def test_resample_normalize_accepts_read_and_model_sizes_only():
    torch.manual_seed(0)
    preprocess = ResampleNormalize(size=8, read_size=12)
    read = torch.randint(0, 256, (2, 3, 12, 12), dtype=torch.uint8)
    mean, std = torch.tensor(preprocess.mean).view(1, 3, 1, 1), torch.tensor(preprocess.std).view(1, 3, 1, 1)
    expected = (affine_resample(read.float() / 255, None, 8) - mean) / std
    torch.testing.assert_close(preprocess(read), expected, rtol=0, atol=0)
    torch.testing.assert_close(preprocess(read.float() / 255), expected, rtol=0, atol=0)
    augmented = torch.rand(2, 3, 8, 8)
    torch.testing.assert_close(preprocess(augmented), (augmented - mean) / std, rtol=0, atol=0)
    assert preprocess(read[0]).shape == (3, 8, 8)
    with pytest.raises(ValueError, match="Expected images of 12x12"):
        preprocess(torch.rand(2, 3, 10, 10))


def test_head_options_and_square_crop_preprocessing_survive_reload(tmp_path):
    from tests.integration.test_integration_train import TinyMockModel

    options = {"trainable_scale": True, "trainable_bias": True, "float32_head": True, "legacy_spherical_init": True}
    model, preprocess = Classifier.build(
        model_type=TinyMockModel(),
        num_classes=3,
        resize_size=8,
        resize_method="crop",
        read_size=12,
        model_args={"pretrained": False},
        **options,
    )
    assert isinstance(preprocess.transform, ResampleNormalize) and (preprocess.transform.size, preprocess.transform.read_size) == (8, 12)
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    reloaded, reloaded_preprocess = Classifier.build(weights=str(tmp_path / "weights.pt"))
    head = reloaded.fc
    assert all(getattr(head, name) for name in options) and all(p.requires_grad for p in _scale_and_bias(head))
    assert head.metadata["resize_method"] == "crop" and head.metadata["read_size"] == 12
    image = np.random.default_rng(0).integers(0, 256, (10, 20, 3), dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "image.png")
    from_path = reloaded_preprocess(str(tmp_path / "image.png"))  # read as a centred square at read_size
    centred = torch.from_numpy(np.array(Image.fromarray(image).crop((5, 0, 15, 10)).resize((12, 12), Image.Resampling.BILINEAR)))
    torch.testing.assert_close(from_path, reloaded_preprocess(centred.permute(2, 0, 1)), rtol=0, atol=0)
