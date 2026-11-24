"""Opt-in batch regularization: timm-style random erasing and Mixup/CutMix of the normalized training batch."""

import numpy as np
import pytest
import torch
from PIL import Image

import mini_trainer.training.mixing as mixing
from mini_trainer.hierarchical.loss import MultiLevelWeightedCrossEntropyLoss
from mini_trainer.training import BatchRegularizer, mixed_loss, soft_targets

SIZES = [11, 5, 3]  # classes per taxonomic rank, finest first


def hierarchical_batch(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    target = torch.stack([torch.randint(0, k, (n,), generator=g) for k in SIZES], 1)
    logits = [torch.randn(n, k, generator=g) for k in SIZES]
    return logits, target


@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_mixed_loss_is_the_soft_target_loss_at_every_rank(smoothing):
    logits, target = hierarchical_batch()
    criterion = MultiLevelWeightedCrossEntropyLoss(SIZES, torch.device("cpu"), torch.float32, label_smoothing=smoothing)
    lam = 0.37
    partner = target.flip(0)
    mixed = mixed_loss(criterion(logits, target), criterion(logits, partner), lam)
    soft = soft_targets(target, partner, lam, SIZES)
    for rank, (k, q) in enumerate(zip(SIZES, soft)):
        assert torch.allclose(q.sum(1), torch.ones(len(q)))  # a distribution per image and rank
        own = q.gather(1, target[:, rank : rank + 1]).squeeze(1)
        same = target[:, rank] == partner[:, rank]
        assert torch.allclose(own, torch.where(same, torch.ones_like(own), torch.full_like(own, lam)))  # the mixing weight
        eps = criterion.label_smoothing[rank]
        reference = torch.nn.functional.cross_entropy(logits[rank], (1 - eps) * q + eps / k)
        assert mixed[rank].item() == pytest.approx(reference.item(), rel=1e-5)


def test_mixup_and_cutmix_follow_timm_batch_mode():
    x = torch.randn(6, 3, 16, 16)
    reg = BatchRegularizer(mixup_alpha=0.8, cutmix_alpha=0.0, seed=1)
    out, lam = reg(x)
    assert 0 < lam < 1 and torch.allclose(out, lam * x + (1 - lam) * x.flip(0))
    reg = BatchRegularizer(mixup_alpha=0.0, cutmix_alpha=1.0, seed=1)
    for _ in range(20):
        out, lam = reg(x)
        from_self = (out == x).all(1)  # pixels kept from each image
        assert out.shape == x.shape and float(from_self.float().mean()) == pytest.approx(lam)  # corrected lam = kept area
    # both alphas: one method per batch, CutMix with probability switch_prob
    reg = BatchRegularizer(mixup_alpha=0.8, cutmix_alpha=1.0, switch_prob=0.5, seed=2)
    kinds = []
    for _ in range(400):
        out, lam = reg(x)
        kinds.append(bool(((out == x) | (out == x.flip(0))).all()))  # CutMix copies pixels; Mixup blends them
    assert 0.42 < np.mean(kinds) < 0.58
    assert BatchRegularizer(mixup_alpha=0.8, cutmix_alpha=1.0, mix_prob=0.0)(x)[1] == 1.0


def test_random_erasing_keeps_shapes_and_fills_one_normal_box():
    x = torch.zeros(400, 3, 32, 32)
    out, lam = BatchRegularizer(erase_p=0.25, seed=0)(x)
    assert lam == 1.0 and out.shape == x.shape and out.dtype == x.dtype and torch.equal(x, torch.zeros_like(x))  # input kept
    erased = (out != 0).flatten(1).any(1)
    assert 0.19 < float(erased.float().mean()) < 0.31
    area = (out[erased] != 0).all(1).flatten(1).float().mean(1)  # erased fraction of each erased image
    assert float(area.min()) >= 0.015 and float(area.max()) <= 0.34
    noise = out[erased][out[erased] != 0]
    assert abs(float(noise.mean())) < 0.02 and abs(float(noise.std()) - 1) < 0.02
    half = BatchRegularizer(erase_p=0.25, seed=0)(x.half())[0]
    assert half.dtype == torch.float16 and torch.isfinite(half).all()


def test_draws_depend_on_seed_rank_and_epoch_only(monkeypatch):
    x = torch.randn(8, 3, 16, 16)

    def draws(seed=0, rank=0, epoch=3):
        monkeypatch.setattr(mixing, "get_rank", lambda: rank)
        reg = BatchRegularizer(erase_p=0.25, mixup_alpha=0.8, cutmix_alpha=1.0, seed=seed)
        reg.set_epoch(epoch)
        return [reg(x) for _ in range(3)]

    state = torch.random.get_rng_state()
    a, b = draws(), draws()
    assert torch.equal(torch.random.get_rng_state(), state)  # the global stream is untouched
    assert all(torch.equal(p[0], q[0]) and p[1] == q[1] for p, q in zip(a, b))  # an epoch replays (resume)
    for other in (draws(seed=1), draws(rank=1), draws(epoch=4)):
        assert any(not torch.equal(p[0], q[0]) for p, q in zip(a, other))


def test_off_is_the_identity_and_invalid_values_are_refused():
    x = torch.randn(4, 3, 8, 8)
    out, lam = BatchRegularizer()(x)
    assert out is x and lam == 1.0
    with pytest.raises(ValueError, match="erase_p"):
        BatchRegularizer(erase_p=1.5)
    with pytest.raises(ValueError, match="non-negative"):
        BatchRegularizer(mixup_alpha=-1)


def train_once(tmp_path, name, **augmentation):
    from mini_trainer.logging import configure_loggers
    from mini_trainer.train import main
    from tests.integration.test_integration_train import TinyMockModel

    data = tmp_path / "data"
    if not data.exists():
        rng = np.random.default_rng(0)
        for cls in ("class_a", "class_b"):
            (data / cls).mkdir(parents=True)
            for i in range(8):
                Image.fromarray(rng.integers(0, 255, (10, 10, 3), dtype=np.uint8)).save(data / cls / f"{i}.jpg")
    main(
        input=str(data),
        output=str(tmp_path / "out"),
        name=name,
        epochs=2,
        size=8,
        device="cpu",
        dtype="float32",
        seed=0,
        model_builder_kwargs={"model_type": TinyMockModel(), "pretrained": False},
        dataloader_builder_kwargs={"batch_size": 4, "num_workers": 0, "train_proportion": 0.75},
        augmentation_builder_kwargs=augmentation,
        logger_builder_kwargs={"logger_cls": configure_loggers()},
    )
    state = torch.load(tmp_path / "out" / name / "weights" / "checkpoint_last.pth", map_location="cpu", weights_only=False)
    return state["model"]


class _Logger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: ""


class _NoTeacher:
    def __bool__(self):
        return False

    def update_parameters(self, *args):
        pass


def one_epoch(batch_regularizer, criterion_kind="hierarchical"):
    """Train a seeded linear hierarchical model for one epoch on fixed batches; return its weights."""
    from mini_trainer.trainer import train_one_epoch

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, sum(SIZES)))

    class Split(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = model

        def forward(self, x):
            return list(self.body(x).split(SIZES, 1))

    net = Split()
    g = torch.Generator().manual_seed(1)
    images = torch.rand(24, 3, 8, 8, generator=g)
    labels = torch.stack([torch.randint(0, k, (24,), generator=g) for k in SIZES], 1)
    batches = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, labels), batch_size=6)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    train_one_epoch(
        net, _NoTeacher(), MultiLevelWeightedCrossEntropyLoss(SIZES, torch.device("cpu"), torch.float32), optimizer,
        torch.amp.GradScaler("cpu", enabled=False), torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0), batches, 0,
        _Logger(), batch_regularizer=batch_regularizer,
    )  # fmt: skip
    return {k: v.clone() for k, v in net.state_dict().items()}


def test_off_reproduces_todays_training_step_and_on_changes_it():
    today = one_epoch(None)
    state = torch.random.get_rng_state()
    zero = one_epoch(BatchRegularizer(erase_p=0.0, mixup_alpha=0.0, cutmix_alpha=0.0))
    assert all(torch.equal(today[k], zero[k]) for k in today)
    on = one_epoch(BatchRegularizer(erase_p=0.25, mixup_alpha=0.8, cutmix_alpha=1.0))
    assert any(not torch.equal(today[k], on[k]) for k in today) and all(torch.isfinite(v).all() for v in on.values())
    torch.random.set_rng_state(state)


def test_training_reads_the_options_from_the_augmentation_kwargs(tmp_path):
    on = train_once(tmp_path, "on", random_erasing=0.25, mixup_alpha=0.8, cutmix_alpha=1.0)
    assert all(torch.isfinite(v).all() for v in on.values() if isinstance(v, torch.Tensor) and v.is_floating_point())
    log = (tmp_path / "out" / "on" / "config.yaml").read_text()
    assert "mixup_alpha: 0.8" in log and "random_erasing: 0.25" in log
