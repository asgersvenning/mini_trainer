"""Research contracts: paired treatments, preserved partitions, and durable stages."""

import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from mini_trainer.modeling import Classifier
from mini_trainer.training import EMLACrossEntropy, class_weight_distribution_regularization
from publication.experiments.training_ablations import study, training
from publication.experiments.training_ablations.data import prepare_data, select_families, select_species, species_table, write_json


def test_phase_memory_retains_peak_across_batch_resets(tmp_path, monkeypatch):
    logger = training.StudyLogger.__new__(training.StudyLogger)
    logger.output_dir, logger._epoch, logger._type = str(tmp_path), 0, "train"
    logger._phase_peak = 0
    monkeypatch.setattr(logger, "summary", lambda: {})
    monkeypatch.setattr(training.MultiLogger, "log_memory_use", lambda self: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    peaks = iter([1000, 600, 200])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: next(peaks))
    logger.log_memory_use()
    logger.log_memory_use()
    logger.save()
    record = json.loads((tmp_path / "learning.jsonl").read_text())
    assert record["peak_allocated_bytes"] == 1000


def metadata():
    return pd.DataFrame(
        [
            {
                "speciesKey": str(c),
                "familyKey": str(c // 2),
                "genusKey": str(c),
                "set": split,
                "gbifID": f"{c}-{split}-{i}",
                "filename": f"{split}-{i}.png",
                "scientificName": f"Species {c}",
            }
            for c in range(8)
            for split in ["0", "1", "2"]
            for i in range(c + 2)
        ]
    )


def fixture_campaign(tmp_path):
    frame = metadata()
    parquet = tmp_path / "source.parquet"
    frame.to_parquet(parquet)
    root = tmp_path / "study"
    root.mkdir()
    config = {
        **study.DEFAULTS,
        "parquet": str(parquet),
        "images": str(tmp_path / "images"),
        "species": 4,
        "size": 16,
        "batch_size": 4,
        "workers": 0,
        "device": "cpu",
        "wandb": False,
        "dtype": "float32",
    }
    prepare_data(config, root)
    write_json(root / "config.json", config)
    return root, config


def test_selection_preserves_splits_and_ignores_holdout_abundance(tmp_path):
    frame = metadata()
    table = species_table(frame)
    first = select_species(table, 4, 42)
    assert first == select_species(species_table(frame.sample(frac=1, random_state=5)), 4, 42)
    assert first == select_species(species_table(pd.concat([frame, frame[frame["set"] == "0"]])), 4, 42)
    assert first == select_species(table, 8, 42)[:4]
    root, _ = fixture_campaign(tmp_path)
    samples = pd.read_parquet(root / "samples.parquet")
    originals = frame.set_index(["speciesKey", "filename"])["set"]
    assert samples.set_index(["speciesKey", "filename"])["set"].equals(
        originals.reindex(samples.set_index(["speciesKey", "filename"]).index)
    )
    assert set(samples.groupby("speciesKey").split.nunique()) == {3}


def test_selection_rejects_cross_partition_observations(tmp_path):
    root, config = fixture_campaign(tmp_path)
    frame = metadata()
    frame["gbifID"] = "same-observation"
    frame.to_parquet(config["parquet"])
    with pytest.raises(ValueError, match="Observation crosses"):
        prepare_data(config, root)


def test_losses_and_regularizer_have_expected_behavior():
    logits = torch.randn(12, 4, requires_grad=True)
    targets = torch.arange(12) % 4
    expected = torch.nn.functional.cross_entropy(logits, targets, label_smoothing=0.25)
    torch.testing.assert_close(EMLACrossEntropy([10] * 4, label_smoothing=0.25)(logits, targets), expected)
    fixed = training.FixedAdjustment([2, 4, 8, 16], label_smoothing=0.25)
    torch.testing.assert_close(
        fixed(logits, targets), torch.nn.functional.cross_entropy(logits + fixed.adjustments, targets, label_smoothing=0.25)
    )
    weights = torch.randn(4, 8, requires_grad=True)
    class_weight_distribution_regularization(weights).backward()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()


def test_shared_projection_and_isolated_random_stream():
    hashes = []
    for normalized in [False, True]:
        head = Classifier(8, 4, hidden=True, normalized=normalized, skip_spherical_init=True)
        training.initialize_head(head, 42)
        hashes.append(training.tensor_hash(head.hidden.state_dict().items()))
    assert hashes[0] == hashes[1]
    stream = training.RandomStream(lambda: torch.rand(3), 42)
    other = training.RandomStream(lambda: torch.rand(3), 42)
    torch.testing.assert_close(stream(), other())
    torch.rand(100)
    torch.testing.assert_close(stream(), other())


def test_tuning_tiebreak_and_main_matrix(tmp_path):
    paths = []
    for index, run in enumerate(study.tuning_runs(study.DEFAULTS)):
        path = tmp_path / str(index)
        path.mkdir()
        write_json(path / "run.json", run)
        write_json(path / "evaluation.json", {"macro_recall": 0.5, "nll": 1})
        paths.append(path)
    chosen = study.select_tuning(paths)
    assert all(v == {"lr": 0.0003, "weight_decay": 0.001} for v in chosen.values())
    runs = study.main_runs(study.DEFAULTS, chosen)
    assert len({r["id"] for r in runs}) == len(runs)
    assert {r["seed"] for r in runs} == {42, 43, 44}


def test_metrics_probability_semantics_and_missing_support():
    result = training.metrics(np.zeros((2, 4)), [0, 1], [1, 2, 3, 4])
    assert result["nll"] == pytest.approx(np.log(4))
    assert result["brier"] == pytest.approx(0.75)
    assert result["recall"] == [1.0, 0.0, None, None]


def test_recovery_reuses_training_and_verifies_completed_artifacts(tmp_path, monkeypatch):
    run = {"id": "example"}
    stages = []

    def child(root, attempt, stage, *args):
        stages.append(stage)
        if stage == "train":
            (attempt / "model/weights").mkdir(parents=True)
            for name in ["model/weights/last.pt", "train.json", "initialization.json", "parameter_groups.json"]:
                (attempt / name).write_text("{}")
        elif stages.count("evaluate") == 1:
            raise RuntimeError("Evaluation interrupted")
        else:
            for name in ["evaluation.json", "predictions.npz"]:
                (attempt / name).write_text("{}")

    monkeypatch.setattr(study, "child", child)
    with pytest.raises(RuntimeError, match="interrupted"):
        study.execute(tmp_path, run, {}, "0", time.monotonic() + 300)
    path = study.execute(tmp_path, run, {}, "0", time.monotonic() + 300, retry=True)
    assert stages == ["train", "evaluate", "evaluate"]
    assert study.execute(tmp_path, run, {}, "0", time.monotonic() + 300) == path
    (path / "model/weights/last.pt").write_text("modified")
    with pytest.raises(ValueError, match="artifact changed"):
        study.completed(path, run)


def test_queue_does_not_hide_data_failure_behind_oom(tmp_path, monkeypatch):
    def execute(root, run, *args):
        if run["id"] == "oom":
            raise study.CUDAOutOfMemory("memory")
        raise ValueError("corrupt image")

    monkeypatch.setattr(study, "execute", execute)
    with pytest.raises(ValueError, match="corrupt image"):
        study.queue(tmp_path, [{"id": "oom"}, {"id": "data"}], {}, ["0", "1"], time.monotonic() + 100, False)


def test_deadline_prevents_new_child(tmp_path):
    with pytest.raises(TimeoutError, match="deadline"):
        study.child(tmp_path, tmp_path, "train", "0", {}, time.monotonic())


def test_deadline_terminates_running_process(tmp_path, monkeypatch):
    popen = subprocess.Popen
    processes = []

    def sleeping_child(command, **kwargs):
        process = popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(study.subprocess, "Popen", sleeping_child)
    with pytest.raises(TimeoutError, match="deadline"):
        study.child(tmp_path, tmp_path, "train", "0", {"threads": 1}, time.monotonic() + 60.2)
    assert processes and processes[0].poll() is not None


@pytest.mark.parametrize(
    "hidden,optimizer,normalized,loss,rank_weights",
    [
        (False, "adamw", True, "emla", None),
        (True, "adamw", True, "emla", None),
        (False, "muon", True, "emla", None),
        (True, "muon", True, "emla", None),
        (True, "muon", False, "fixed", None),
        (True, "muon", False, "ce", None),
        (True, "muon", False, "emla", None),
        (False, "adamw", False, "ce", None),
        (True, "muon", True, "emla", [1.0, 0.0, 0.0]),
        (True, "muon", True, "emla", [1 / 3] * 3),
    ],
)
def test_tiny_train_reload_evaluate(tmp_path, monkeypatch, hidden, optimizer, normalized, loss, rank_weights):
    from mini_trainer.modeling import classifier
    from tests.integration.test_integration_train import TinyMockModel

    root, config = fixture_campaign(tmp_path)
    config["log_gates"] = True
    if rank_weights is not None:
        config.update(
            hierarchy=True,
            cohort_families=select_families(species_table(metadata()), 1000, config["selection_seed"]),
            species=8,
            train_image_budget=1000,
        )
        prepare_data(config, root)
        # Targeted campaigns mix heads without enabling campaign-wide hierarchy.
        if rank_weights[1] > 0:
            config["hierarchy"] = False
    for row in pd.read_parquet(root / "samples.parquet").itertuples():
        path = tmp_path / "images" / row.sample_id
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((16, 16, 3), 30 + 20 * row.label, dtype=np.uint8)).save(path)
    original = classifier.get_model
    monkeypatch.setattr(
        classifier,
        "get_model",
        lambda *args, **kwargs: original(
            TinyMockModel(), model_args={"pretrained": False}, transform=training.transforms.ConvertImageDtype(torch.float32)
        ),
    )
    torch.save(TinyMockModel().state_dict(), root / "pretrained.pt")
    # Test the real training/evaluation orchestration without compiling augmentation kernels.
    monkeypatch.setattr(training.StudyBuilder, "build_augmentation", lambda **kwargs: training.transforms.Compose([]))
    built = []
    build_model = training.StudyBuilder.build_model

    def capture(cls, **kwargs):
        result = build_model(**kwargs)
        built.append(result)
        return result

    monkeypatch.setattr(training.StudyBuilder, "build_model", classmethod(capture))
    attempt = root / "attempt"
    attempt.mkdir()
    run = {
        **study.FULL,
        "hidden": hidden,
        "optimizer": optimizer,
        "normalized": normalized,
        "loss": loss,
        "seed": 42,
        "lr": 0.001,
        "weight_decay": 0.01,
        "epochs": 2,
        "id": "tiny",
        "rank_weights": rank_weights,
        "screening": True,
    }
    training.train(root, attempt, config, run)
    training.evaluate(root, attempt, config, run)
    result = json.loads((attempt / "evaluation.json").read_text())
    if rank_weights is not None:
        assert np.isfinite(result["genus"]["nll"]) and np.isfinite(result["family"]["nll"])
    assert result["split"] == "validation"
    assert result["backbone_parameters_changed"]
    assert sum(result["support"]) > 0
    assert np.isfinite(result["nll"])
    training.export_predictions(root, attempt, config, run, tmp_path / "predicted", provenance={"source": "test"})
    test_ids = pd.read_parquet(root / "samples.parquet").query("split == 'test'").sample_id.tolist()
    index = pd.read_parquet(tmp_path / "predicted/index.parquet")
    assert index.path.tolist() == test_ids and json.loads((tmp_path / "predicted/prediction.json").read_text())["split"] == "test"
    ranks = sorted(p.name for p in (tmp_path / "predicted").glob("rank-*"))
    assert ranks == (["rank-0", "rank-1", "rank-2"] if rank_weights is not None else ["rank-0"])
    for name in ranks:
        log_probabilities = pd.read_parquet(tmp_path / "predicted" / name).drop(columns="row").to_numpy(np.float64)
        assert np.allclose(np.exp(log_probabilities).sum(1), 1, atol=1e-2)
    embeddings = pd.read_parquet(tmp_path / "predicted/embeddings").drop(columns="row").to_numpy(np.float64)
    assert len(embeddings) == len(test_ids)
    if normalized:
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-2)
    if hidden and optimizer == "adamw" and normalized and rank_weights is None:
        import shutil

        from publication.experiments.training_ablations.data import digest
        from publication.experiments.training_ablations.embeddings import extract

        write_json(attempt / "run.json", {**run, "variant": "full"})
        write_json(attempt / "complete.json", {name: digest(attempt / name) for name in ["run.json", "model/weights/last.pt"]})
        shutil.copytree(attempt, root / "runs/tiny/attempt-000")
        write_json(
            root / "prepared.json", {"files": {name: digest(root / name) for name in ["config.json", "classes.json", "samples.parquet"]}}
        )
        extract(root, tmp_path / "embeddings", per_class=2)
        observed = pd.read_csv(tmp_path / "embeddings/tiny.csv")
        assert observed.samples.tolist() == [2] * 4
        assert np.isfinite(observed.mean_within_class_angle).all()
    live, preprocess = built[0]
    from mini_trainer.hierarchical.model import HierarchicalClassifier

    head_cls = HierarchicalClassifier if rank_weights is not None else Classifier
    loaded, loaded_preprocess = head_cls.build(weights=str(attempt / "model/weights/last.pt"), device="cpu")
    live.eval()
    loaded.eval()
    images = torch.randint(0, 256, (3, 3, 16, 16), dtype=torch.uint8)
    with torch.inference_mode():
        torch.testing.assert_close(live(preprocess(images)), loaded(loaded_preprocess(images)))


def test_corrected_index_preserves_taxonomy_splits_and_paths(tmp_path):
    index = {"path": [], "split": [], "label": []}
    for label in range(6):
        for split in ["train", "validation", "test"]:
            relative = f"images_gbif/{label}/{label}-{split}.jpg"
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            index["path"].append(relative)
            index["split"].append(split)
            index["label"].append([str(label), str(label // 2), str(label // 4)])
    source = tmp_path / "data_index.json"
    write_json(source, index)
    root = tmp_path / "prepared"
    root.mkdir()
    config = {"data_index": str(source), "images": str(tmp_path), "size": 384}
    prepare_data(config, root)
    frame = pd.read_parquet(root / "samples.parquet")
    assert set(frame.sample_id) == set(index["path"])
    assert frame.groupby("speciesKey").split.nunique().tolist() == [3] * 6
    assert "gbifID" not in frame and "set" not in frame
    spec = json.loads((root / "classes.json").read_text())
    assert spec["hierarchy"]["masks"] == [[0, 0, 1, 1, 2, 2], [0, 0, 1]]
    index["label"][1][1] = "different-genus"
    write_json(source, index)
    with pytest.raises(ValueError, match="Conflicting taxonomy"):
        prepare_data(config, root)
    index["label"][1][1] = "0"
    index["path"][1] = index["path"][0]
    write_json(source, index)
    with pytest.raises(ValueError, match="Duplicate image"):
        prepare_data(config, root)
    index["path"][1] = "../outside.jpg"
    write_json(source, index)
    with pytest.raises(ValueError, match="relative"):
        prepare_data(config, root)


def test_lower_unique_support_preserves_draws_and_held_out_examples(tmp_path):
    index = {"path": [], "split": [], "label": []}
    for label in range(6):
        for split in ["train", "validation", "test"]:
            for sample in range(2 if split == "train" else 1):
                relative = f"images_gbif/{label}/{label}-{split}-{sample}.jpg"
                path = tmp_path / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                index["path"].append(relative)
                index["split"].append(split)
                index["label"].append([str(label), str(label // 2), str(label // 4)])
    source = tmp_path / "data_index.json"
    write_json(source, index)
    root = tmp_path / "prepared"
    root.mkdir()
    config = {"data_index": str(source), "images": str(tmp_path), "size": 16, "train_support_cap": 1, "train_support_seed": 42}
    prepare_data(config, root)

    samples = pd.read_parquet(root / "samples.parquet")
    train = samples[samples.split == "train"]
    assert len(train) == 12
    assert train.groupby("speciesKey").sample_id.nunique().tolist() == [1, 1, 2, 2, 2, 2]
    assert set(samples[samples.split != "train"].sample_id) == {
        path for path, split in zip(index["path"], index["split"]) if split != "train"
    }
    species = pd.read_csv(root / "species.csv", index_col=0)
    assert species.train.tolist() == [2] * 6
    assert species.unique_train_support.tolist() == [1, 1, 2, 2, 2, 2]
    assert species.training_draws.tolist() == [2] * 6
    selection = json.loads((root / "selection.json").read_text())
    assert selection["support_reduction"]["unique_train_images_before"] == 12
    assert selection["support_reduction"]["unique_train_images_after"] == 10


def test_gate_observer_preserves_loss_gradients_and_rng():
    inputs = torch.tensor([[0.0, 0, 0], [8.0, 0, 0], [0, 0, 8.0]], requires_grad=True)
    target = torch.tensor([0, 1, 2])
    loss = EMLACrossEntropy([1, 10, 100])
    baseline = loss(inputs, target)
    gradient = torch.autograd.grad(baseline, inputs)[0]
    recorder = training.GateRecorder()
    loss.register_forward_pre_hook(recorder.observe("species", [1, 10, 100]))
    before = torch.get_rng_state().clone()
    observed = loss(inputs, target)
    torch.testing.assert_close(observed, baseline, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(observed, inputs)[0], gradient, rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())
    summary = recorder.summary()
    assert summary["gate/species/tail/mean"] == pytest.approx(0, abs=1e-6)
    assert summary["gate/species/head/mean"] > 0.99


def test_targeted_comparisons_share_controls_and_keep_complete_cells():
    runs = study.main_runs(
        {**study.DEFAULTS, "targeted": True, "seeds": [42], "epochs": 10}, {"muon": {"lr": 0.003, "weight_decay": 0.001}}
    )
    assert len(runs) == 8
    rows = [
        {
            **r,
            "split": "validation",
            "score": 2 * r["normalized"] + 3 * r["regularization"] + 5 * bool(r.get("rank_weights")) * r["regularization"],
        }
        for r in runs
    ]
    results = study.factorial_contrasts(rows, metrics=("score",))
    interaction = [r for r in results if r["factors"] == ["hierarchy", "regularization"]]
    assert interaction[0]["difference"] == 5
    assert [r for r in results if r["factors"] == ["normalization", "regularization"]][0]["difference"] == 0
    missing = study.factorial_contrasts([r for r in rows if r["variant"] != "no_normalization"], metrics=("score",))
    assert all("normalization" not in r["factors"] for r in missing)
    with pytest.raises(ValueError, match="Duplicate"):
        study.factorial_contrasts(rows + [rows[0]], metrics=("score",))


def test_corrected_source_audit_retains_overlap_and_merge_evidence(tmp_path):
    from publication.experiments.training_ablations.data import audit_source_metadata

    source = pd.DataFrame(
        {
            "PN_hash": ["a", "b", "c"],
            "PN_observation_id": ["o1", "o1", "o2"],
            "species_id": ["1", "2", "3"],
            "split": ["train", "val", "test"],
        }
    )
    path = tmp_path / "source.csv"
    source.to_csv(path, index=False)
    samples = pd.DataFrame(
        {"sample_id": ["images_gbif/10/a.jpg", "images_gbif/10/b.jpg"], "speciesKey": ["10", "10"], "split": ["train", "validation"]}
    )
    audit = audit_source_metadata(samples, {"source_metadata": str(path)}, tmp_path)
    assert audit["cross_partition_observations"] == 1
    assert audit["merged_corrected_classes"] == 1
    assert audit["excluded_source_classes"] == ["3"]
    samples.loc[1, "split"] = "train"
    with pytest.raises(ValueError, match="preserve"):
        audit_source_metadata(samples, {"source_metadata": str(path)}, tmp_path)


@pytest.mark.parametrize(
    ("config", "dataset"),
    [
        ({"data_index": "/work/plantnet/data_index.json", "parquet": None}, "plantnet300k"),
        ({"data_index": None, "parquet": "/work/global_lepi/metadata.parquet"}, "global_lepidoptera"),
    ],
)
def test_evidence_dataset_follows_the_prepared_data_source(config, dataset):
    from publication.experiments.evidence import dataset_name
    from publication.experiments.training_ablations.data import data_source

    assert dataset_name(config) == data_source(config)[0] == dataset
