"""Offline UCloud plan and evidence-integrity contracts."""

import json
from pathlib import Path

import pytest

from dev.releases.mambo_v3.ucloud_release import RECIPE, configuration, jobs, validated_report

CONFIG = Path("dev/releases/mambo_v3/ucloud_release.json")


def test_quality_plan_preserves_global_population_and_legacy_isolation():
    config = configuration(CONFIG.resolve())
    plan = jobs(config, "qualification")
    assert len(plan) == 5
    for job in plan:
        command = job["command"]
        assert command[command.index("--presets") + 1] == "full"
        assert command[command.index("--count") + 1] == "256"
        if job["legacy"]:
            assert "-P" in command and "--precision" not in command
        else:
            assert command[command.index("--precision") + 1] == "auto"
            assert command[command.index("--tta") + 1] == (RECIPE if job["variant"].endswith("tta") else "none")
    assert all("--count" not in j["command"] for j in jobs(config, "full"))


def test_benchmark_bank_and_trials_match_across_backends():
    config = configuration(CONFIG.resolve())
    config["gpu_batches"] = [1, 8, 32, 64]
    plan = jobs(config, "benchmark")
    assert len(plan) == 30 and len({j["name"] for j in plan}) == 30
    for j in plan:
        c = j["command"]
        assert c[c.index("--bank-size") + 1] == "64"
        if j["legacy"] and j["device"] == "cpu":
            assert "--cpu-float32" in c
    assert plan[0]["variant"] == "v2" and plan[10]["variant"] == "onnx-tta"


def test_configuration_rejects_regional_only_quality(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["quality_presets"] = ["north_europe"]
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="global"):
        configuration(p)


def test_completed_job_rejects_changed_predictions(tmp_path):
    p = tmp_path / "full"
    p.mkdir()
    (p / "mini_metric.csv").write_text("changed")
    (tmp_path / "report.json").write_text(json.dumps({"status": "complete", "csv_sha256": {"full": "0" * 64}}))
    with pytest.raises(ValueError, match="Changed predictions"):
        validated_report(tmp_path)
    (tmp_path / "report.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="Incomplete job"):
        validated_report(tmp_path)


def test_bank_identity_matches_legacy_and_portable_reports(tmp_path):
    from dev.benchmarks.inference.onnx_inference import file_hash
    from dev.releases.mambo_v3.evaluation_data import write_json
    from dev.releases.mambo_v3.ucloud_release import bank_identity

    records = [{"path": "image.jpg", "sha256": "a" * 64}]
    write_json(tmp_path / "samples.json", records)
    legacy = {"samples": 1, "sample_ids_sha256": file_hash(tmp_path / "samples.json")}
    assert bank_identity(tmp_path, legacy) == bank_identity(tmp_path, {"samples": records})
    write_json(tmp_path / "samples.json", [{"path": "different.jpg"}])
    with pytest.raises(ValueError, match="Changed legacy"):
        bank_identity(tmp_path, legacy)


def test_summary_preserves_ranks_scopes_and_rejects_changed_evidence(tmp_path):
    from dev.benchmarks.inference.onnx_inference import file_hash
    from dev.releases.mambo_v3.evaluation_data import write_json
    from dev.releases.mambo_v3.metrics import METRIC_SCHEMA, REVISION
    from dev.releases.mambo_v3.ucloud_release import VARIANTS
    from dev.releases.mambo_v3.ucloud_summary import summarize

    for phase in ("full", "benchmark"):
        plan = dict(status="complete", fingerprint={"inputs": "same"}, environment_id="fixture", jobs=[], completed=[], reports_sha256={})
        for variant in VARIANTS:
            directory = tmp_path / phase / variant
            directory.mkdir(parents=True)
            job = dict(name=variant, variant=variant, device="cpu")
            plan["jobs"].append(job)
            plan["completed"].append(variant)
            report = dict(status="complete", peak_rss_kib_linux=1024)
            if phase == "full":
                (directory / "full").mkdir()
                csv = directory / "full/mini_metric.csv"
                csv.write_text("fixture predictions")
                digest = file_hash(csv)
                report.update(samples=632913, sample_ids_sha256="same-population", csv_sha256={"full": digest})
                metrics = dict(
                    source_sha256=digest,
                    metric_schema=METRIC_SCHEMA,
                    mini_metrics_revision=REVISION,
                    ranks={r: {"images": 632913} for r in ("species", "genus", "family")},
                    all={"f1": {str(i): 0.4 for i in range(3)}},
                    known={"f1": {str(i): 0.5 for i in range(3)}},
                )
                write_json(directory / "full/metrics.json", metrics)
            else:
                records = [{"path": "image.jpg"}]
                report.update(samples=records, cells=[dict(preset="full", batch_size=8, end_to_end={"seconds": [2] * 7})])
                if variant == "v2":
                    write_json(directory / "samples.json", records)
                    report.update(samples=1, sample_ids_sha256=file_hash(directory / "samples.json"))
            write_json(directory / "report.json", report)
            plan["reports_sha256"][variant] = file_hash(directory / "report.json")
        write_json(tmp_path / phase / "plan.json", plan)
    result = summarize(tmp_path, tmp_path / "summary")
    assert len(result["quality"]) == 30 and len(result["speed"]) == 5
    assert {r["f1"] for r in result["quality"]} == {0.4, 0.5}
    assert all(r["images_per_second"] == 4 for r in result["speed"])
    (tmp_path / "full/v2/full/mini_metric.csv").write_text("changed")
    with pytest.raises(ValueError, match="Changed predictions"):
        summarize(tmp_path, tmp_path / "changed-summary")


def test_setup_rejects_wrong_metadata_before_downloads(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from dev.releases.mambo_v3 import setup_ucloud_release

    path = tmp_path / "wrong.parquet"
    path.write_bytes(b"not the original snapshot")
    monkeypatch.setattr(setup_ucloud_release, "default_bundle", lambda: pytest.fail("Downloaded before metadata validation"))
    with pytest.raises(ValueError, match="metadata snapshot"):
        setup_ucloud_release.setup(SimpleNamespace(metadata=path))
