"""Offline UCloud plan and evidence-integrity contracts."""

import json
from pathlib import Path

import pytest

from dev.releases.mambo_v3.ucloud_release import RECIPE, configuration, jobs, validated_report

CONFIG = Path("dev/releases/mambo_v3/ucloud_release.json")


@pytest.mark.parametrize("phase", ["qualification", "full", "benchmark"])
def test_plan_routes_runtime_loading_and_batch_controls(phase):
    config = configuration(CONFIG.resolve())
    config["qualification_count"] = 7
    legacy = {job["name"]: job["command"] for job in jobs(config, phase) if job["legacy"]}
    config.update(onnx_python="/isolated/onnx/bin/python", decode_workers=16, prefetch_batches=2, v3_batch_size=256)
    for job in jobs(config, phase):
        command = job["command"]
        interpreter = "onnx_python" if job["variant"].startswith("onnx") else "v2_python" if job["legacy"] else "v3_python"
        assert command[0] == config[interpreter]
        if job["legacy"]:
            assert "-P" in command and "--precision" not in command and "--prefetch-batches" not in command
            if phase != "benchmark":
                assert command == legacy[job["name"]]
        else:
            assert command[command.index("--precision") + 1] == "auto"
            assert command[command.index("--tta") + 1] == (RECIPE if job["variant"].endswith("tta") else "none")
            worker_option = "--stream-workers" if phase == "benchmark" else "--decode-workers"
            assert command[command.index(worker_option) + 1] == "16"
            assert command[command.index("--prefetch-batches") + 1] == "2"
        if phase == "benchmark":
            assert command[command.index("--bank-size") + 1] == "256"
            sizes = command[command.index("--batches") + 1 : command.index("--bank-size")]
            assert ("256" in sizes) == (not job["legacy"] and job["device"] != "cpu")
        else:
            assert command[command.index("--presets") + 1] == "full"
            if not job["legacy"]:
                assert command[command.index("--batch-size") + 1] == "256"
            if phase == "qualification":
                assert command[command.index("--count") + 1] == "7"
            else:
                assert "--count" not in command


def test_benchmark_bank_and_trials_match_across_backends():
    config = configuration(CONFIG.resolve())
    config["gpu_batches"] = [1, 8, 32, 64]
    plan = jobs(config, "benchmark")
    assert plan and len({j["name"] for j in plan}) == len(plan)
    for j in plan:
        c = j["command"]
        assert c[c.index("--bank-size") + 1] == "64"
        if j["legacy"] and j["device"] == "cpu":
            assert "--cpu-float32" in c


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
                if variant != "v2":
                    report["cells"][0]["streaming"] = {"images": 1024, "seconds": [4, 4, 4]}
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
    assert len(result["streaming_speed"]) == 4
    assert all(r["images_per_second"] == 256 for r in result["streaming_speed"])
    assert (tmp_path / "summary/streaming_speed.csv").is_file()
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


def test_legacy_archive_uses_repository_root_from_any_working_directory(tmp_path, monkeypatch):
    import subprocess

    from dev.releases.mambo_v3.setup_ucloud_release import COMMIT, HERE, prepare_legacy_source

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "legacy" / COMMIT
    prepare_legacy_source(source)
    expected = subprocess.check_output(["git", "show", f"{COMMIT}:mini_trainer/__init__.py"], cwd=HERE.parents[2])
    assert (source / "mini_trainer/__init__.py").read_bytes() == expected
    assert (source / "mini_trainer/deploy.py").is_file()
    prepare_legacy_source(source)  # Reuse the completed extraction on setup retries.


def test_configuration_preserves_virtual_environment_interpreter(tmp_path):
    import subprocess
    import venv

    environment = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    interpreter = environment / "bin/python"
    config = json.loads(CONFIG.read_text())
    for key in ("v2_python", "v3_python", "metrics_python"):
        config[key] = "runtime/bin/python"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    loaded = configuration(path)
    for key in ("v2_python", "v3_python", "metrics_python"):
        assert loaded[key] == str(interpreter)
    prefix = subprocess.check_output([loaded["v2_python"], "-c", "import sys; print(sys.prefix)"], text=True).strip()
    assert Path(prefix) == environment


def test_runtime_evidence_detects_installed_package_changes(tmp_path):
    import subprocess
    import venv

    from dev.releases.mambo_v3.ucloud_release import runtime_environments

    environment = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=False).create(environment)
    interpreter = str(environment / "bin/python")
    config = {key: interpreter for key in ("v2_python", "v3_python", "metrics_python")}
    before = runtime_environments(config)
    site = Path(subprocess.check_output([interpreter, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
    dist = site / "example-1.0.dist-info"
    dist.mkdir()
    metadata = dist / "METADATA"
    metadata.write_text("Metadata-Version: 2.1\nName: example\nVersion: 1.0\n")
    installed = runtime_environments(config)
    assert before != installed
    assert installed[interpreter]["packages"] == [["example", "1.0", None]]
    metadata.write_text("Metadata-Version: 2.1\nName: example\nVersion: 2.0\n")
    assert runtime_environments(config) != installed


def test_new_campaign_reuses_assets_and_preserves_existing_evidence(tmp_path, monkeypatch):
    from dev.releases.mambo_v3 import ucloud_release

    original = configuration(CONFIG.resolve())
    original["onnx_python"] = str(tmp_path / "onnx/bin/python")
    interpreter = str(tmp_path / "venv/bin/python")
    monkeypatch.setattr(ucloud_release.sys, "executable", interpreter)
    output = tmp_path / "new-campaign"
    updated = ucloud_release.new_campaign(original, output)
    assert configuration(output / "config.json") == updated
    for key in ("v2_python", "v3_python", "metrics_python"):
        assert updated[key] == interpreter
        assert original[key] != interpreter
    for key in ("manifest", "root", "bundle", "legacy_source", "legacy_weights", "hf_cache"):
        assert updated[key] == original[key]
    assert updated["onnx_python"] == original["onnx_python"]
    assert updated["output"] == str(output)
    before = (output / "config.json").read_bytes()
    with pytest.raises(FileExistsError):
        ucloud_release.new_campaign(original, output)
    assert (output / "config.json").read_bytes() == before


@pytest.mark.parametrize("tamper", [None, "manifest", "environment", "script", "csv", "samples"])
def test_reuse_completed_v2_only_when_evidence_matches(tmp_path, tamper):
    import copy

    from dev.benchmarks.inference.onnx_inference import file_hash
    from dev.releases.mambo_v3.evaluation_data import write_json
    from dev.releases.mambo_v3.ucloud_release import reuse_v2

    source = tmp_path / "old/full"
    (source / "v2/full").mkdir(parents=True)
    (source / "v2/full/mini_metric.csv").write_text("original predictions")
    (source / "v2/samples.json").write_text("[]")
    report = dict(
        status="complete",
        csv_sha256={"full": file_hash(source / "v2/full/mini_metric.csv")},
        sample_ids_sha256=file_hash(source / "v2/samples.json"),
    )
    write_json(source / "v2/report.json", report)
    inputs = dict(
        manifest="manifest",
        bundle="bundle",
        environments={"python": {"version": "same"}},
        scripts={"legacy.py": "same", "dev/releases/mambo_v3/evaluate.py": "old"},
    )
    job = dict(name="v2", command=["python", "--output", "old-output"])
    plan = dict(completed=["v2"], jobs=[job], fingerprint={"inputs": inputs}, reports_sha256={"v2": file_hash(source / "v2/report.json")})
    write_json(source / "plan.json", plan)
    frozen = {"inputs": copy.deepcopy(inputs)}
    frozen["inputs"]["scripts"]["dev/releases/mambo_v3/evaluate.py"] = "updated-v3-only"
    config = dict(reuse_v2_from=str(source.parent), output=str(tmp_path / "new"), v2_python="python")
    if tamper == "manifest":
        frozen["inputs"]["manifest"] = "different"
    elif tamper == "environment":
        frozen["inputs"]["environments"]["python"] = {"version": "different"}
    elif tamper == "script":
        frozen["inputs"]["scripts"]["legacy.py"] = "different"
    elif tamper == "csv":
        (source / "v2/full/mini_metric.csv").write_text("changed")
    elif tamper == "samples":
        (source / "v2/samples.json").write_text("changed")
    if tamper:
        with pytest.raises(ValueError):
            reuse_v2(config, "full", job, frozen)
        assert not (tmp_path / "new/full/v2").exists()
    else:
        result = reuse_v2(config, "full", job, frozen)
        assert result["source"] == str(source)
        assert (tmp_path / "new/full/v2/full/mini_metric.csv").read_text() == "original predictions"
