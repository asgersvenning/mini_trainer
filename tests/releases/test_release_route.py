"""Branch preparation must not confuse products or accept mismatched release versions."""

import pytest

from dev.release_route import resolve


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname="mt-trainer"\nversion="0.3.0"\n')
    (tmp_path / "deployment").mkdir()
    (tmp_path / "deployment/pyproject.toml").write_text('[project]\nname="mambo-v3"\nversion="0.3.0"\n')
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github/model-releases.toml").write_text(
        '[models.mambo-v3]\nproject="deployment"\nmodule="dev.releases.mambo_v3"\ntag="MAMBO_v3"\ntag_version="0.3.0"\n'
    )
    return tmp_path


@pytest.mark.parametrize("kind,product", [("packages", "mt-trainer"), ("models", "mambo-v3"), ("demos", "mambo-v3")])
def test_branch_and_manual_preparation_select_only_the_requested_product(repo, kind, product):
    result = resolve(repo, kind, "push", f"refs/heads/release/{kind}/{product}", {})
    assert result["enabled"] == "true" and result["product"] == product
    assert resolve(repo, kind, "push", "refs/heads/master", {}) == {"enabled": "false"}
    assert resolve(repo, kind, "push", f"refs/tags/{result['tag']}", {}) == {"enabled": "false"}
    assert resolve(repo, kind, "workflow_dispatch", "refs/heads/master", {}, product) == result


@pytest.mark.parametrize(
    "kind,tag,enabled",
    [
        ("packages", "packages/mt-trainer/v0.3.0", "true"),
        ("models", "MAMBO_v3", "true"),
        ("packages", "MAMBO_v3", "false"),
        ("models", "packages/mt-trainer/v0.3.0", "false"),
        ("models", "unrelated-release", "false"),
        ("demos", "MAMBO_v3", "false"),
    ],
)
def test_release_routes_are_independent(repo, kind, tag, enabled):
    assert resolve(repo, kind, "release", f"refs/tags/{tag}", {"release": {"tag_name": tag}})["enabled"] == enabled


def test_mismatched_or_unknown_product_release_is_rejected(repo):
    for tag in ("packages/mt-trainer/v0.4.0", "packages/another/v0.3.0"):
        with pytest.raises(ValueError):
            resolve(repo, "packages", "release", f"refs/tags/{tag}", {"release": {"tag_name": tag}})
    (repo / "deployment/pyproject.toml").write_text('[project]\nname="mambo-v3"\nversion="0.3.1"\n')
    with pytest.raises(ValueError, match="Explicit release tag"):
        resolve(repo, "models", "release", "refs/tags/MAMBO_v3", {"release": {"tag_name": "MAMBO_v3"}})


def test_future_model_uses_convention_without_workflow_changes(repo):
    (repo / "future").mkdir()
    (repo / "future/pyproject.toml").write_text('[project]\nname="another-model"\nversion="1.2.0"\n')
    with (repo / ".github/model-releases.toml").open("a") as stream:
        stream.write('[models.another-model]\nproject="future"\nmodule="dev.releases.another_model"\n')
    tag = "models/another-model/v1.2.0"
    result = resolve(repo, "models", "release", f"refs/tags/{tag}", {"release": {"tag_name": tag}})
    assert result["product"] == "another-model" and result["module"] == "dev.releases.another_model"
    assert resolve(repo, "demos", "push", "refs/heads/release/demos/another-model", {})["product"] == "another-model"
