"""Tests for package metadata."""

from pathlib import Path
import tomllib

import cmn_ai
import yaml
from pre_commit.clientlib import load_config


ROOT = Path(__file__).resolve().parents[1]


def _local_pre_commit_hooks() -> dict[str, dict[str, object]]:
    config = load_config(str(ROOT / ".pre-commit-config.yaml"))
    return {
        hook["id"]: hook
        for repo in config["repos"]
        if repo["repo"] == "local"
        for hook in repo["hooks"]
    }


def test_project_license_metadata_matches_license_file() -> None:
    """Test pyproject license metadata matches the Apache license file."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]

    assert "Apache License" in (ROOT / "LICENSE").read_text()
    assert project["license"] == "Apache-2.0"
    assert (
        "License :: OSI Approved :: Apache Software License"
        in project["classifiers"]
    )
    assert "License :: OSI Approved :: MIT License" not in project[
        "classifiers"
    ]


def test_package_version_matches_project_metadata() -> None:
    """Test the public package version matches pyproject metadata."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert cmn_ai.__version__ == pyproject["project"]["version"]
    assert "__version__" in cmn_ai.__all__


def test_python_requires_is_capped_to_tested_versions() -> None:
    """Test Python support has an explicit upper bound."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert pyproject["project"]["requires-python"] == ">=3.13,<3.15"


def test_docs_extra_includes_versioning_provider() -> None:
    """Test docs dependencies include the mkdocs version provider."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    docs_deps = pyproject["project"]["optional-dependencies"]["docs"]

    assert any(dep.startswith("mike") for dep in docs_deps)


def test_package_ships_pep561_marker() -> None:
    """Test package includes a PEP 561 marker for inline typing."""
    assert (ROOT / "cmn_ai" / "py.typed").is_file()


def test_mypy_is_available_in_dev_dependencies() -> None:
    """Test mypy is installed with the dev extra."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dev_deps = pyproject["project"]["optional-dependencies"]["dev"]

    assert any(dep.startswith("mypy") for dep in dev_deps)


def test_pre_commit_runs_mypy() -> None:
    """Test pre-commit runs mypy over the package."""
    mypy_hook = _local_pre_commit_hooks()["mypy"]

    assert mypy_hook["entry"] == "mypy"
    assert mypy_hook["args"] == ["cmn_ai"]
    assert mypy_hook["language"] == "system"
    assert mypy_hook["pass_filenames"] is False


def test_mypy_excludes_local_scratch_file() -> None:
    """Test mypy ignores the kept scratch file inside the package tree."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert "cmn_ai/scratch-pad.py" in pyproject["tool"]["mypy"]["exclude"]


def test_pre_commit_runs_tests_only_on_pre_push() -> None:
    """Test slow test hooks run on pre-push and clean is not a hook."""
    local_hooks = _local_pre_commit_hooks()
    test_hook = local_hooks["test"]

    assert test_hook["stages"] == ["pre-push"]
    assert test_hook["pass_filenames"] is False
    assert "clean" not in local_hooks
    assert all("pass_filename" not in hook for hook in local_hooks.values())


def test_pytest_warnings_are_visible() -> None:
    """Test pytest is not configured to hide warnings."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    addopts = pyproject["tool"]["pytest"]["ini_options"]["addopts"]

    assert "--disable-pytest-warnings" not in addopts


def test_documentation_workflow_builds_prs_and_deploys_main_only() -> None:
    """Test docs workflow builds pull requests and deploys main pushes only."""
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "documentation.yml").read_text()
    )

    assert workflow["permissions"]["contents"] == "read"
    assert "mkdocs build --strict" in str(workflow["jobs"]["build-docs"])

    deploy_job = workflow["jobs"]["deploy-docs"]
    assert (
        deploy_job["if"]
        == "github.event_name == 'push' && github.ref == 'refs/heads/main'"
    )
    assert deploy_job["needs"] == "build-docs"
    assert deploy_job["permissions"]["contents"] == "write"
    assert "mkdocs gh-deploy --force" in str(deploy_job)


def test_readme_and_docs_do_not_link_planned_404_pages() -> None:
    """Test planned docs are described without linking to missing pages."""
    for rel_path in ("README.md", "docs/index.md"):
        text = (ROOT / rel_path).read_text()

        assert "imaddabbura.github.io/cmn_ai/api/" not in text
        assert "imaddabbura.github.io/cmn_ai/tutorials/" not in text
        assert "imaddabbura.github.io/cmn_ai/advanced/" not in text
        assert "Planned documentation" in text


def test_contributing_section_has_current_guidance() -> None:
    """Test contributing docs do not point to missing future guidelines."""
    for rel_path in ("README.md", "docs/index.md"):
        text = (ROOT / rel_path).read_text()

        assert "Stay tuned for contribution guidelines" not in text
        assert "open an issue or pull request" in text
