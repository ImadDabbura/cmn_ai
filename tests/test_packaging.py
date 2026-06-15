"""Tests for package metadata."""

from pathlib import Path
import tomllib

import cmn_ai


ROOT = Path(__file__).resolve().parents[1]


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
