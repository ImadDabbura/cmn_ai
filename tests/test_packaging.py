"""Tests for package metadata."""

from pathlib import Path
import tomllib


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
