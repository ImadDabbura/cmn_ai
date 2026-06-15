"""Tests for tabular EDA utilities."""

import importlib

import cmn_ai.tabular.eda as eda


def test_importing_eda_does_not_set_global_seaborn_style(monkeypatch):
    """Test importing EDA utilities does not mutate global seaborn style."""
    calls = []
    monkeypatch.setattr(
        eda.sns, "set", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    importlib.reload(eda)

    assert calls == []
