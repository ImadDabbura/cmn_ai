from types import SimpleNamespace

import cmn_ai.utils.utils as utils_mod
from cmn_ai.utils.utils import listify, setify, tuplify, uniqueify


def test_listify():
    assert listify(None) == []
    assert listify([]) == []
    assert listify([1]) == [1]
    assert listify([1, 2]) == [1, 2]
    assert listify((1,)) == [1]
    assert listify("") == [""]
    assert listify("str") == ["str"]


def test_tuplify():
    assert tuplify(None) == ()
    assert tuplify([]) == ()
    assert tuplify([1]) == (1,)
    assert tuplify([1, 2]) == (1, 2)
    assert tuplify((1,)) == (1,)
    assert tuplify("") == ("",)
    assert tuplify("str") == ("str",)
    assert tuplify(()) == ()
    assert tuplify((1,)) == (1,)


def test_setify():
    assert setify(None) == set()
    assert setify([]) == set()
    assert setify(set()) == set()
    assert setify([1]) == {1}
    assert setify({1}) == {1}
    assert setify([1, 2]) == {1, 2}
    assert setify((1,)) == {1}
    assert setify("") == {""}
    assert setify("str") == {"str"}


def test_uniqueify():
    assert uniqueify([1, 1]) == [1]
    assert uniqueify((1, 1)) == [1]
    assert uniqueify([1, -1, 3], True) == [-1, 1, 3]
    assert uniqueify((1.0, 2.0, -100), True)


def test_clean_memory_clears_only_available_device_caches(monkeypatch):
    calls = []

    monkeypatch.setattr(
        utils_mod, "clean_traceback", lambda: calls.append("traceback")
    )
    monkeypatch.setattr(
        utils_mod, "clean_ipython_history", lambda: calls.append("history")
    )
    monkeypatch.setattr(utils_mod.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(
        utils_mod.torch.cuda, "is_available", lambda: False
    )
    monkeypatch.setattr(
        utils_mod.torch.cuda, "empty_cache", lambda: calls.append("cuda")
    )
    monkeypatch.setattr(
        utils_mod.torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )
    monkeypatch.setattr(
        utils_mod.torch,
        "mps",
        SimpleNamespace(empty_cache=lambda: calls.append("mps")),
        raising=False,
    )

    utils_mod.clean_memory()

    assert calls == ["traceback", "history", "gc", "mps"]
