from dl_utils.utils import listify, tuplify


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
