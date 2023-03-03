from dl_utils.utils import listify


def test_listify():
    assert listify(None) == []
    assert listify([]) == []
    assert listify([1]) == [1]
    assert listify([1, 2]) == [1, 2]
    assert listify((1,)) == [1]
    assert listify("") == [""]
    assert listify("str") == ["str"]
