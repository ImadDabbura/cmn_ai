import numpy as np

from cmn_ai.utils.utils import listify, set_seed, setify, tuplify, uniqueify


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


class TestSetSeeds:
    def test_numpy_equal(self):
        set_seed(123)
        np.testing.assert_array_equal(
            np.random.randint(-100, 100, 5), np.array([9, 26, -34, -2, -83])
        )
        set_seed(42)
        np.testing.assert_array_almost_equal(
            np.random.randn(5),
            np.array(
                [0.49671415, -0.1382643, 0.64768854, 1.52302986, -0.23415337]
            ),
        )

    def test_numpy_not_equal(self):
        pass

    def test_pytorch_equal(self):
        pass

    def test_pytorch_not_equal(self):
        pass
