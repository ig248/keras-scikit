"""See
https://github.com/cloudpipe/cloudpickle/blob/master/tests/cloudpickle_test.py
Our aim is not to test cloudpickle but to make sure we use an implementation
that can pickle lambdas etc.
"""
import tempfile

import pytest

from scikit_keras.utils.pickle import dumpf, loadf


def assert_is(a, b):
    assert a is b


def assert_eq(a, b):
    assert a == b


def assert_lambda_eq(a, b):
    for x in range(10):
        assert a(x) == b(x)


@pytest.mark.parametrize(
    'obj, compare', [
        (None, assert_is), ([1, 2, 3], assert_eq),
        (lambda x: x * x, assert_lambda_eq)
    ]
)
def test_pickle(obj, compare):
    with tempfile.NamedTemporaryFile(suffix='.pickle', delete=True) as f:
        dumpf(obj, f.name)
        obj2 = loadf(f.name)
    compare(obj, obj2)


if __name__ == '__main__':
    pytest.main()
