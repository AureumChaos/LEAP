import collections

import pytest

import leap_ec.ops as ops
from leap_ec.data import test_population


##############################
# Test iteriter_op
##############################
def test_iteriter_op_1():
    """If an iteriter_op is given an iterator as input, no exception should be thrown, and we should return
    the wrapped function's output."""

    @ops.iteriter_op
    def f(x):
        return iter([4, 5, 6])

    result = f(iter([1, 2, 3]))  # Passing in an iterator, as expected

    assert(isinstance(result, collections.abc.Iterator)), f"{result}"
    assert(list(result) == [4, 5, 6])


def test_iteriter_op_2():
    """If an iteriter_op is given something besides an iterator as input, raise a ValueError."""

    @ops.iteriter_op
    def f(x):
        return iter([4, 5, 6])

    with pytest.raises(ValueError):
        f([1, 2, 3])  # Passing in a list instead of an iterator


def test_iteriter_op_3():
    """If an iteriter_op returns something besides an iterator as output, raise a ValueError."""

    @ops.iteriter_op
    def f(x):
        return [4, 5, 6]  # Returning a list instead of an iterator

    with pytest.raises(ValueError):
        result = f(iter([1, 2, 3]))


##############################
# Test listlist_op
##############################
def test_listlist_op_1():
    """If a listlist_op is given a list as input, no exception should be thrown, and we should return
    the wrapped function's output."""

    @ops.listlist_op
    def f(x):
        return [4, 5, 6]

    result = f([1, 2, 3])  # Passing in a list, as expected

    assert(isinstance(result, list)), f"{result}"
    assert(result == [4, 5, 6])


def test_listlist_op_2():
    """If a listlist_op is given something besides a list as input, raise a ValueError."""

    @ops.listlist_op
    def f(x):
        return [4, 5, 6]

    with pytest.raises(ValueError):
        f(iter([1, 2, 3]))  # Passing in an iterator instead of an list


def test_listlist_op_3():
    """If a listlist_op returns something besides a list as output, raise a ValueError."""

    @ops.listlist_op
    def f(x):
        return iter([4, 5, 6])  # Returning an iterator instead of an list

    with pytest.raises(ValueError):
        result = f([1, 2, 3])


##############################
# Test listiter_op
##############################
def test_listiter_op_1():
    """If a listiter_op is given a list as input, no exception should be thrown, and we should return
    the wrapped function's output."""

    @ops.listiter_op
    def f(x):
        return iter([4, 5, 6])

    result = f([1, 2, 3])  # Passing in a list, as expected

    assert(isinstance(result, collections.abc.Iterator)), f"{result}"
    assert(list(result) == [4, 5, 6])


def test_listiter_op_2():
    """If a listiter_op is given something besides a list as input, raise a ValueError."""

    @ops.listiter_op
    def f(x):
        return iter([4, 5, 6])

    with pytest.raises(ValueError):
        f(iter([1, 2, 3]))  # Passing in an iterator instead of a list


def test_listiter_op_3():
    """If a listiter_op returns something besides an iterator as output, raise a ValueError."""

    @ops.listiter_op
    def f(x):
        return [4, 5, 6]  # Returning a list instead of an iterator

    with pytest.raises(ValueError):
        result = f([1, 2, 3])


##############################
# Test iterlist_op
##############################
def test_iterlist_op_1():
    """If an iterlist_op is given an iterator as input, no exception should be thrown, and we should return
    the wrapped function's output."""

    @ops.iterlist_op
    def f(x):
        return [4, 5, 6]

    result = f(iter([1, 2, 3]))  # Passing in an iterator, as expected

    assert(isinstance(result, list)), f"{result}"
    assert(result == [4, 5, 6])


def test_iterlist_op_2():
    """If an iterlist_op is given something besides an iterator as input, raise a ValueError."""

    @ops.iterlist_op
    def f(x):
        return [4, 5, 6]

    with pytest.raises(ValueError):
        f([1, 2, 3])  # Passing in a list instead of an iterator


def test_iterlist_op_3():
    """If an iterlist_op returns something besides a list as output, raise a ValueError."""

    @ops.iterlist_op
    def f(x):
        return iter([4, 5, 6])  # Returning an iterator instead of a list

    with pytest.raises(ValueError):
        result = f(iter([1, 2, 3]))


##############################
# Test const_evaluate()
##############################
def test_const_evaluate():
    """Constant evaluation should ignore the existing fitness function and
    set the fitness of all individuals to the same value."""
    pop = test_population
    pop = ops.const_evaluate(pop, value=123456789.0)
    
    for ind in pop:
        assert(pytest.approx(123456789.0) == ind.fitness)


##############################
# Test pool()
##############################
def test_pool():
    """If a pool of size 3 is used, the first 3 individuals in the input iterator should be collected
    into a list."""
    pop = iter([ 'a', 'b', 'c', 'd', 'e' ])
    pop = ops.pool(pop, size=3)

    assert(len(pop) == 3)
    assert(pop == [ 'a', 'b', 'c' ])
