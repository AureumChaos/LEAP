"""
    Unit tests for the various pipeline probes.

    Note that this *NOT* use python3 unittest.  pytest?
"""
import io

from leap_ec.probe import *
from leap_ec import core, data, ops


##############################
# Tests for CSVProbe
##############################
def test_AttributesCSVProbe_1():
    """When recording the attribute 'my_value' and leaving other arguments at their default,
    running CSVProbe on a population of individuals with just the attribute 'my_value' should
    produce the correct CSV-formatted output."""
    # Set up a population
    pop = data.test_population

    # Assign distinct values to an attribute on each individual
    attr = 'my_value'
    vals = ['GREEN', 15, 'BLUE', 72.81]
    for (ind, val) in zip(pop, vals):
        ind.__dict__[attr] = val

    # Setup a probe that writes to a str in memory
    stream = io.StringIO()
    probe = AttributesCSVProbe(core.context, ['my_value'], stream)

    # Set the generation in the context
    core.context['leap']['generation'] = 10

    # Execute
    probe(pop)
    result = stream.getvalue()
    stream.close()

    # Test
    expected = "step,my_value\n" + \
               "10,GREEN\n" + \
               "10,15\n" + \
               "10,BLUE\n" + \
               "10,72.81\n"
    assert (result == expected)


def test_AttributesCSVProbe_2():
    """When recording the attribute 'my_value' and leaving other arguments at their default,
    running CSVProbe on a population of individuals with several attributes should
    produce CSV-formatted output that only records 'my_value'."""
    # Set up a population
    pop = data.test_population

    # Assign distinct alues to an attribute on each individual
    attrs = [('foo', ['GREEN', 15, 'BLUE', 72.81]),
             ('bar', ['Colorless', 'green', 'ideas', 'sleep']),
             ('baz', [['a', 'b', 'c'], [1, 2, 3], [None, None, None], [0.1, 0.2, 0.3]])]
    for attr, vals in attrs:
        for (ind, val) in zip(pop, vals):
            ind.__dict__[attr] = val

    # Setup a probe that writes to a str in memory
    stream = io.StringIO()
    probe = AttributesCSVProbe(core.context, ['foo', 'bar'], stream)

    # Set the generation in the context
    core.context['leap']['generation'] = 10

    # Execute
    probe(pop)
    result = stream.getvalue()
    stream.close()

    # Test
    expected = "step,foo,bar\n" + \
               "10,GREEN,Colorless\n" + \
               "10,15,green\n" + \
               "10,BLUE,ideas\n" + \
               "10,72.81,sleep\n"
    assert (result == expected)


def test_AttributesCSVProbe_3():
    """Changing the order of the attributes list changes the order of the columns."""
    # Set up a population
    pop = data.test_population

    # Assign distinct values to an attribute on each individual
    attrs = [('foo', ['GREEN', 15, 'BLUE', 72.81]),
             ('bar', ['Colorless', 'green', 'ideas', 'sleep']),
             ('baz', [['a', 'b', 'c'], [1, 2, 3], [None, None, None], [0.1, 0.2, 0.3]])]
    for attr, vals in attrs:
        for (ind, val) in zip(pop, vals):
            ind.__dict__[attr] = val

    # Setup a probe that writes to a str in memory
    stream = io.StringIO()
    # Passing params in reverse order from the other test above
    probe = AttributesCSVProbe(core.context, ['bar', 'foo'], stream)

    # Set the generation in the context
    core.context['leap']['generation'] = 10

    # Execute
    probe(pop)
    result = stream.getvalue()
    stream.close()

    # Test
    expected = "step,bar,foo\n" + \
               "10,Colorless,GREEN\n" + \
               "10,green,15\n" + \
               "10,ideas,BLUE\n" + \
               "10,sleep,72.81\n"
    assert (result == expected)


def test_AttributesCSVProbe_4():
    """Providng an attribute that contains list data should work flawlessly.."""
    # Set up a population
    pop = data.test_population

    # Assign distinct values to an attribute on each individual
    attrs = [('foo', ['GREEN', 15, 'BLUE', 72.81]),
             ('bar', ['Colorless', 'green', 'ideas', 'sleep']),
             ('baz', [['a', 'b', 'c'], [1, 2, 3], [None, None, None], [0.1, 0.2, 0.3]])]
    for attr, vals in attrs:
        for (ind, val) in zip(pop, vals):
            ind.__dict__[attr] = val

    # Setup a probe that writes to a str in memory
    stream = io.StringIO()
    probe = AttributesCSVProbe(core.context, ['bar', 'baz'], stream)

    # Set the generation in the context
    core.context['leap']['generation'] = 10

    # Execute
    probe(pop)
    result = stream.getvalue()
    stream.close()

    # Test
    expected = "step,bar,baz\n" + \
               "10,Colorless,\"['a', 'b', 'c']\"\n" + \
               "10,green,\"[1, 2, 3]\"\n" + \
               "10,ideas,\"[None, None, None]\"\n" + \
               "10,sleep,\"[0.1, 0.2, 0.3]\"\n"
    assert (result == expected)
