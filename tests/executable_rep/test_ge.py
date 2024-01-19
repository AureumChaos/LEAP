"""Unit tests for the grammar.py module."""
from io import StringIO

import pytest

from leap_ec.executable_rep.ge import *


@pytest.fixture
def arithmetic_grammar():
    grammar_str = """
    <expression> ::= <expression><op><expression>
                   | (<expression>)
                   | <variable>
                   | <constant>
    <variable> ::= x | y | z
    <constant> ::= GE_RANGE:20
    <op> ::= + | - | * | /
    """
    return Grammar(
        file_stream=StringIO(grammar_str),
        num_permutation_ramps=10 # XXX Using an arbitrary value here until I can work out what this parameter does
    )


def test_grammar_numsymbols_arithmetic(arithmetic_grammar):
    """Ensure that the correct number of terminals and non-terminals are 
    identified in the arithmetic test grammar."""

    # Expect 4 non-terminals: <expression>, <variable>, <constant>, and <op>
    assert 4 == len(arithmetic_grammar.non_terminals), f"{arithmetic_grammar.non_terminals}"
    # Expect 29 terminals: x, y, z, +, -, *, /, (, ), 0, 1, ..., 19
    assert 29 == len(arithmetic_grammar.terminals), f"{arithmetic_grammar.terminals}"


def test_number_of_grammar_rules_arithmetic(arithmetic_grammar):
    """Ensure that rules (i.e. sets of choices) for all 4 non-terminals are parsed."""

    assert 4 == len(arithmetic_grammar.rules)
    

def test_grammar_min_steps_arithmetic(arithmetic_grammar):
    """Ensure that the minimum number of steps between each non-terminal and 
    the nearest terminal are correctly identified in the arithmetic test
    grammar."""
    expected_min_steps = {
        '<expression>': 2,
        '<variable>': 1,
        '<constant>': 1,
        '<op>': 1
    }

    result = { nt: nt_dict['min_steps'] for nt, nt_dict in arithmetic_grammar.non_terminals.items() }

    assert expected_min_steps == result, f"{result}"


def test_grammar_recursive_arithmetic(arithmetic_grammar):
    """Ensure that each choice for expanding each non-terminal is
     correctly identified as being recursive or not."""
    expected_recursive = {
        '<expression>': [True, True, False, False ],
        '<variable>': [ False, False, False ],
        '<constant>': 20*[False],
        '<op>': 4*[False]
    }

    result = {}
    for k, v in arithmetic_grammar.rules.items():
        choices = v['choices']
        result[k] = [ choice_dict['recursive'] for choice_dict in choices ]

    assert expected_recursive == result, f"{result}"