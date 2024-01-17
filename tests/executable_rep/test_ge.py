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

def test_grammar_rules_arithmeteic(arithmetic_grammar):
    """Ensure that the correct production rules available for each 
    non-terminal are correctly identified in the arithmetic test grammar."""

    #result = { nt: nt_dict['choices'] for nt, nt_dict in arithmetic_grammar.non_terminals.items() }

    # FIXME Where I left off.  I was inspecting the format of the rules dict manually to understand how
    #       it is supposed to be structured.
    assert {} == arithmetic_grammar.rules, f"{arithmetic_grammar.rules}"
    

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
    """Ensure that the recursive rules are correclty identified in the
    arithmetic test grammar."""
    expected_recursive = {
        '<expression>': True,
        '<variable>': False,
        '<constant>': False,
        '<op>': False
    }

    # FIXME: This is the wrong data to pull.  `recursive` is set on the rules, not the NT dict
    result = { nt: nt_dict['recursive'] for nt, nt_dict in arithmetic_grammar.non_terminals.items() }

    assert expected_recursive == result, f"{result}"