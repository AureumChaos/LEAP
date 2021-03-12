# LEAP SOFTWARE DEVELOPMENT GUIDELINES

This document provides guidance on developing for this project.

## PEP8 style compliance

Code should be [PEP8](https://www.python.org/dev/peps/pep-0008/) compliant, 
wherever possible.  However, we're a little relaxed when it comes to 
whitespaces, and enforcing wrapping at column 80 for block comments.  The 
latter addresses situations where URLs and paths just won't nicely fit 
within 80 columns.

We recommend the [`flake8`](https://pypi.org/project/flake8/) PEP8 linter.

## Modules, classes, and functions should have docstrings.

Essentially, everything that can have a docstring, should have a docstring, 
which mostly captures the spirit of [PEP 257](https://www.python.org/dev/peps/pep-0257/)
for new code.

## We encourage use of Rich Structure Formatted descriptors in docstrings

[PEP 287](https://www.python.org/dev/peps/pep-0287/) specifies use of reStructuredText Docstring formats for docstrings, 
though it doesn't do a good job of specifying on standardizing docstring 
content.  

Generally, for functions we try to document them with this docstring pattern:

```python
def foo(a, b, c):
    """ One liner summary for foo
    
        :param a: does this
        :param b: does that
        :param c: does something else
        :returns: a * b * c
    """
    return a * b * c
```

## We encourage use of type hints
[PEP 484](https://www.python.org/dev/peps/pep-0484/) describes the python3 type 
hint syntax, and we encourage its use.  

(However, we admit at the time of this writing that we, ourselves, are 
inconsistent in the LEAP implementation, and which we will address at a 
later point.)

## We encourage inclusion of comment blocks before functions and classes
We have an admittedly idiosyncratic standard for including a comment block 
of this form before functions and classes:

```python
##############################
# Class ExampleClass
##############################
```

And:

```python
##############################
# Function foo()
##############################
```

Some editors support a summary window of the entire file, such as sublime 
and PyCharm, and sometimes those types of comment blocks stand out in those 
windows to make it easier to pick out module organization.

(We confess we're not consistent in doing this ourselves, but we're trying 
to get better at enforcement.)

## We encourage the use of doctests

[Doctests](https://docs.python.org/3/library/doctest.html) are a handy way to not only provide examples of use, but provide a 
simple unit test for a given function.  With that in mind, we also encourage 
the writing of doctests for any new functions.

## We encourage implementation of unittest

We also encourage the inclusion of unittests that will be regularly 
exercised in our CI pipeline after pushing to the central repository.  You 
can see examples of existing unit tests in `./tests`.

Note that we also have stochastic unit tests, which are important for 
evolutionary algorigthms because they're inherently stochastic.

## Use `leap_ec.context` to track state

`leap_ec.context` to track state that needs to persist outside, say, pipeline
operators or function invocations.  If you create a new operator or that
function that relies on `leap_ec.context`, please make it the last argument
and have it default to `leap_ec.context.context`.


## Add an optional `key` argument for new selection operators

Add an optional `key` argument for selection operators as seen for `max()` 
and `sort()` because this allows for passing in functions for changing the
default selection criteria.  One common use for this is to add in parsimony
pressure.
