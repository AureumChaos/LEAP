def print_list(l):
    """
    Return a string representation of a list.

    This uses __str__() to resolve the elements of the list:

    >>> from leap.core import Individual
    >>> l = [Individual([0, 1, 2]), Individual([3, 4, 5])]
    >>> print_list(l)
    [[0, 1, 2], [3, 4, 5]]

    As opposed to the standard printing mechanism, which calls __repr__() on the elements to produce

    >>> print(l)
    [Individual([0, 1, 2], None, None), Individual([3, 4, 5], None, None)]

    :param l:
    :return:
    """
    print('[' + ', '.join([x.__str__() for x in l]) + ']')
