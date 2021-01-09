"""Manual invocation of doctests, because while PyTest will run them
automatically, tools such as Coveralls.io will not."""
import doctest
from importlib import import_module
import pkgutil

import leap_ec

if __name__ == '__main__':
    # Walk all of LEAP's modules
    for _, module_name, _ in pkgutil.walk_packages(leap_ec.__path__, leap_ec.__name__ + '.'):
        # Load the module & run its doctests
        mod = import_module(module_name)
        doctest.testmod(mod, optionflags=doctest.ELLIPSIS)
