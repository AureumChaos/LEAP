"""
    This is the simplest example of using LEAP, where one can rely on the
    very high-level function, ea_solve(), to optimize the given real-valued
    function.
"""
import os

from leap_ec import test_env_var
from leap_ec.simple import ea_solve

# When running the test harness, just run for two generations
# (we use this to quickly ensure our examples don't get bitrot)
if os.environ.get(test_env_var, False) == 'True':
    generations = 2
    viz = False
else:
    generations = 100
    viz = True


##############################
# Fitness function
##############################
def function(values):
    """A simple fitness function, evaluating the sum of squared parameters."""
    return sum([x ** 2 for x in values])


##############################
# Entry point
##############################
if __name__ == '__main__':
    ea_solve(function,
             generations=generations,
             bounds=[(-5.12, 5.12) for _ in range(5)],
             viz=viz,
             mutation_std=0.1)
