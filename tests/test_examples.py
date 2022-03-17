"""
Run all of our example algorithms and ensure they have no exceptions.
"""
import pathlib
import runpy
import os
import sys

import pytest

from leap_ec import test_env_var


scripts = pathlib.Path(__file__, '..', '..', 'examples').resolve().rglob('*.py')


##############################
# Tests for example scripts
##############################
@pytest.mark.parametrize('script', scripts)
def test_script_execution(script):
    print(f"Running {script}")
    # We have to tweak sys.argv to avoid the pytest arguments from being passed along to our scripts
    sys_orig = sys.argv
    sys.argv = [ str(script) ]
    os.environ[test_env_var] = 'True'

    try:
        runpy.run_path(str(script), run_name='__main__')
    except SystemExit as e:
        # Some scripts may explicitly call `sys.exit()`, in which case we'll check the error code
        assert(e.code == 0)
    
    sys.argv = sys_orig
