"""
Run all of our example algorithms and ensure they have no exceptions.
"""
import pathlib
import runpy
import sys

import pytest


scripts = pathlib.Path(__file__, '..', '..', 'examples').resolve().rglob('*.py')


##############################
# Tests for example scripts
##############################
@pytest.mark.slow
@pytest.mark.parametrize('script', scripts)
def test_script_execution(script):
    print(f"Running {script}")
    # We have to tweak sys.argv to avoid the pytest arguments from being passed along to our scripts
    sys_orig = sys.argv
    sys.argv = [ str(script) ]

    try:
        runpy.run_path(str(script), run_name='__main__')
    except SystemExit as e:
        # Some scripts may explicitly call `sys.exit()`, in which case we'll check the error code
        assert(e.code == 0)
    
    sys.argv = sys_orig
