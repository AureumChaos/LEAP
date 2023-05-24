"""
Run all of our example algorithms and ensure they have no exceptions.
"""
import pathlib
import runpy
import subprocess
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
    os.environ[test_env_var] = 'True'
    proc = subprocess.run([sys.executable, str(script)], stdout=sys.stdout, stderr=sys.stderr)
    assert proc.returncode == 0, f"Script {script} returned non-zero exit status {proc.returncode}"