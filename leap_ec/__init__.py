from .__version__ import __version__
from .individual import Individual
from .representation import Representation
from .decoder import Decoder
from .global_vars import context

"""
This defines the name of the logger LEAP modules use.
"""
leap_logger_name = 'leap_ec'

"""
The environment variable we use to signal that our
test harness is being run.
"""
test_env_var = 'LEAP_TESTING'