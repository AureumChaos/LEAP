"""
    Unit tests for inc_generation()
"""
from leap_ec.global_vars import context
from leap_ec.util import inc_generation

def test_inc_generation():
    """ Test the inc_generation() function.
    """
    my_inc_generation = inc_generation(context)

    # The core context is set to the current generation, which will start as zero
    assert context['leap']['generation'] == 0

    # We can also directly query the internal state of the generation incrementer
    assert my_inc_generation.generation() == 0

    # Now do the increment
    curr_generation = my_inc_generation()

    # The generation incrementer always returns the new generation
    assert curr_generation == 1

    # The core context state should have updated
    assert context['leap']['generation'] == 1

    # And, of course, the internal state will reflect that reality, too
    assert my_inc_generation.generation() == 1


def test_inc_generation_callbacks():
    """ Test inc_generation() callback support
    """
    def callback(new_generation):
        assert new_generation == 1

    my_inc_generation = inc_generation(context, callbacks=[callback])

    # Incremented the generation should call our test callback
    my_inc_generation()

