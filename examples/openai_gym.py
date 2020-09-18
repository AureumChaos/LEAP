"""This example uses a Pitt-approach rule system to evolve agent controllers
for the problems in the OpenAI Gym problem set. """
import sys

import click
import gym
from matplotlib import pyplot as plt
import numpy as np

from leap_ec.context import context
from leap_ec.executable_rep import problem, rules, executable
from leap_ec.individual import Individual
from leap_ec import probe, ops
from leap_ec.representation import Representation
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.algorithm import generational_ea


@click.group()
def cli():
    """This LEAP example application uses a Pitt-approach rule system to
    evolve agent controllers for the problems in the OpenAI Gym problem set. """
    pass


@cli.command()
def list_envs():
    """List all available OpenAI Gym environments."""
    for env in gym.envs.registry.all():
        print(env.id)


@cli.command()
@click.option('--runs', default=100,
              help='Number of independent times to run the environment.')
@click.option('--steps', default=100,
              help='Max number of steps to run the environment for.')
@click.option('--env', default='CartPole-v0',
              help='The OpenAI Gym environment to run.')
def keyboard(runs, steps, env):
    """Run an environment and control it with the keyboard."""
    environment = gym.make(env)
    controller = executable.KeyboardExecutable(
                 environment.observation_space,
                 environment.action_space)

    # Wire our controller's keyboard function into the viewer's events
    environment.render()
    environment.unwrapped.viewer.window.on_key_press = controller.key_press
    environment.unwrapped.viewer.window.on_key_release = controller.key_release

    # Hand over control of the environment to the Problem
    p = problem.ExecutableProblem(runs, steps, environment, 'reward')
    print(p.evaluate(controller))


@cli.command()
@click.option('--runs', default=100,
              help='Number of independent times to run the environment.')
@click.option('--steps', default=100,
              help='Max number of steps to run the environment for.')
@click.option('--env', default='CartPole-v0',
              help='The OpenAI Gym environment to run.')
def random(runs, steps, env):
    """Run an environment controlled by a random `Executable`."""
    environment = gym.make(env)
    controller = executable.RandomExecutable(
                 environment.observation_space,
                 environment.action_space)
    p = problem.ExecutableProblem(runs, steps, environment, 'reward')
    print(p.evaluate(controller))


@cli.command()
@click.option('--runs', default=1,
              help='Number of independent times to run the environment per '
                   'fitness eval.')
@click.option('--steps', default=500,
              help='Max number of steps to run the environment for each run.')
@click.option('--env', default='CartPole-v0',
              help='The OpenAI Gym environment to run.')
@click.option('--evals', default=100, help='Fitness evaluations to run for')
@click.option('--pop-size', default=5, help='Population size')
@click.option('--num-rules', default=10,
              help='Number of rules to use in the Pitt-style genome.')
@click.option('--mutate-std', default=0.05,
              help='Standard deviation of Gaussian mutation')
@click.option('--output', default='./genomes.csv',
              help='File to record best-of-gen genomes & fitness to.')
@click.option('--gui/--no-gui', default=True,
              help='Toggle GUI visualization of each simulation.')
def evolve_pitt(runs, steps, env, evals, pop_size,
                num_rules, mutate_std, output, gui):
    """Evolve a controller using a Pitt-style rule system."""

    print(f"Loading environment '{env}'...")
    environment = gym.make(env)
    num_inputs = int(np.prod(environment.observation_space.shape))
    print(f"# of Inputs: {num_inputs}")
    num_outputs = int(np.prod(environment.action_space.shape))
    print(f"# of Outputs: {num_inputs}")
    stdout_probe = probe.FitnessStatsCSVProbe(context, stream=sys.stdout)

    with open(output, 'w') as genomes_file:
        file_probe = probe.AttributesCSVProbe(
            context,
            stream=genomes_file,
            do_fitness=True,
            do_genome=True)
        plt.figure()
        plt.ylabel("Fitness")
        plt.xlabel("Generations")
        plt.title("Best-of-Generation Fitness")
        fitness_viz_probe = probe.PopulationPlotProbe(
            context, ylim=(
                0, 1), xlim=(
                0, 1), modulo=1, ax=plt.gca())
        ea = generational_ea(generations=evals, pop_size=pop_size,
                             # Solve a problem that executes agents in the
                             # environment and obtains fitness from it
                             problem=problem.ExecutableProblem(
                                 runs, steps, environment, 'reward', gui),

                             representation=Representation(
                                 decoder=rules.PittRulesDecoder(  # Decode genomes into Pitt-style rules
                                     input_space=environment.observation_space,
                                     output_space=environment.action_space,
                                     priority_metric=rules.PittRulesExecutable.PriorityMetric.RULE_ORDER,
                                     num_memory_registers=0
                                 ),

                                 initialize=create_real_vector(  # Initialized genomes are random real-valued vectors.
                                     # Initialize each element between 0 and 1.
                                     bounds=(
                                         [[-0.0, 1.0]] * (num_inputs * 2 + num_outputs)) * num_rules
                                 )
                             ),

                             # The operator pipeline.
                             pipeline=[
                                 ops.tournament,
                                 ops.clone,
                                 mutate_gaussian(
                                     std=mutate_std, hard_bounds=(0, 1)),
                                 ops.evaluate,
                                 ops.pool(size=pop_size),
                                 stdout_probe,
                                 fitness_viz_probe
                             ])
        list(ea)


@cli.command()
@click.option('--runs', default=1,
              help='Number of independent times to run the environment per fitness eval.')
@click.option('--steps', default=500,
              help='Max number of steps to run the environment for each run.')
@click.option('--env', default='CartPole-v0',
              help='The OpenAI Gym environment to run.')
@click.argument('rules')
def run_pitt(runs, steps, env, rules):
    """
    Used a fixed Pitt-style ruleset to control an agent in the given
    environment.

    Takes ruleset to use as the controller as the RULES argument, in the form
    of a string, ex. \"c1, c1\',  c2, c2\', ... cn, cn\',  a1, ... am, m1,
    ... mr\"
    """
    # Convert rules string into a list of floats
    rules = list(map(float, rules.split(',')))
    environment = gym.make(env)
    decoder = rules.PittRulesDecoder(
        input_space=environment.observation_space,
        output_space=environment.action_space,
        priority_metric=rules.PittRulesExecutable.PriorityMetric.RULE_ORDER,
        num_memory_registers=0
    )
    controller = rules.decode(rules)
    problem = problem.ExecutableProblem(
        runs, steps, environment, 'survival')
    print(problem.evaluate(controller))


if __name__ == '__main__':
    cli()
