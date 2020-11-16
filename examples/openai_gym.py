"""An example of a complete program that uses LEAP to learn controllers
for various robotics and behavior tasks provided by OpenAI Gym."""
import sys

import click
from dask.distributed import Client
import gym
from matplotlib import pyplot as plt
import numpy as np

from leap_ec.context import context
from leap_ec.distributed import synchronous
from leap_ec.executable_rep import problems, rules, neural_network, executable
from leap_ec.individual import Individual
from leap_ec import probe, ops
from leap_ec.representation import Representation
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.algorithm import generational_ea


@click.group()
def cli():
    """This LEAP example program evolves agent controllers for the problems 
    in the OpenAI Gym problem set.  It supports a couple of different 
    evolutionary representations.
    
    Note that you may need to install OpenGL packages for your system (such as 
    'freeglut' or 'glu' in order for the visualizations to work."""
    pass


##############################
# List command
##############################
@cli.command(name='list')
def list_command():
    """List all available OpenAI Gym environments."""
    for env in gym.envs.registry.all():
        print(env.id)


##############################
# Evolve command
##############################
@cli.command()
@click.option('--runs', default=5,
              help='Number of independent times to run the environment per '
                   'fitness eval.')
@click.option('--steps', default=500,
              help='Max number of steps to run the environment for each run.')
@click.option('--env', default='CartPole-v0',
              help='The OpenAI Gym environment to run.')
@click.option('--rep', default='neural',
              help="The evolutionary representation to use ('neural' or 'pitt')")
@click.option('--gens', default=100, help='Generations to run for')
@click.option('--pop-size', default=5, help='Population size')
@click.option('--num-nodes', default=0,
              help='Number of rules (if Pitt-style) or hidden notes (if neural) to use in the controller.')
@click.option('--mutate-std', default=0.05,
              help='Standard deviation of Gaussian mutation')
@click.option('--output', default='./genomes.csv',
              help='File to record best-of-gen genomes & fitness to.')
@click.option('--gui/--no-gui', default=True,
              help='Toggle GUI visualization of each simulation.')
def evolve(runs, steps, env, rep, gens, pop_size,
                num_nodes, mutate_std, output, gui):
    """Evolve a controller using a Pitt-style rule system."""
    check_rep(rep)

    print(f"Loading environment '{env}'...")
    environment = gym.make(env)
    print(f"\tObservation space:\t{environment.observation_space}")
    print(f"\tAction space:     \t{environment.action_space}")

    with open(output, 'w') as genomes_file:
        if rep == 'pitt':
            representation = pitt_representation(environment, num_rules=num_nodes)
        elif rep == 'neural':
            representation = neural_representation(environment, num_hidden_nodes=num_nodes)

        probes = get_probes(genomes_file, environment, rep)

        from leap_ec.problem import ConstantProblem
        with Client() as dask_client:
            ea = generational_ea(generations=gens, pop_size=pop_size,
                                # Solve a problem that executes agents in the
                                # environment and obtains fitness from it
                                #problem=ConstantProblem(),
                                problem=problems.EnvironmentProblem(
                                    runs, steps, environment, 'reward', gui),

                                representation=representation,

                                # The operator pipeline.
                                pipeline=[
                                    ops.tournament_selection,
                                    ops.clone,
                                    mutate_gaussian(
                                        std=mutate_std, hard_bounds=(-1, 1)),
                                    ops.evaluate,
                                    ops.pool(size=pop_size),
                                    #synchronous.eval_pool(client=dask_client, size=pop_size),
                                    *probes  # Inserting all the probes at the end
                                ])
            list(ea)


def check_rep(rep):
    """Input validation on the representation option."""
    allowed_reps = ['neural', 'pitt']
    if rep not in allowed_reps:
        raise ValueError(f"Unrecognized representation '{rep}'.  Must be one of '{allowed_reps}'.")


def pitt_representation(environment, num_rules):
    """Return a Pitt-approach rule system representation suitable for
    learning a controller for this environment."""
    num_inputs = int(np.prod(environment.observation_space.shape))
    num_outputs = int(np.prod(environment.action_space.shape))

    # Decode genomes into Pitt-style rules
    decoder = rules.PittRulesDecoder(  
                    input_space=environment.observation_space,
                    output_space=environment.action_space,
                    priority_metric=rules.PittRulesExecutable.PriorityMetric.RULE_ORDER,
                    num_memory_registers=0)

    # Initialized genomes are random real-valued vectors.
    initialize = create_real_vector(  
                    # Initialize each element between 0 and 1.
                    bounds=([[0.0, 1.0]] * (num_inputs * 2 + num_outputs)) * num_rules)

    return Representation(decoder, initialize)
    

def neural_representation(environment, num_hidden_nodes):
    """Return a neural network representation suitable for learning a
    controller for this environment."""
    num_inputs = int(np.prod(environment.observation_space.shape))
    num_actions = environment.action_space.n

    # Decode genomes into a feed-forward neural network,
    # but also wrap an argmax around the networks so their
    # output is a single integer
    decoder = executable.WrapperDecoder(
                wrapped_decoder=neural_network.SimpleNeuralNetworkDecoder(shape=(num_inputs, num_hidden_nodes, num_actions)),
                decorator=executable.ArgmaxExecutable)

    # Initialized genomes are random real-valued vectors.
    initialize = create_real_vector(bounds=([[-1, 1]]*decoder.wrapped_decoder.length))

    return Representation(decoder, initialize)


def get_probes(genomes_file, environment, rep):
    """Set up probes for writings results to file and terminal and
    displaying live metric plots."""
    assert(genomes_file is not None)
    assert(environment is not None)
    assert(rep is not None)
    assert(rep in ['neural', 'pitt'])
    num_inputs = int(np.prod(environment.observation_space.shape))
    num_outputs = int(np.prod(environment.action_space.shape))

    probes = []

    # Print fitness stats to stdout
    probes.append(probe.FitnessStatsCSVProbe(context, stream=sys.stdout))

    # Save genome of the best individual to a file
    probes.append(probe.AttributesCSVProbe(
                  context,
                  stream=genomes_file,
                  best_only =True,
                  do_fitness=True,
                  do_genome=True))

    # Open a figure to plot a fitness curve to
    plt.figure()
    plt.ylabel("Fitness")
    plt.xlabel("Generations")
    plt.title("Best-of-Generation Fitness")
    probes.append(probe.PopulationPlotProbe(
                    context, ylim=(
                        0, 1), xlim=(
                        0, 1), modulo=1, ax=plt.gca()))
    
    # Open a figure to plot a projection of the best Pitt rules to
    if rep == 'pitt':
        plt.figure()
        plt.ylabel("Sensor 1")
        plt.xlabel("Sensor 0")
        plt.title("Rule Coverage")
        probes.append(rules.PlotPittRuleProbe(context, num_inputs, num_outputs, (0, 1), ax=plt.gca()))

    return probes


##############################
# Run command
##############################
@cli.command()
@click.option('--evals', default=100,
              help='Number of fitness evaluations to collect.')
@click.option('--runs', default=1,
              help='Number of independent times to run the environment per fitness eval.')
@click.option('--steps', default=500,
              help='Max number of steps to run the environment for each run.')
@click.option('--rep', default='neural',
              help="The evolutionary representation to use ('neural' or 'pitt')")
@click.option('--env', default='CartPole-v0',
              help='The OpenAI Gym environment to run.')
@click.option('--num-nodes', default=0,
              help='Number of rules (if Pitt-style) or hidden notes (if neural) to use in the controller.')
@click.option('--gui/--no-gui', default=True,
              help='Toggle GUI visualization of each simulation.')
def run(evals, runs, steps, rep, env, num_nodes, gui):
    """
    Load the parameters defining a controller on stdin into an agent to drive 
    it in the given environment.
    """
    check_rep(rep)
    rule_string = sys.stdin.readlines()[0]
    # Convert parameter string into a list of floats
    rule_set = list(map(float, rule_string.split(',')))

    environment = gym.make(env)
    if rep == 'neural':
        decoder = neural_representation(environment, num_hidden_nodes=num_nodes).decoder
    elif rep == 'pitt':
        decoder = pitt_representation(environment, num_rules=num_nodes).decoder

    controller = decoder.decode(rule_set)
    for i in range(evals):
        p = problems.EnvironmentProblem(runs, steps, environment, 'reward', gui)
        print(p.evaluate(controller))


##############################
# Keyboard command
##############################
@cli.command()
@click.option('--evals', default=10,
              help='Number of independent times to run the environment.')
@click.option('--steps', default=500,
              help='Max number of steps to run the environment for.')
@click.option('--env', default='CartPole-v0',
              help='The OpenAI Gym environment to run.')
def keyboard(evals, steps, env):
    """Run an environment and control it with the keyboard."""
    environment = gym.make(env)
    controller = executable.KeyboardExecutable(
                    environment.observation_space,
                    environment.action_space)

    for i in range(evals):
        # Wire our controller's keyboard function into the viewer's events
        environment.render()
        environment.unwrapped.viewer.window.on_key_press = controller.key_press
        environment.unwrapped.viewer.window.on_key_release = controller.key_release
        # Hand over control of the environment to the Problem
        p = problems.EnvironmentProblem(1, steps, environment, 'reward', gui=True)
        print(p.evaluate(controller))
    
    environment.close()


##############################
# Random command
##############################
@cli.command()
@click.option('--evals', default=10,
              help='Number of fitness evaluations to collect.')
@click.option('--runs', default=1,
              help='Number of independent times to run the environment.')
@click.option('--steps', default=500,
              help='Max number of steps to run the environment for.')
@click.option('--env', default='CartPole-v0',
              help='The OpenAI Gym environment to run.')
def random(evals, runs, steps, env):
    """Run an environment controlled by a random `Executable`."""
    environment = gym.make('CartPole-v0')
    controller = executable.RandomExecutable(
                    environment.observation_space,
                    environment.action_space)
    for i in range(evals):
        p = problems.EnvironmentProblem(runs, steps, environment, 'reward', guit=True)
        print(p.evaluate(controller))


if __name__ == '__main__':
    cli()
