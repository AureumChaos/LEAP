import sys

import click
import gym
import numpy as np

from leap import brains, core, real, probe
from leap import operate as op
from leap.example.simple_ea import simple_ea


@click.group()
def cli():
    pass


@cli.command()
def list_envs():
    """List all available OpenAI Gym environments."""
    for env in gym.envs.registry.all():
        print(env.id)


@cli.command()
@click.option('--runs', default=100, help='Number of independent times to run the environment.')
@click.option('--steps', default=100, help='Max number of steps to run the environment for.')
@click.option('--env', default='CartPole-v0', help='The OpenAI Gym environment to run.')
def keyboard(runs, steps, env):
    """Run an environment and control it with the keyboard."""
    environment = gym.make(env)
    brain = brains.KeyboardBrain(environment.observation_space, environment.action_space)

    # Wire our brain's keyboard function into the viewer's events
    environment.render()
    environment.unwrapped.viewer.window.on_key_press = brain.key_press
    environment.unwrapped.viewer.window.on_key_release = brain.key_release

    # Hand over control of the environment to the Problem
    p = brains.BrainProblem(runs, steps, environment, brains.reward_fitness)
    print(p.evaluate(brain))


@cli.command()
@click.option('--runs', default=100, help='Number of independent times to run the environment.')
@click.option('--steps', default=100, help='Max number of steps to run the environment for.')
@click.option('--env', default='CartPole-v0', help='The OpenAI Gym environment to run.')
def random(runs, steps, env):
    """Run an environment controlled by a random brain."""
    environment = gym.make(env)
    brain = brains.RandomBrain(environment.observation_space, environment.action_space)
    p = brains.BrainProblem(runs, steps, environment, brains.reward_fitness)
    print(p.evaluate(brain))


@cli.command()
@click.option('--runs', default=1, help='Number of independent times to run the environment per fitness eval.')
@click.option('--steps', default=500, help='Max number of steps to run the environment for each run.')
@click.option('--env', default='CartPole-v0', help='The OpenAI Gym environment to run.')
@click.option('--evals', default=100, help='Fitness evaluations to run for')
@click.option('--pop-size', default=5, help='Population size')
@click.option('--num-rules', default=10, help='Number of rules to use in the Pitt-style genome.')
@click.option('--mutate-prob', default=0.1, help='Per-gene Gaussian mutation rate')
@click.option('--mutate-std', default=0.05, help='Standard deviation of Gaussian mutation')
def evolve_pitt(runs, steps, env, evals, pop_size, num_rules, mutate_prob, mutate_std):
    """Evolve a controller using a Pitt-style LCS."""

    environment = gym.make(env)
    num_inputs = int(np.prod(environment.observation_space.shape))
    num_outputs = int(np.prod(environment.action_space.shape))
    csv_probe = probe.CSVFitnessStatsProbe(sys.stdout)

    ea = simple_ea(evals=evals, pop_size=pop_size,
                   individual_cls=core.Individual,  # Use the standard Individual as the prototype for the population.
                   decoder=brains.PittRulesDecoder(
                       input_space=environment.observation_space,
                       output_space=environment.action_space,
                       priority_metric=brains.PittRulesBrain.PriorityMetric.RULE_ORDER,
                       num_memory_registers=0
                   ),
                   problem=brains.BrainProblem(runs, steps, environment, brains.survival_fitness),
                   evaluate=op.evaluate,  # Evaluate fitness with the basic evaluation operator.

                   # Initialized genomes are random real-valued vectors.
                   initialize=real.initialize_vectors_uniform(
                       # Initialize each element between 0 and 1.
                       bounds=([[-0.0, 1.0]] * (num_inputs*2 + num_outputs)) * num_rules
                   ),

                   # Step notification for our CSV probe
                   step_notify_list=[csv_probe.set_step],

                   # The operator pipeline.
                   pipeline=[
                       csv_probe,
                       # Select mu parents via tournament selection.
                       op.tournament(n=pop_size),
                       # Clone them to create offspring.
                       op.cloning,
                       # Apply Gaussian mutation to each gene with a certain probability.
                       op.mutate_gaussian(prob=mutate_prob, std=mutate_std)
                   ])
    list(ea)


if __name__ == '__main__':
    cli()
