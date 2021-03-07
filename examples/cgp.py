"""Example demonstrating the use of Cartesion genetic programming (CGP) to 
evolve logic circuits to solve Boolean functions.

This application provides both an evolutionary CGP solver, and an alternative
random search algorithm, so that the two may be compared.
"""
import sys

import click
from matplotlib import pyplot as plt

from leap_ec.algorithm import generational_ea, random_search
from leap_ec import ops, probe
from leap_ec.representation import Representation
from leap_ec.executable_rep import cgp, neural_network, problems


##############################
# CGP components
##############################

# The CGPDecoder is the heart of our CGP representation.
# We'll set it up first because it's needed as a parameter 
# to a few different components.
cgp_decoder = cgp.CGPDecoder(
                    primitives=[
                        lambda x, y: not (x and y),  # NAND
                        lambda x, y: not x,  # NOT (ignoring y)
                    ],
                    num_inputs = 2,
                    num_outputs = 1,
                    num_layers=50,
                    nodes_per_layer=1,
                    max_arity=2
                )


xor_problem = problems.TruthTableProblem(
                    boolean_function=lambda x: [ x[0] ^ x[1] ],  # XOR
                    num_inputs = 2,
                    num_outputs = 1
                )


cgp_representation = Representation(
                        decoder=cgp_decoder,
                        # We use a sepecial initializer that obeys the CGP constraints
                        initialize=cgp.create_cgp_vector(cgp_decoder)
                    )


def cgp_visual_probes(modulo):
    """Set up the graphical probes that we'll use."""
    plt.figure()
    p1 = probe.PopulationPlotProbe(modulo=modulo, ax=plt.gca())
    plt.figure()
    p2 = neural_network.GraphPhenotypeProbe(modulo=modulo, ax=plt.gca())
    return [ p1, p2 ]


##############################
# cli entry point
##############################
@click.group()
def cli():
    """Example of Cartesian Genetic Programming."""


##############################
# cgp command
##############################
@cli.command(name='cgp')
@click.option('--gens', default=1000)
def cgp_cmd(gens):
    """Use an evolutionary CGP approach to solve the XOR function."""
    pop_size = 5

    ea = generational_ea(gens, pop_size, 

            representation=cgp_representation,

            # Our fitness function will be to solve the XOR problem
            problem=xor_problem,

            pipeline=[
                ops.tournament_selection,
                ops.clone,
                cgp.cgp_mutate(cgp_decoder, expected_num_mutations=1),
                ops.evaluate,
                ops.pool(size=pop_size),
                probe.FitnessStatsCSVProbe(stream=sys.stdout)
            ] + cgp_visual_probes(modulo=10)
    )

    list(ea)


##############################
# random command
##############################
@cli.command('random')
@click.option('--evals', default=5000)
def random(evals):
    """Use random search over a CGP representation to solve the XOR function."""
    ea = random_search(evals, 
            representation=cgp_representation,

            # Our fitness function will be to solve the XOR problem
            problem=xor_problem,

            pipeline=[
                probe.FitnessStatsCSVProbe(stream=sys.stdout)
            ] + cgp_visual_probes(modulo=10)
    )

    list(ea)


##############################
# main
##############################
if __name__ == '__main__':
    cli()
