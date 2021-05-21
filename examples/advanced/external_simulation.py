"""
This examples demonstrates tuning the paramaters of an external 
simulation by use of ExternalProcessProblem.

Each fitness evaluation launches an external command that you provide,
and sends a phenome to it as a CSV on stdin.  The external command
returns a real number on its stdout, which LEAP reads as the fitness
value.
"""
"""An example of using the LEAP framework to implement an
evolutionary algorithm for tuning CARLsim models.
"""
import logging
import sys

from leap_ec import ops
from leap_ec import probe
from leap_ec.algorithm import generational_ea
from leap_ec import leap_logger_name
from leap_ec.problem import ExternalProcessProblem
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.representation import Representation
from matplotlib import pyplot as plt


if __name__ == '__main__':
    ##############################
    # Parameters
    ##############################
    plots = True
    generations = 100
    pop_size = 50
    num_genes = 4
    max_individuals_per_chunk = 25

    ##############################
    # Problem
    ##############################
    # The ExternalProcessProblem wraps our CARLsim executable in a
    # fitness function that launches the given command and pipes
    # individuals into it in CSV format.
    problem = ExternalProcessProblem(executable_path='python -c "print(0.1)"', maximize=True)
    problem.bounds = [0, 1]  # Setting the min and max values that genes are allowed to assume

    # Uncomment these lines to see logs of what genomes and fitness values are sent to your external process.
    # This is useful for debugging a simulation.
    logging.getLogger().addHandler(logging.StreamHandler())  # Log to stderr
    logging.getLogger(leap_logger_name).setLevel(logging.DEBUG) # Log debug messages

    ##############################
    # Visualization probes
    ##############################
    # We set up visualizations in advance so that we can arrange them nicely
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    pop_probe = probe.CartesianPhenotypePlotProbe(xlim=problem.bounds, ylim=problem.bounds, pad=[0.1, 0.1], ax=plt.gca())
    plt.subplot(122)
    fit_probe = probe.FitnessPlotProbe(ylim=[0, 0.1], ax=plt.gca())
    viz_probes = [pop_probe, fit_probe]

    ##############################
    # The Evolutionarly Algorithm
    ##############################
    ea = generational_ea(generations=generations, pop_size=pop_size,
                            problem=problem,  # Fitness function
                            
                            # By default, the initial population is evaluated one-at-a-time.
                            # Passing group_evaluate into init_evaluate evaluates the population in batches.
                            init_evaluate=ops.grouped_evaluate(problem=problem, max_individuals_per_chunk=max_individuals_per_chunk),

                            # Representation
                            representation=Representation(
                                # Initialize a population of integer-vector genomes
                                initialize=create_real_vector(
                                    bounds=[problem.bounds] * num_genes)
                            ),

                            # Operator pipeline
                            pipeline=[
                                ops.tournament_selection(k=2),
                                ops.clone,  # Copying individuals before we change them, just to be safe
                                mutate_gaussian(std=0.2, hard_bounds=[problem.bounds]*num_genes,
                                                expected_num_mutations=1),
                                ops.pool(size=pop_size),
                                # Here again, we use grouped_evaluate to send chunks of individuals to the ExternalProcessProblem.
                                ops.grouped_evaluate(problem=problem, max_individuals_per_chunk=max_individuals_per_chunk),
                                # Print fitness statistics to stdout at each genration
                                probe.FitnessStatsCSVProbe(stream=sys.stdout)
                            ] + (viz_probes if plots else [])
                        )

    best_inds = list(ea)
