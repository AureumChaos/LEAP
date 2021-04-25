
from leap_ec.algorithm import generational_ea
from leap_ec.decoder import IdentityDecoder
from leap_ec.executable_rep import problems, executable, neural_network
from leap_ec.individual import Individual
from leap_ec.real_rep.problems import SpheroidProblem, SchwefelProblem
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.representation import Representation
import leap_ec.ops as ops


##############################
# Class FunctionDecoder
##############################
class FunctionDecoder():
    """A Decoder that takes a genome and applies a given callable to it to produce
    a phenotype.

    Our interest in this is that it allows us to easily wrap other objects---such as
    an evolved executable object---to produce a decoder.  So it allows things like
    a representation that itself evolves.
    """
    def __init__(self, f):
        assert(callable(f))
        self.f = f

    def decode(self, genome):
        assert(genome is not None)
        return self.f(genome)

    @staticmethod
    def from_function(f):
        return FunctionDecoder(f)


##############################
# Class AdaptiveRepresentationDecoder
##############################
class AdaptiveRepresentationDecoder():
    """A Decoder that takes a genome with two segments, interprets the first segment as
    another Decoder, and uses that to decode the second segment.

    :param executable_decoder: a Decoder that takes segment and produces an executable
        object.
    """
    def __init__(self, executable_decoder):
        assert(executable_decoder is not None)
        self.meta_decoder = executable.WrapperDecoder(
            wrapped_decoder=executable_decoder,
            decorator=function_to_decoder
        )

    def decode(self, genome, *args, **kwargs):
        assert(len(genome) == 2), f"Expected 2 genome segments, got {len(genome)}."
        dec_genome, ind_genome = genome
        decoder = self.meta_decoder.decode(dec_genome)
        return decoder.decode(ind_genome)

##############################
# Class TwoSegmentMutator
##############################
class TwoSegmentMutator():
    """A mutation operator that wraps two mutation functions, applying the first
    to the first segment of the genome and the second to the second."""
    def __init__(self, first_mutator, second_mutator):
        assert(first_mutator is not None)
        assert(second_mutator is not None)
        self.first_mutator = first_mutator
        self.second_mutator = second_mutator

    def __call__(self, next_individual):
        ind = next(next_individual)
        assert(len(ind.genome) == 2), f"Expected 2 genome segments, got {len(ind.genome)}."
        dec_genome, ind_genome = ind.genome
        dec_genome = self.first_mutator(dec_genome)
        ind_genome = self.second_mutator(ind_genome)
        
        ind.genome = [ dec_genome, ind_genome ]
        ind.fitness = None  # Reset fitness since we've borked with the genomes
        yield ind


##############################
# Entry point
##############################
if __name__ == '__main__':
    plot_probe = PopulationPlotProbe(ylim=(0, 70))

    l=10
    pop_size=10
    ea = generational_ea(generations=100, pop_size=pop_size,
                        problem=SpheroidProblem(maximize=False),
                        
                        representation=Representation(
                            individual_cls=Individual,
                            decoder=AdaptiveRepresentationDecoder(
                                executable_decoder=neural_network.SimpleNeuralNetworkDecoder(
                                    
                                )
                            ),
                            initialize=create_real_vector(bounds=[[-5.12, 5.12]] * l)
                        ),

                        pipeline=[
                            ops.tournament_selection,
                            ops.clone,
                            mutate_gaussian(std=0.5),
                            ops.evaluate,
                            ops.pool(size=pop_size),
                            plot_probe  # Insert the probe into the pipeline like so
                        ])
    list(ea);