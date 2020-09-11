import numpy as np

from .phenotype import *

##############################
# Class PittRulesDecoder
##############################
class PittRulesDecoder(core.Decoder):
    """Construct a Pitt-approach rule system (phenotype) out of a real-valued 
    genome.
    """
    def __init__(self, input_space, output_space, priority_metric,
                 num_memory_registers):
        assert (input_space is not None)
        assert (output_space is not None)
        assert (num_memory_registers >= 0)
        self.input_space = input_space
        self.num_inputs = int(np.prod(input_space.shape))
        self.output_space = output_space
        self.num_outputs = int(np.prod(output_space.shape))
        self.priority_metric = priority_metric
        self.num_memory_registers = num_memory_registers

    def decode(self, genome, *args, **kwargs):
        assert (genome is not None)
        assert (len(genome) > 0)
        rule_length = self.num_inputs * 2 + \
                      self.num_outputs + self.num_memory_registers
        assert (len(genome) % rule_length == 0)
        rules = np.reshape(genome, (-1, rule_length))
        return PittRulesExecutable(self.input_space, self.output_space, rules,
                              self.priority_metric)