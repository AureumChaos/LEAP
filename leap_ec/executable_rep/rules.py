from enum import Enum
import numpy as np

from leap_ec.decoder import Decoder
from leap_ec.executable_rep.executable import Executable

##############################
# Class PittRulesDecoder
##############################
class PittRulesDecoder(Decoder):
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


##############################
# Class PittRulesExecutable
##############################
class PittRulesExecutable(Executable):
    """
    An `Executable` phenotype that interprets a Pittsburgh-style ruleset and 
    outputs the appropriate action.

    :param input_space: an OpenAI-gym-style space defining the inputs
    :param output_space: an OpenAI-gym-style space defining the outputs
    :param init_memory: a list of initial values for the memory registers
    :param rules: a list of rules, each of the form `[ c1 c1'  c2 c2' ... cn
        cn'  a1 ... am m1 ... mr]`
    :param priority_metric: the rule prioritization strategy used to resolve
        conflicts

    Rulesets are lists of rules.  Rules are lists of the form `[ c1 c1'  c2
    c2' ... cn cn'  a1 ... am m1 ... mr]`, where `(cx, cx')` are are the min
    and max bounds that the rule covers, `a1 .. am` are the output actions,
    and `m1 ... mr` are values to write to the memory registers.

    For example, this ruleset has two rules.  The first rule covers the
    square bounded by `(0.0, 0.6)' and `(0.0, 0.4)`, returning the output
    action `0` if the input falls within that range:

    >>> rules = [[0.0,0.6, 0.0,0.4, 0],
    ...          [0.4,1.0, 0.6,1.0, 1]]

    The input and output spaces are defined in the style of OpenAI gym.  For
    example, here's how you would set up a PittRulesExecutable with the above
    ruleset that takes two continuous input variables on `(0.0, 1.0)`,
    and outputs discrete values in `{0, 1}`:

    >>> import numpy as np
    >>> from gym import spaces
    >>> input_space = spaces.Box(low=np.array((0, 0)), high=np.array((1.0, 1.0)), dtype=np.float32)
    >>> output_space = spaces.Discrete(2)
    >>> rules = PittRulesExecutable(input_space, output_space, rules,
    ...                             priority_metric=PittRulesExecutable.PriorityMetric.RULE_ORDER)
    """

    PriorityMetric = Enum('PriorityMetric', 'RULE_ORDER GENERALITY PERIMETER')

    def __init__(self, input_space, output_space, rules, priority_metric,
                 init_mem=[]):
        assert (input_space is not None)
        assert (output_space is not None)
        assert (rules is not None)
        assert (len(rules) > 0)
        assert (priority_metric in \
                PittRulesExecutable.PriorityMetric.__members__.values())

        self.input_space = input_space
        self.num_inputs = int(np.prod(input_space.shape))
        self.output_space = output_space
        self.num_outputs = int(np.prod(output_space.shape))
        self.memory_registers = np.array(init_mem)
        self.num_memory = len(self.memory_registers)
        self.rules = rules
        self.priorities = [self.__priority(r, rule, priority_metric) for
                           (r, rule) in enumerate(rules)]

    def __priority(self, rule_order, rule, priority_metric):
        """Compute the priority value to a given rule."""
        if priority_metric == PittRulesExecutable.PriorityMetric.RULE_ORDER:
            return rule_order
        elif priority_metric == PittRulesExecutable.PriorityMetric.GENERALITY:
            pass
        elif priority_metric == PittRulesExecutable.PriorityMetric.PERIMETER:
            pass
        else:
            raise ValueError(
                'Unrecognized priority_metric "{0}".'.format(priority_metric))

    def __match_set(self, input):
        """Build the match set for a set of rules."""

        all_input = np.append(input, self.memory_registers)
        best_match_score = -1
        match_list = []  # indices of matchedrules

        for (r, rule) in enumerate(self.rules):
            match_score = 0
            for c in range(len(all_input)):  # Loop through all conditions
                # TODO Normalize this, in case the possible ranges
                # differ greatly
                low, high = (rule[c * 2], rule[c * 2 + 1])
                if all_input[c] >= low and all_input[c] <= high:
                    diff = 0  # Within range
                else:
                    diff = min(abs(low - all_input[c]), abs(high - all_input[c]))
                match_score += diff * diff  # Distance w/o sqrt

            if match_list == [] or match_score < best_match_score:
                best_match_score = match_score
                match_list = [r]
            elif match_score == best_match_score:
                match_list.append(r)
        return match_list, best_match_score

    def __fire(self, rule_index):
        winner = self.rules[rule_index]
        if self.num_memory == 0:
            return winner[-self.num_outputs:]
        else:
            self.memory_registers = winner[-self.num_memory:]
            return winner[-self.num_outputs - self.num_memory:-self.num_memory]

    def output(self, input):
        """

        :param input:
        :return:

        For example, take this set of two rules:

        >>> ruleset = [[0.0,0.6, 0.0,0.5, 0],
        ...            [0.4,1.0, 0.3,1.0, 1]]

        We build an executable around it like so:

        >>> import numpy as np
        >>> from gym import spaces
        >>> input_space = spaces.Box(low=np.array((0, 0)), high=np.array((1.0, 1.0)), dtype=np.float32)
        >>> output_space = spaces.Discrete(2)
        >>> rules = PittRulesExecutable(input_space, output_space, ruleset,
        ...                             priority_metric=PittRulesExecutable.PriorityMetric.RULE_ORDER)

        It outputs `0` for inputs that are covered by only the first rule:

        >>> rules.output([0.1, 0.1])
        0

        >>> rules.output([0.5, 0.3])
        0

        It outputs `1` for inputs that are covered by only the second rule:

        >>> rules.output([0.9, 0.9])
        1

        >>> rules.output([0.5, 0.6])
        1

        If a point is covered by both rules, the first rule fires (because we set `priority_metric` to `RULE_ORDER`),
        and it outputs `0`:

        >>> rules.output([0.5, 0.5])
        0

        Note that if the system has more than one output, a list is returned:

        >>> ruleset = [[0.0,0.6, 0.0,0.5, 0, 1],
        ...            [0.4,1.0, 0.3,1.0, 1, 0]]
        >>> output_space = spaces.MultiBinary(2)  # A space with two binary outputs
        >>> rules = PittRulesExecutable(input_space, output_space, ruleset,
        ...                             priority_metric=PittRulesExecutable.PriorityMetric.RULE_ORDER)
        >>> rules.output([0.1, 0.1])
        [0, 1]

        """
        # Compute the match set
        match_list, best_match_score = self.__match_set(input)

        # If our best-matching rules are exact matches
        if best_match_score == 0:
            # then cull the matchList based on priority.
            highest_priority_rule = min(match_list,
                                        key=lambda x: self.priorities[x])
            highest_priority_value = self.priorities[highest_priority_rule]
            match_list = [r for r in match_list if
                          self.priorities[r] <= highest_priority_value]

        # Among the rules set of rules with the highest priority, choose one
        # randomly.

        # TODO Implement other common conflict-resolutions strategies: like
        #  picking the output with the most rules

        # TODO advocating for it (i.e. vote).
        winner = np.random.choice(match_list)
        output = np.round(self.__fire(winner)).astype(int).tolist()
        assert (len(output) == self.num_outputs)
        if self.num_outputs > 1:
            return output  # Return a list of outputs if there are more than one
        else:
            return output[0]  # Return just the raw output if there is only one 
