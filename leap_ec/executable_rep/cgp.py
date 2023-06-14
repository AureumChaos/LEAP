"""Cartesian genetic programming (CGP) representation.

The CGPDecoder does most of the work here: it converts a linear genome
into a graph structure, and wraps the latter in a CGPExecutable (which 
knows how to execute the graph)."""

from abc import ABC, abstractmethod
from typing import Iterator, List

import networkx as nx
import numpy as np

from leap_ec import ops, context
from leap_ec.decoder import Decoder
from leap_ec.int_rep.initializers import create_int_vector
from leap_ec.int_rep.ops import mutate_randint, individual_mutate_randint
from .executable import Executable


##############################
# Primitives
##############################
class Primitive(ABC):
    """Abstract class that primitive functions inherit from for CGP.

    You don't need to use this class to define primitive for CGP.
    But if you do, it allows CGP to know the arity of each function—
    which CGPDecoder can use to prune un-needed edges in the resulting
    graph. This sometimes leads better performance or simpler graphs.
    """
    @property
    @abstractmethod
    def arity(self) -> int:
        """How many args are used inside the __call__ function"""
        return 0

    @abstractmethod
    def __call__(self, *args):
        pass

class FunctionPrimitive(Primitive):
    """A convenience wrapper that defines a generic primitive function
    for CGP from a function (ex. a lambda).  Basically this lets us
    define a function that we can also query the arity of.

    >>> f = FunctionPrimitive(lambda x, y: x ^ y, 2)
    >>> f(True, False)
    True
    >>> f.arity
    2
    """
    def __init__(self, func, f_arity: int):
        assert(func is not None)
        assert(f_arity >= 0)
        self.func = func
        self.f_arity = f_arity

    @property
    def arity(self):
        return self.f_arity

    def __call__(self, *args):
        return self.func(*args)

class NAND(Primitive):
    """Primitive NAND function for use in genetic programming.

    >>> f = NAND()
    >>> f(True, True)
    False
    >>> f(True, False)
    True
    """

    @property
    def arity(self):
        return 2

    def __call__(self, *args):
        return not (args[0] and args[1])


class NotX(Primitive):
    """Primitive NOT function for use in genetic programming.

    >>> f = NotX()
    >>> f(True)
    False
    >>> f(False)
    True
    """
    @property
    def arity(self):
        return 1

    def __call__(self, *args):
        return not args[0]


##############################
# Class CGPExecutable
##############################
class CGPExecutable(Executable):
    """Represents a decoded CGP circuit, which can be executed on inputs."""

    def __init__(self, primitives, num_inputs, num_outputs, graph):
        assert(primitives is not None)
        assert(len(primitives) > 0)
        assert(num_inputs > 0)
        assert(num_outputs > 0)
        assert(graph is not None)
        self.primitives = primitives
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.graph = graph

    def __call__(self, input_):
        assert(len(input_) == self.num_inputs)

        def get_sorted_input_nodes(node_id):
            """Pull all the nodes that feed into a given node, ordered by the
            port they feed into on the target node."""
            in_edges = list(self.graph.in_edges(node_id, data='order'))
            # Sort inputs by which "port" they are supposed to feed into
            in_edges = sorted(in_edges, key=lambda e: e[2])  # Element 2 is the 'order' attribute
            in_nodes = [ self.graph.nodes[e[0]] for e in in_edges ]
            return in_nodes


        # Assign the input values
        for i in range(self.num_inputs):
            self.graph.nodes[i]['output'] = input_[i]

        # Compute the values of hidden nodes, going in order so preprequisites are computed first
        num_hidden = len(self.graph.nodes) - self.num_inputs - self.num_outputs
        for i in range(self.num_inputs, self.num_inputs + num_hidden):

            # Collect all of the inputs into this node from other nodes
            in_nodes = get_sorted_input_nodes(i)
            node_inputs = [ n['output'] for n in in_nodes ]

            # Collect any additional constant parameters
            if 'parameters' in self.graph.nodes[i]:
                params = [ p for p in self.graph.nodes[i]['parameters'] ]
                node_inputs.extend(params)

            # Execute the node's function
            f = self.graph.nodes[i]['function']
            self.graph.nodes[i]['output'] = f(*node_inputs)

        # Copy the output values into their nodes
        result = []
        for i in range(self.num_inputs + num_hidden, self.num_inputs + num_hidden + self.num_outputs):
            in_edges = list(self.graph.in_edges(i))
            assert(len(in_edges) == 1), f"CGP output node {i} is connected to {len(in_edges)} nodes, but must be connected to exactly 1."
            in_node = self.graph.nodes[in_edges[0][0]]
            oi = in_node['output']
            self.graph.nodes[i]['output'] = oi
            result.append(oi)

        return result


##############################
# Class CGPDecoder
##############################
class CGPDecoder(Decoder):
    """Implements the genotype-phenotype decoding for Cartesian genetic programming (CGP).

    A CGP genome is linear, but made up of one sub-sequence for each circuit element.  In our
    version here, the first gene in each sub-sequence indicates the primitive (i.e., function) that node computes,
    and the subsequence genes indicate the inputs to that primitive.

    That is, each node is specified by three genes `[p_id, i_1, i_2]`, where `p_id` is the index of the node's
    primitive, and `i_1, i_2` are the indices of the nodes that feed into it.

    The sequence `[ 0, 2, 3 ]` indicates an element that computes the 0th primitive
    (as an index of the `primitives` list) and takes its inputs from nodes 2 and 3, respectively.
    """

    def __init__(self, primitives, num_inputs, num_outputs, num_layers, nodes_per_layer, max_arity, prune: bool=True, levels_back=None):
        assert(primitives is not None)
        assert(len(primitives) > 0)
        assert(num_inputs > 0)
        assert(num_outputs > 0)
        assert(num_layers > 0)
        assert(nodes_per_layer > 0)
        assert(max_arity > 0)
        self.primitives = primitives
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.nodes_per_layer = nodes_per_layer
        self.max_arity = max_arity
        self.prune = prune
        self.levels_back = levels_back if levels_back is not None else num_layers

    def initializer(self):
        """Convenience method that returns an initialization function for creating
        integer-vector genomes that obey this CGP representation's constraints."""

        def create():
            return create_int_vector(self.bounds())()

        return create

    def num_genes(self):
        """The number of genes we expect to find in each genome.  This will equal the number of outputs plus the total number
        of genes needed to specify the nodes of the graph.

        The number of inputs has no effect on the size of the genome.

        For example, a 2x2 CGP individual with 2 outputs an a `max_arity` of 2 will have 14 genes: $3*4 = 12$ genes to
        specify the primitive and inputs (1 + 2) for each internal node, plus 2 genes to specify the circuit outputs.

        >>> decoder = CGPDecoder([sum], num_inputs=2, num_outputs=2, num_layers=2, nodes_per_layer=2, max_arity=2, levels_back=1)
        >>> decoder.num_genes()
        14
        """
        return self.num_layers*self.nodes_per_layer*(self.max_arity + 1) + self.num_outputs

    def num_cgp_nodes(self):
        """Return the total number of nodes that will be in the resulting CGP graph, including inputs and outputs.

        For example, a 2x2 CGP individual with 2 outputs and 2 inputs will have $4 + 2 + 2 = 8$ total graph nodes.

        >>> decoder = CGPDecoder([sum], num_inputs=2, num_outputs=2, num_layers=2, nodes_per_layer=2, max_arity=2, levels_back=1)
        >>> decoder.num_cgp_nodes()
        8
        """
        return self.num_inputs + self.num_layers*self.nodes_per_layer + self.num_outputs

    def get_primitive(self, genome, layer, node):
        """Given a linear CGP genome, return the primitive object for the given node in the
        given layer."""
        assert(genome is not None)
        assert(layer >= 0)
        assert(layer < self.num_layers)
        assert(node >= 0)
        assert(node < self.nodes_per_layer)
        primitive_id = genome[(layer*self.nodes_per_layer + node)*(self.max_arity + 1)]
        assert(primitive_id < len(self.primitives)), f"The gene for node {node} of layer {layer} specifies a primitive function id {primitive_id}, but that's out of range: we only have {len(self.primitives)} primitive(s)!"
        return self.primitives[primitive_id]


    def get_input_sources(self, genome, layer, node):
        """Given a linear CGP genome, return the list of all of the input sources (as integers)
        which feed into the given node in the given layer."""
        assert(genome is not None)
        assert(layer >= 0)
        assert(layer < self.num_layers)
        assert(node >= 0)
        assert(node < self.nodes_per_layer)


        def ith_input_gene(genome, layer, node, i):
            """Helper function that tells us which gene defines the ith input of
            the element at a given layer and node in a Cartesian circuit."""
            return (layer*self.nodes_per_layer + node)*(self.max_arity + 1) + 1 + i

        input_sources = []
        for i in range(self.max_arity):
            gene_id = ith_input_gene(genome, layer, node, i)
            input_sources.append(genome[gene_id])

        assert(len(input_sources) == self.max_arity)
        return input_sources

    def get_output_sources(self, genome):
        """Given a linear CGP genome, return the list of nodes that connect to each output."""
        first_output = self.num_layers*self.nodes_per_layer*(self.max_arity + 1)
        output_sources = genome[first_output:]
        return output_sources

    def _min_bounds(self):
        """Return the minimum allowed value they every gene may assume, taking into account the levels_back parameter, etc.

        These values should be used by initialization and mutation operators to ensure that CGP's constraints are met.

        For example, in a 2x2 CGP grid with two inputs and `levels_back=1`, nodes in the first layer can take inputs
        from nodes 0 or greater (i.e., any of the input nodes), while nodes in the second layer take inputs from node
        2 or greater (i.e. any of the nodes in layer 1).

        This is expressed in the following min-bounds for each gene (recall that each node takes three genes
        `[p_id, i_1, i_2]`, where `p_id` is the index of the node's primitive, and `i_1, i_2` are the indices of the
        nodes that feed into it).

        >>> decoder = CGPDecoder([sum], num_inputs=2, num_outputs=2, num_layers=2, nodes_per_layer=2, max_arity=2, levels_back=1)
        >>> decoder._min_bounds()
        [0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 2, 2, 0, 0]
        """
        mins = []

        # Examine the nodes layer-by-layer
        for l in range(self.num_layers):
            # Determine the minimum node ID that nodes in this layer are allowed to connect to
            if (l == 0) or (l < self.levels_back):  # If we're within levels_back layers from the beginning, we can connect to any prior node
                min_source_value_for_layer = 0
            else:  # Otherwise we can only connect to nodes that are within levels_back layers from this later
                min_source_value_for_layer = self.num_inputs + (l - self.levels_back)*self.nodes_per_layer

            # Assign min bounds for each node in this layer
            for _ in range(self.nodes_per_layer):
                mins.append(0)  # The gene that defines the node's primitive
                for _ in range(self.max_arity):  # The node's input sources
                    mins.append(min_source_value_for_layer)

        # Outputs can connect to any node in the entire circuit
        for _ in range(self.num_outputs):
            mins.append(0)
        return mins

    def _max_bounds(self):
        """Return the maximum allowed value they every gene may assume, taking into account the levels structure.

        These values should be used by initialization and mutation operators to ensure that CGP's constraints are met.

        For example, in a 2x2 CGP grid with two inputs, nodes in the first layer can take inputs
        from up to node 1 (i.e., any of the input nodes), while nodes in the second layer take inputs from up to node
        3 (i.e. any of inputs or nodes in layer 1).

        This is expressed in the following max-bounds for each gene (recall that each node takes three genes
        `[p_id, i_1, i_2]`, where `p_id` is the index of the node's primitive, and `i_1, i_2` are the indices of the
        nodes that feed into it).

        >>> primitives = [ sum, lambda x: x[0] - x[1], lambda x: x[0] * x[1] ]
        >>> decoder = CGPDecoder(primitives, num_inputs=2, num_outputs=2, num_layers=2, nodes_per_layer=2, max_arity=2, levels_back=1)
        >>> decoder._max_bounds()
        [2, 1, 1, 2, 1, 1, 2, 3, 3, 2, 3, 3, 5, 5]
        """
        maxes = []

        # Examine the nodes layer-by-layer
        for l in range(self.num_layers):
            # We can accept inputs from any node in an earlier layer than this one
            max_source_value_for_layer = self.num_inputs - 1 + l*self.nodes_per_layer

            # Assign max bounds for each node in this layer
            for _ in range(self.nodes_per_layer):
                maxes.append(len(self.primitives) - 1)  # The gene that defines the node's primitive
                for _ in range(self.max_arity):  # The node's input sources
                    maxes.append(max_source_value_for_layer)

        # Outputs can connect to any node in the entire circuit, except for other outputs
        for _ in range(self.num_outputs):
            maxes.append(self.num_cgp_nodes() - 1 - self.num_outputs)
        return maxes

    def bounds(self):
        """
        Return the (min, max) allowed value they every gene may assume, taking into account the levels structure.

        These values should be used by initialization and mutation operators to ensure that CGP's constraints are met.

        >>> primitives = [ sum, lambda x: x[0] - x[1], lambda x: x[0] * x[1] ]
        >>> decoder = CGPDecoder(primitives, num_inputs=2, num_outputs=2, num_layers=2, nodes_per_layer=2, max_arity=2, levels_back=1)
        >>> decoder.bounds()
        [(0, 2), (0, 1), (0, 1), (0, 2), (0, 1), (0, 1), (0, 2), (2, 3), (2, 3), (0, 2), (2, 3), (2, 3), (0, 5), (0, 5)]
        """
        return list(zip(self._min_bounds(), self._max_bounds()))

    def check_constraints(self, next_individual: Iterator):
        """An operator that checks whether individual's genomes satisfy the CGP constraints.

        For example, say we have the following population:

        >>> from leap_ec import Individual
        >>> genome0 = np.array([ 0, 0, 1, 1, 0, 1, 2, 2, 3, 0, 2, 3, 4, 5 ])
        >>> genome1 = np.array([ 0, 0, 1, 1, 0, 1, 2, 2, 3, 0, 2, 1, 4, 5 ])
        >>> genome2 = np.array([ 0, 0, 1, 4, 0, 1, 2, 2, 3, 0, 2, 3, 4, 5 ])
        >>> genome3 = np.array([ 0, 0, 1, 1, 0, 1, 2, 2, 3, 0, 2, 3, 4, 5, 3, 4, 5 ])
        >>> genome4 = np.array([ 0.0, 0.0, 1.0, 1.0, 0, 1.0, 2.0, 2.0, 3.0, 0.0, 2.0, 3.0, 4.0, 5.0 ])
        >>> population = iter([ Individual(genome0),
        ...                     Individual(genome1),
        ...                     Individual(genome2),
        ...                     Individual(genome3),
        ...                     Individual(genome4) ])

        Then given this decoder:

        >>> primitives = [ sum, lambda x: x[0] - x[1], lambda x: x[0] * x[1] ]
        >>> decoder = CGPDecoder(primitives, num_inputs=2, num_outputs=2, num_layers=2, nodes_per_layer=2, max_arity=2, levels_back=1)

        The first individual (genome0) satisfies the constraints:

        >>> op = decoder.check_constraints
        >>> next(op(population))
        Individual<...>(...)

        The next fails (genome1), however, because it violates the levels_back constraint:

        >>> next(op(population))
        Traceback (most recent call last):
        ...
        ValueError: CGP constraints violated by individual: expected gene at locus 11 to be between the values of (2, 3) (inclusive), but found a value of 1.

        Then genome2 fails because it contains a cycle:

        >>> next(op(population))
        Traceback (most recent call last):
        ...
        ValueError: CGP constraints violated by individual: expected gene at locus 3 to be between the values of (0, 2) (inclusive), but found a value of 4.

        The new (genome3) fails because it has the incorrect genome length:

        >>> next(op(population))
        Traceback (most recent call last):
        ...
        ValueError: CGP constraints violated by individual: genome of length 17 found, but expected 14 genes.

        And the last (genome4) fails because the genes are of the wrong type:

        >>> next(op(population))
        Traceback (most recent call last):
        ...
        ValueError: CGP constraints violated by individual: genome must contain only integers, but the gene at locus 0 has a non-integral value of 0.0.

        """
        while True:
            ind = next(next_individual)
            bounds = self.bounds()

            if len(ind.genome) != len(bounds):
                raise ValueError(f"CGP constraints violated by individual: genome of length {len(ind.genome)} found, but expected {len(bounds)} genes.")

            for i, (g, (_min, _max)) in enumerate(zip(ind, bounds)):
                if not isinstance(g, int) and not isinstance(g, np.integer):
                    raise ValueError(f"CGP constraints violated by individual: genome must contain only integers, but the gene at locus {i} has a non-integral value of {g}.")

                if g < _min or g > _max:
                    raise ValueError(f"CGP constraints violated by individual: expected gene at locus {i} to be between the values of ({_min}, {_max}) (inclusive), but found a value of {g}.")

            yield ind

    def decode(self, genome, *args, **kwargs):
        """Decode a linear CGP genome into an executable circuit.

        >>> primitives = [ sum, lambda x: x[0] - x[1], lambda x: x[0] * x[1] ]
        >>> decoder = CGPDecoder(primitives, num_inputs=2, num_outputs=2, num_layers=2, nodes_per_layer=2, max_arity=2)
        >>> genome = [ 0, 0, 1, 1, 0, 1, 2, 2, 3, 0, 2, 3, 4, 5 ]
        >>> decoder.decode(genome)
        <leap_ec.executable_rep.cgp.CGPExecutable object at ...>

        """
        assert(genome is not None)
        assert(len(genome) == self.num_genes()), f"Expected a genome of length {self.num_genes()}, but was given one of length {len(genome)}."
        all_node_ids = [i for i in range(self.num_cgp_nodes())]

        graph = nx.MultiDiGraph()
        graph.add_nodes_from(all_node_ids)

        # Add edges connecting interior nodes to their sources
        for layer in range(self.num_layers):
            for node in range(self.nodes_per_layer):
                # TODO Consider using Miller's pre-processing algorithm here to omit nodes that are disconnected from the circuit (making execution more efficient)
                node_id = self.num_inputs + layer*self.nodes_per_layer + node
                function = self.get_primitive(genome, layer, node)
                graph.nodes[node_id]['function'] = function
                inputs = self.get_input_sources(genome, layer, node)

                # If we know the arity of the function, we don't need to connect all the input nodes to the graph.
                if hasattr(function, 'arity'):
                    inputs = inputs[:function.arity]

                # Mark each edge with an 'order' attribute so we know which port they feed into on the target node
                graph.add_edges_from([(i, node_id, {'order': o}) for o, i in enumerate(inputs)])

        # Add edges connecting outputs to their sources
        output_sources = self.get_output_sources(genome)
        output_nodes = all_node_ids[-self.num_outputs:]
        graph.add_edges_from(zip(output_sources, output_nodes))

        if self.prune:
            graph = self.prune_graph(graph, self.num_inputs, self.num_outputs)

        return CGPExecutable(self.primitives, self.num_inputs, self.num_outputs, graph)

    @staticmethod
    def prune_graph(graph, num_inputs: int, num_outputs: int):
        """Prune parts of the graph that do not feed into any of the output nodes."""

        # Get the IDs of the output nodes
        output_nodes = list(graph.nodes())[-num_outputs:]

        # Omit nodes that are disconnected from the circuit (making execution more efficient).
        # We will do this by inducing a subgraph whose necessary nodes are the input nodes, the output nodes,
        # and the "ancestors" of the output nodes.
        necessary_nodes = set(list(range(num_inputs)) + output_nodes)

        # XXX: This for-loop could be a single call if we augment the graph with a new node
        # that is connected to all of the output nodes. Then we would only call ancestors once
        # on the augmented node. However, this node would have to be removed or ignored later.
        for output_node in output_nodes:
            necessary_nodes.update(nx.ancestors(graph, output_node))

        # Induce a subgraph based on nodes.
        graph = nx.subgraph(graph, list(necessary_nodes))
        # The subgraph likely has fewer nodes than before, so we want to reindex them.
        #   This is required, for instance, because when we execute a graph, we assume
        #   that the nodes are labeled consecutively.
        graph = nx.relabel.convert_node_labels_to_integers(graph, ordering="sorted")

        return graph



##############################
# Class CGPWithParametersDecoder
##############################
class CGPWithParametersDecoder(CGPDecoder):
    """
    A CGP decoder that takes a genome with two segments: an integer vector defining
    the usual CGP genome (functions and connectivity), and an auxiliary vector
    defining additional constant parameters to be fed into each node's function.

    Much like bias weights in a neural network, these parameters allow a slightly
    different computation to be performed at different nodes that use the same
    primitive function.

    """
    def __init__(self, primitives, num_inputs: int, num_outputs: int, num_layers: int, nodes_per_layer: int, max_arity: int, num_parameters_per_node: int, prune: bool=True, levels_back=None):
        assert(primitives is not None)
        assert(len(primitives) > 0)
        assert(num_inputs > 0)
        assert(num_outputs > 0)
        assert(num_layers > 0)
        assert(nodes_per_layer > 0)
        assert(max_arity > 0)

        # Tell the superclass *not* to prune the graph of disconnected nodes & edges
        #    because we will want to add the parameters to the graph first ourselves
        #    before we prune.
        prune_super = False
        super().__init__(primitives, num_inputs, num_outputs, num_layers, nodes_per_layer, max_arity, prune_super, levels_back)
        self.prune_later = prune
        self.num_parameters_per_node = num_parameters_per_node

    def decode(self, genome, *args, **kwargs):
        """
        Decode a genome containing both a CGP graph and a list of auxiliary parameters.

        >>> primitives=[
        ...                lambda x, y, z: sum([x, y, z]),
        ...                lambda x, y, z: (x - y)*z,
        ...                lambda x, y, z: (x*y)*z
        ...            ]
        >>> decoder = CGPWithParametersDecoder(primitives, num_inputs=2, num_outputs=2, num_layers=2, nodes_per_layer=2, max_arity=2, num_parameters_per_node=1)
        >>> genome = [ [ 0, 0, 1, 1, 0, 1, 2, 2, 3, 0, 2, 3, 4, 5 ], [ 0.5, 15, 2.7, 0.0 ] ]
        >>> executable = decoder.decode(genome)
        >>> executable
        <leap_ec.executable_rep.cgp.CGPExecutable object at ...>

        Now node #2 (i.e. the first computational node, skipping the two inputs #0 and #1) should have a parameter value of
        0.5, and so on:

        >>> executable.graph.nodes[2]['parameters']
        [0.5]
        >>> executable.graph.nodes[3]['parameters']
        [15]
        >>> executable.graph.nodes[4]['parameters']
        [2.7]
        >>> executable.graph.nodes[5]['parameters']
        [0.0]
        """
        assert(len(genome) == 2), f"Genome should be made up of two segments, but found {len(genome)} top-level elements."
        circuit_genome, parameters_genome = genome

        # Construct the standard CGP circuit
        executable = super().decode(circuit_genome)

        # Add the parameter attributes to each node of the phenotype
        assert(len(parameters_genome) == self.num_parameters_per_node*self.nodes_per_layer*self.num_layers), f"Expected the parameter segment of the genome to contain {self.num_parameters_per_node*self.cgp_decoder.nodes_per_layer*self.cgp_decoder.num_layers} parameters ({self.num_parameters_per_node} for each computational node), but found {len(parameters_genome)}."
        for layer in range(self.num_layers):
            for node in range(self.nodes_per_layer):
                computational_node_id = layer*self.nodes_per_layer + node
                params = parameters_genome[computational_node_id*self.num_parameters_per_node:computational_node_id*self.num_parameters_per_node + self.num_parameters_per_node]
                executable.graph.nodes[self.num_inputs + computational_node_id]['parameters'] = params

        if self.prune_later:
            executable.graph = CGPDecoder.prune_graph(executable.graph, self.num_inputs, self.num_outputs)

        return executable

    def initialize(self, parameters_initializer):
        """Return an initializer for creating the two-segment
        genomes that this decoder expects as input.

        The first segment will be initialized with our standard
        CGP initializer.  The second will use the provided
        initializer.
        """
        def create():
            genome = [
                create_cgp_vector(self)(),
                parameters_initializer()
            ]
            return genome

        return create


##############################
# Function cgp_mutate
##############################
def cgp_mutate(cgp_decoder,
               expected_num_mutations: float = None,
               probability: float = None):
    """A special integer-vector mutation operator that respects the constraints on valid genomes
    that are implied by the parameters of the given CGPDecoder.

    :param cgp_decoder: the Decoder, which informs us about the bounds genes should obey
    :param expected_num_mutations: on average how many mutations done (specificy either this or probability, but not both)
    :param probability: the probability of mutating any given gene (specificy either this or expected_num_mutations, but not both)
    """
    assert(cgp_decoder is not None)

    mutator = mutate_randint(bounds=cgp_decoder.bounds(),
                             expected_num_mutations=expected_num_mutations,
                             probability=probability)

    @ops.iteriter_op
    def mutate(next_individual: Iterator):
        return mutator(next_individual)

    return mutate


##############################
# Function cgp_genome_mutate()
##############################
def cgp_genome_mutate(cgp_decoder,
               expected_num_mutations: float = None,
               probability: float = None):
    assert(cgp_decoder is not None)

    def mutate(genome):
        return individual_mutate_randint(genome,
                                bounds=cgp_decoder.bounds(),
                                expected_num_mutations=expected_num_mutations,
                                probability=probability)

    return mutate


##############################
# Function create_cgp_vector
##############################
def create_cgp_vector(cgp_decoder):
    assert(cgp_decoder is not None)

    return create_int_vector(cgp_decoder.bounds())



##############################
# Function cgp_art_primitives()
##############################
def cgp_art_primitives():
    """
    Returns a standard set of primitives that Ashmore and Miller
    originally published in an online report on "Evolutionary Art with Cartesian Genetic Programming" (2004).
    """
    return [
        lambda x, y, p: np.bitwise_or(x.astype(int), y.astype(int)),  # Bitwise OR of two numbers
        lambda x, y, p: np.bitwise_and(p.astype(int), x.astype(int)),  # Bitwise AND of parameter and a number
        lambda x, y, p: x/(1.0 + y + p),
        lambda x, y, p: x * y * 255,
        lambda x, y, p: (x + y) * 255,
        lambda x, y, p: np.maximum(x - y, y - x),
        lambda x, y, p: 255 - x,
        lambda x, y, p: np.abs(np.cos(x) * 255),
        lambda x, y, p: np.abs(np.tan((x % 45) * np.pi/180.0 * 255)),
        lambda x, y, p: np.abs(np.tan(x) * 255) % 255,  # The original paper has a typo here which I interpretted as an erroneous trailing parenthesis
        lambda x, y, p: np.minimum(np.sqrt( (x - p)**2 + (y - p)**2), 255),  # My interpretation of the original papers ambiguous remark that this be "thresholded at 255"
        lambda x, y, p: x % (p + 1) + (255 - p),
        lambda x, y, p: (x + y)/2,
        lambda x, y, p: np.minimum(255 * (y + 1)/(x + 1), 255 * (x + 1)/(y + 1)),
        lambda x, y, p: np.sqrt(np.abs(x**2 - p**2 + y**2 - p**2 )) % 255
    ]
