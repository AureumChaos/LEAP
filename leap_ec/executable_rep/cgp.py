"""Cartesian genetic programming (CGP) representation."""
from typing import Iterator

import networkx as nx
import numpy as np

from leap_ec import ops
from leap_ec.decoder import Decoder
from leap_ec.int_rep.initializers import create_int_vector
from leap_ec.int_rep.ops import mutate_randint
from .executable import Executable


##############################
# Class CGPExecutable
##############################
class CGPExecutable(Executable):
    """Represented a decoded CGP circuit, which can be executed on inputs."""

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

    def __init__(self, primitives, num_inputs, num_outputs, num_layers, nodes_per_layer, max_arity, levels_back=None):
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
        self.levels_back = levels_back if levels_back is not None else num_layers

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
                graph.nodes[node_id]['function'] = self.get_primitive(genome, layer, node)
                inputs = self.get_input_sources(genome, layer, node)
                # Mark each edge with an 'order' attribute so we know which port they feed into on the target node
                graph.add_edges_from([(i, node_id, {'order': o}) for o, i in enumerate(inputs)])

        # Add edges connecting outputs to their sources
        output_sources = self.get_output_sources(genome)
        output_nodes = all_node_ids[-self.num_outputs:]
        graph.add_edges_from(zip(output_sources, output_nodes))

        return CGPExecutable(self.primitives, self.num_inputs, self.num_outputs, graph)



##############################
# Class CGPWithParametersDecoder
##############################
class CGPWithParametersDecoder():
    """
    A CGP decoder that takes a genome with two segments: an integer vector defining
    the usual CGP genome (functions and connectivity), and an auxiliary vector
    defining additional constant parameters to be fed into each node's function.

    Much like bias weights in a neural network, these paramaters allow a slightly
    different computation to be performed at different nodes that use the same
    primitive function.
    
    """
    def __init__(self, primitives, num_inputs: int, num_outputs: int, num_layers: int, nodes_per_layer: int, max_arity: int, num_parameters_per_node: int, levels_back=None):
        assert(primitives is not None)
        assert(len(primitives) > 0)
        assert(num_inputs > 0)
        assert(num_outputs > 0)
        assert(num_layers > 0)
        assert(nodes_per_layer > 0)
        assert(max_arity > 0)
        self.cgp_decoder = CGPDecoder(primitives, num_inputs, num_outputs, num_layers, nodes_per_layer, max_arity, levels_back)
        self.num_parameters_per_node = num_parameters_per_node

    def decode(self, genome):
        """
        Decode a genome containing both a CGP graph and a list of auxiliary parameters.

        >>> primitives = [ sum, lambda x: x[0] - x[1], lambda x: x[0] * x[1] ]
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
        executable = self.cgp_decoder.decode(circuit_genome)

        # Add the parameter attributes to each node of the phenotype
        assert(len(parameters_genome) == self.num_parameters_per_node*self.cgp_decoder.nodes_per_layer*self.cgp_decoder.num_layers), f"Expected the parameter segment of the genome to contain {self.num_parameters_per_node*self.cgp_decoder.nodes_per_layer*self.cgp_decoder.num_layers} parameters ({self.num_parameters_per_node} for each computational node), but found {len(parameters_genome)}."
        for layer in range(self.cgp_decoder.num_layers):
            for node in range(self.cgp_decoder.nodes_per_layer):
                computational_node_id = layer*self.cgp_decoder.nodes_per_layer + node
                params = parameters_genome[computational_node_id*self.num_parameters_per_node:computational_node_id*self.num_parameters_per_node + self.num_parameters_per_node]
                executable.graph.nodes[self.cgp_decoder.num_inputs + computational_node_id]['parameters'] = params

        return executable


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
# Function create_cgp_vector
##############################
def create_cgp_vector(cgp_decoder):
    assert(cgp_decoder is not None)

    def create():
        return create_int_vector(cgp_decoder.bounds())()

    return create



##############################
# Function cgp_art_primitives()
##############################
def cgp_art_primitives():
    """
    Returns a standard set of primitives that Ashmore and Miller
    originally published in an online report on "Evolutionary Art with Cartesian Genetic Programming" (2004).
    """
    p = 3  # Index of the "parameter" associated with each function
    return [
        lambda x: x[0] | x[1],  # Bitwise OR of two numbers
        lambda x: x[p] & x[0],  # Bitwise AND of parameter and a number
        lambda x: x[0]/(1.0 + x[1] + x[p]),
        lambda x: x[0] * x[1] * 255,
        lambda x: (x[0] + x[1]) * 255,
        lambda x: x[0] - x[1] if x[0] > x[1] else x[1] - x[0],
        lambda x: 255 - x[0],
        lambda x: np.abs(np.cos(x[0]) * 255),
        lambda x: np.abs(np.tan((x[0] % 45) * np.PI/180.0 * 255)),
        lambda x: np.abs(np.tan(x[0]) * 255) % 255,  # The original paper has a typo here which I interpretted as an erroneous trailing parenthesis
        lambda x: min(np.sqrt( (x[0] - x[p])**2 + (x[1] - x[p])**2), 255),  # My interpretation of the original papers ambiguous remark that this be "thresholded at 255"
        lambda x: x[0] % (x[p] + 1) + (255 - x[p]),
        lambda x: (x[0] + x[1])/2,
        lambda x: 255 * (x[1] + 1)/(x[0] + 1) if x[0] > x[1] else 255 * (x[0] + 1)/(x[1] + 1),
        lambda x: np.abs(np.sqrt(x[0]**2 - x[p]**2 + x[1]**2 - x[p]**2 ) % 255 )
    ]