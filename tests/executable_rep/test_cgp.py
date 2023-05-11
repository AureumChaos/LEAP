from collections import Counter
import itertools
from math import floor, ceil

import numpy as np
import networkx as nx
import pytest

from leap_ec.individual import Individual
from leap_ec.executable_rep import cgp
import leap_ec.statistical_helpers as stat


##############################
# Tests for CGPDecoder
##############################
def test_num_genes1():
    """A linear genome with just one input, node, and output, and arity of 1 should 
    have 3 genes."""
    decoder = cgp.CGPDecoder(primitives=[lambda x: not x], num_inputs=1, num_outputs=1, num_layers=1, nodes_per_layer=1, max_arity=1)
    assert(3 == decoder.num_genes())


def test_decode1():
    """A linear genome with just one input, node, and output, should yield a
    graph connected all three."""
    genome = np.array([0, 0, 1])
    decoder = cgp.CGPDecoder(primitives=[lambda x: not x], num_inputs=1, num_outputs=1, num_layers=1, nodes_per_layer=1, max_arity=1)
    phenome = decoder.decode(genome)

    assert(3 == phenome.graph.number_of_nodes())
    assert(2 == phenome.graph.number_of_edges())
    assert(phenome.graph.has_edge(0, 1))
    assert(phenome.graph.has_edge(1, 2))


@pytest.fixture
def test_2layer_circuit():
    """A simple unpruned CGP circuit that computes AND and is made up of four NAND gates."""
    nand = lambda x, y: not (x and y)
    primitives = [ nand ]
    genome = np.array([ 0, 0, 1,  # Node 2
                        0, 1, 0,  # Node 3
                        0, 2, 3,  # Node 4
                        0, 3, 2,  # Node 5
                        5 ])  # Output is node 5

    decoder = cgp.CGPDecoder(primitives=primitives,
                             num_inputs=2,
                             num_outputs=1,
                             num_layers=2,
                             nodes_per_layer=2,
                             max_arity=2,
                             prune=False)
    
    return genome, decoder.decode(genome), decoder


def test_decode2(test_2layer_circuit):
    """When primitives have arity > 1, the edges of the decoded graph should have an `order` 
    attribute that correctly indicates which input they feed to on the destination node."""
    _, phenome, _ = test_2layer_circuit
    assert(phenome.num_inputs == 2)
    assert(phenome.num_outputs == 1)

    graph = phenome.graph

    assert(7 == graph.number_of_nodes())
    assert(9 == graph.number_of_edges())
    assert(graph.has_edge(0, 2))
    assert(graph.has_edge(1, 2))
    assert(graph.has_edge(0, 3))
    assert(graph.has_edge(1, 3))
    assert(graph.has_edge(2, 4))
    assert(graph.has_edge(3, 4))
    assert(graph.has_edge(2, 5))
    assert(graph.has_edge(3, 5))

    # Each internal node should have arity of 2
    for i in [2, 3, 4, 5]:
        assert(2 == len(list(graph.in_edges(i))))

    # The output node takes only one input
    assert(1 == len(list(graph.in_edges(6))))

    # Check that the edges are feeding into the correct ports
    assert(graph.edges[0, 2, 0]['order'] == 0)  # Input 0 feeds into the 0th port of node 2
    assert(graph.edges[1, 2, 0]['order'] == 1)  # Input 1 feeds into the 1st port of node 2
    assert(graph.edges[0, 3, 0]['order'] == 1)  # Input 0 feeds into the 1st port of node 3
    assert(graph.edges[1, 3, 0]['order'] == 0)  # Input 1 feeds into the 0th port of node 3


def test_decode3():
    '''When we prune unused nodes in a sample graph (which is the default behavior), we
    should see one of the nodes disappear after decoding and the remaining nodes be
    relabeled so that node labels still form contiguous integers.'''
    nand = lambda x, y: not (x and y)
    primitives = [ nand ]
    genome = np.array([ 0, 0, 1,  # Node 2
                        0, 1, 0,  # Node 3
                        0, 2, 3,  # Node 4
                        0, 3, 2,  # Node 5
                        5 ])  # Output (node 6) takes value from node 5

    decoder = cgp.CGPDecoder(primitives=primitives,
                             num_inputs=2,
                             num_outputs=1,
                             num_layers=2,
                             nodes_per_layer=2,
                             max_arity=2)

    phenome = decoder.decode(genome)

    assert(phenome.num_inputs == 2)
    assert(phenome.num_outputs == 1)

    graph = phenome.graph

    # There are 6 nodes now instead of 7: 1 has been pruned
    assert(6 == graph.number_of_nodes())
    # There are 7 edges instead of 9: 2 have been pruned
    assert(7 == graph.number_of_edges())
    assert(graph.has_edge(0, 2))
    assert(graph.has_edge(1, 2))
    assert(graph.has_edge(0, 3))
    assert(graph.has_edge(1, 3))
    # So, node 4 has been pruned.  But then the nodes are relabeled,
    #   so node 5 becomes node 4 and keeps its edges.
    assert(graph.has_edge(2, 4))
    assert(graph.has_edge(3, 4))
    # Because node 5 gets relabeled as 4, there should be
    #   no edges pointing to it.
    assert(not graph.has_edge(2, 5))
    assert(not graph.has_edge(3, 5))
    # The output now takes its value from node 4, since 5 was relabeled to 4
    assert(graph.has_edge(4, 5))

    # Each internal node should have arity of 2
    for i in [2, 3, 4]:
        assert(2 == len(list(graph.in_edges(i))))

    # The output node takes only one input
    assert(1 == len(list(graph.in_edges(5))))

    # Check that the edges are feeding into the correct ports
    assert(graph.edges[0, 2, 0]['order'] == 0)  # Input 0 feeds into the 0th port of node 2
    assert(graph.edges[1, 2, 0]['order'] == 1)  # Input 1 feeds into the 1st port of node 2
    assert(graph.edges[0, 3, 0]['order'] == 1)  # Input 0 feeds into the 1st port of node 3
    assert(graph.edges[1, 3, 0]['order'] == 0)  # Input 1 feeds into the 0th port of node 3


##############################
# Tests for CGPExecutable
##############################
@pytest.fixture
def tt_inputs():
    """Returns the 4 permutations of Boolean inputs for a 2-input truth table."""
    return [ [ True, True ],
             [ True, False ],
             [ False, True ],
             [ False, False ] ]


def test_call1(test_2layer_circuit, tt_inputs):
    """The test individuals should compute the AND function."""
    _, phenome, _ = test_2layer_circuit
    assert(phenome.num_inputs == 2)
    assert(phenome.num_outputs == 1)

    # Truth table for AND
    expected = [ [True], [False], [False], [False] ]

    result = [ phenome(in_vals) for in_vals in tt_inputs ]

    assert(expected == result)


def test_call2(tt_inputs):
    """A circuit with an element that takes two inputs from the same input
    source should execute with no issues (this checks to make sure we support
    multiple edges between the same two nodes).
    """
    genome = [0, 1, 1, 1, 0, 0, 1]

    cgp_decoder = cgp.CGPDecoder(
                        primitives=[
                            lambda x, y: not (x and y),  # NAND
                            lambda x, y: not x,  # NOT (ignoring y)
                        ],
                        num_inputs = 2,
                        num_outputs = 1,
                        num_layers=1,
                        nodes_per_layer=2,
                        max_arity=2
                    )

    phenome = cgp_decoder.decode(genome)

    result = [ phenome(in_vals) for in_vals in tt_inputs ]


def test_call3(tt_inputs):
    """
    A circuit decoded by CGPWithParametersDecoder with 4 nodes that each take a tunable parameter and apply
    arithemtic operations should compute the function we expect it to compute.
    """
    genome = [  # Graph connections
                [ 0, 0, 1,
                 1, 0, 1,
                 2, 2, 3,
                 0, 2, 3,
                 4, 5 ], 
                 # Parameters
               [ 0.5, 15, 2.7, 0.0 ]
            ]

    cgp_decoder = cgp.CGPWithParametersDecoder(
                        primitives=[
                            lambda x, y, z: sum([x, y, z]),
                            lambda x, y, z: (x - y)*z,
                            lambda x, y, z: (x*y)*z
                        ],
                        num_inputs = 2,
                        num_outputs = 2,
                        num_layers=1,
                        nodes_per_layer=4,
                        max_arity=2,
                        num_parameters_per_node=1
                    )


    def expected_function(x, y):
        """This is the function we expect the circuit defined by the above genome to compute."""
        i0, i1 = x, y
        n2 = sum([i0, i1, 0.5])
        n3 = (i0 - i1)*15
        n4 = (n2*n3)*2.7
        n5 = sum([n2, n3, 0.0])
        o6 = n4
        o7 = n5
        return o6, o7

    phenome = cgp_decoder.decode(genome)

    inputs = [
        [1, 1],
        [0, 10],
        [36.5, 0],
        [15, 16]
    ]

    for input in inputs:
        assert(expected_function(*input) == pytest.approx(phenome(input)))



##############################
# Tests for cgp_mutate
##############################
@pytest.mark.stochastic
def test_cgp_mutate1(test_2layer_circuit):
    genome, _, decoder = test_2layer_circuit

    N = 5000
    mutator = cgp.cgp_mutate(decoder, expected_num_mutations=1)
    # Copying the parent N times, since mutation is destructive
    parents = ( Individual(genome.copy()) for _ in range(N) )
    offspring = list(mutator(parents))
    print(offspring)

    observed = {}
    observed[0] = Counter([ ind.genome[0] for ind in offspring ])
    observed[1] = Counter([ ind.genome[1] for ind in offspring ])
    observed[2] = Counter([ ind.genome[2] for ind in offspring ])
    observed[3] = Counter([ ind.genome[3] for ind in offspring ])
    observed[4] = Counter([ ind.genome[4] for ind in offspring ])
    observed[5] = Counter([ ind.genome[5] for ind in offspring ])
    observed[6] = Counter([ ind.genome[6] for ind in offspring ])


    expected = {}
    # Genes 0, 3, 6, and 9 specify primitives.  Since we only have one 
    # primitive, this gene will not change.
    expected[0] = { 0: N }
    expected[3] = { 0: N }
    expected[6] = { 0: N }
    expected[9] = { 0: N }

    # We expect the mutation chance to be 1/L
    p_mut = 1/len(genome)
    p_stay = 1 - p_mut

    # Genes 1 and 2 may be mutated to one of the input nodes,
    # with probability 1/L and uniform sampling
    expected[1] = { 0: floor((p_stay + p_mut*0.5)*N), 1: ceil(p_mut*0.5*N) }
    expected[2] = { 0: floor(p_mut*0.5*N), 1: ceil((p_stay + p_mut*0.5)*N) }
    expected[4] = { 0: floor(p_mut*0.5*N), 1: ceil((p_stay + p_mut*0.5)*N) }
    expected[5] = { 0: floor((p_stay + p_mut*0.5)*N), 1: ceil(p_mut*0.5*N) }

    p = 0.001
    for i in range(7):
        print(f"Gene {i}, expected={expected[i]}, observed={observed[i]}")
        assert(stat.stochastic_equals(expected[i], observed[i], p=p))
