from collections import Counter
import itertools
from math import floor, ceil

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
    genome = [0, 0, 1]
    decoder = cgp.CGPDecoder(primitives=[lambda x: not x], num_inputs=1, num_outputs=1, num_layers=1, nodes_per_layer=1, max_arity=1)
    phenome = decoder.decode(genome)

    assert(3 == phenome.graph.number_of_nodes())
    assert(2 == phenome.graph.number_of_edges())
    assert(phenome.graph.has_edge(0, 1))
    assert(phenome.graph.has_edge(1, 2))


@pytest.fixture
def test_2layer_circuit():
    """A simple CGP circuit that computes AND and is made up of four NAND gates."""
    nand = lambda x, y: not (x and y)
    primitives = [ nand ]
    genome = [ 0, 0, 1,  # Node 2
               0, 1, 0,  # Node 3
               0, 2, 3,  # Node 4
               0, 3, 2,  # Node 5
               5 ]  # Output is node 5

    decoder = cgp.CGPDecoder(primitives=primitives,
                             num_inputs=2,
                             num_outputs=1,
                             num_layers=2,
                             nodes_per_layer=2,
                             max_arity=2)
    
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
    source should execut with no issues (this checks to make sure we support
    multiple edges between the same two nodes)."""
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



##############################
# Tests for cgp_mutate
##############################
@pytest.mark.stochastic
def test_cgp_mutate1(test_2layer_circuit):
    genome, _, decoder = test_2layer_circuit

    N = 1000
    mutator = cgp.cgp_mutate(decoder)
    parents = ( Individual(genome[:]) for _ in range(N) )  # Copying the parent N times, since mutation is destructive
    offspring = list(mutator(parents))

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

    for i in range(7):
        print(f"Gene {i}, expected={expected[i]}, observed={observed[i]}")
        assert(stat.stochastic_equals(expected[i], observed[i]))
