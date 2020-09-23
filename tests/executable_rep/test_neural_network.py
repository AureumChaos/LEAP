import pytest
from pytest import approx
import numpy as np

from leap_ec.executable_rep import neural_network


########################
# Tests for SimpleNeuralNetworkDecoder
########################
def test_decode1():
    """If we're expecting a single layer, create a single weight matrix from
    the genome."""
    n_inputs = 10
    n_outputs = 5

    # Adding one more "input" to the weights shape to serve as the bias weight
    dec = neural_network.SimpleNeuralNetworkDecoder((n_inputs, n_outputs))

    genome = list(range(0, n_outputs))*(n_inputs + 1)
    expected = np.array([ [ 0, 1, 2, 3, 4 ] ] * (n_inputs + 1)) # +1 because there is an extra bias weight
    nn = dec.decode(genome)
    matrix = nn.weight_matrices[0]
    assert((expected == matrix).all())


def test_decode2():
    """If we receive a genome that is too long, throw an exception."""
    n_inputs = 10
    n_outputs = 5

    dec = neural_network.SimpleNeuralNetworkDecoder((n_inputs, n_outputs))
    correct_length = (n_inputs + 1)*n_outputs  # +1 because there is an extra bias weight
    genome = np.random.uniform(0, 1, correct_length + 1)

    with pytest.raises(ValueError):
        nn = dec.decode(genome)


def test_decode3():
    """If we receive a genome that is too short, throw an exception."""
    n_inputs = 10
    n_outputs = 5

    # Adding one more "input" to the weights shape to serve as the bias weight
    dec = neural_network.SimpleNeuralNetworkDecoder((n_inputs, n_outputs))
    correct_length = (n_inputs + 1)*n_outputs   # +1 because there is an extra bias weight
    genome = np.random.uniform(0, 1, correct_length - 1)

    with pytest.raises(ValueError):
        nn = dec.decode(genome)


def test_decode4():
    """If we're expecting three layers, create three weight matrices out of 
    the genome."""
    n_inputs = 4
    n_hidden1 = 3
    n_hidden2 = 2
    n_outputs = 2

    shape = [ n_inputs, n_hidden1, n_hidden2, n_outputs ]
    dec = neural_network.SimpleNeuralNetworkDecoder(shape)

    genome = list(range(0, 29))
    expected = [ np.reshape(np.arange(0, 15), (5, 3)),
                 np.reshape(np.arange(15, 23), (4, 2)),
                 np.reshape(np.arange(23, 29), (3, 2)) ]
    
    nn = dec.decode(genome)
    result = nn.weight_matrices
    assert(len(expected) == len(result))
    print(expected)
    print(result)
    for e, r in zip(expected, result):
        assert((e == r).any())


##############################
# Tests for SimpleNeuralNetworkExecutable
##############################
def test_output1():
    """If a single-layer neural network has homogeneous inputs and a
    homogeneous weight matrix, all the outputs should be the same."""
    n_inputs = 10
    n_outputs = 5
    weights = np.ones((n_inputs + 1, n_outputs))  # Adding an extra weight for the bias

    nn = neural_network.SimpleNeuralNetworkExecutable([ weights ], neural_network.sigmoid)

    x = [1]*n_inputs
    result = nn(x)
    assert(len(result) == n_outputs)
    for o in result:
        assert(approx(result[0]) == o)

def test_output2():
    """If a multi-layer neural network has homogeneous inputs and
    homogeneous weight matrices, all the outputs should be the same."""
    n_inputs = 10
    n_hidden1 = 5
    n_hidden2 = 3
    n_outputs = 5
    # Three layers, three matrices
    weights = [ np.ones((n_inputs + 1, n_hidden1)),  # Adding an extra weight for the bias
                3*np.ones((n_hidden1 + 1, n_hidden2)),
                2*np.ones((n_hidden2 + 1, n_outputs)) ]

    nn = neural_network.SimpleNeuralNetworkExecutable(weights, neural_network.sigmoid)

    x = [1]*n_inputs
    result = nn(x)
    assert(len(result) == n_outputs)
    for o in result:
        assert(approx(result[0]) == o)


def test_output3():
    """If we give each sigmoid neuron in a single-layer network a very high
    input, their output will be approximately 1.0."""
    n_inputs = 10
    n_outputs = 5
    # Random weights all > 100, adding an extra weight for the bias
    weights = 100*np.random.uniform(1, 2, (n_inputs + 1, n_outputs))  

    nn = neural_network.SimpleNeuralNetworkExecutable([ weights ], neural_network.sigmoid)

    x = [1]*n_inputs
    result = nn(x)
    assert(len(result) == n_outputs)
    for o in result:
        assert(approx(1.0) == o)


def test_output4():
    """If we give each sigmoid neuron in a single-layer network a very 
    negative input, their output should be approximately zero."""
    n_inputs = 10
    n_outputs = 5
    # Random weights all < -100, adding an extra weight for the bias
    weights = -100*np.random.uniform(1, 2, (n_inputs + 1, n_outputs))  

    nn = neural_network.SimpleNeuralNetworkExecutable([ weights ], neural_network.sigmoid)

    x = [1]*n_inputs
    result = nn(x)
    assert(len(result) == n_outputs)
    for o in result:
        assert(approx(0.0) == o)


def test_output5():
    """If we give each ReLu neuron in a single-layer network a 
    negative input, their output should be zero."""
    n_inputs = 10
    n_outputs = 5
    # Random weights all < 0, adding an extra weight for the bias
    weights = -np.random.uniform(0, 1, (n_inputs + 1, n_outputs)) 

    nn = neural_network.SimpleNeuralNetworkExecutable([ weights ], neural_network.relu)

    x = [1]*n_inputs
    result = nn(x)
    assert(len(result) == n_outputs)
    for o in result:
        assert(approx(0.0) == o)


def test_output6():
    """If we give each ReLu neuron in a single-layer network a 
    positive input, with network inputs of all ones, their
    output should equal the sum of all the weights."""
    n_inputs = 10
    n_outputs = 5
    # Random weights all < 0, adding an extra weight for the bias
    weights = np.random.uniform(0, 1, (n_inputs + 1, n_outputs)) 

    nn = neural_network.SimpleNeuralNetworkExecutable([ weights ], neural_network.relu)

    x = [1]*n_inputs
    result = nn(x)
    assert(len(result) == n_outputs)
    for i, o in enumerate(result):
        expected = np.sum(weights[:,i])
        assert(approx(expected) == o)


##############################
# Tests for Softmax
##############################
def test_softmax1():
    input_ = np.array([8, 5, 0])
    output = neural_network.softmax(input_)
    expected = np.array([0.9522698261237778, 0.04741072293787844, 0.0003194509383437505])

    for e, o in zip(expected, output):
        assert(approx(e) == o)
