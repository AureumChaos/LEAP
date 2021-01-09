"""Tools for decoding and executing a neural network from its genetic representation."""
from typing import Tuple

import numpy as np

from .executable import Executable


##############################
# Function sigmoid
##############################
def sigmoid(x):
    """A logistic sigmoid activation function.  Accepts array-like inputs,
    and uses NumPy for efficient computation.
    """
    return 1/(1+np.exp(-x))


##############################
# Function ReLu
##############################
def relu(x):
    """A rectified linear unit (ReLu) activation function.  Accept array-like
    inputs, and uses NumPy for efficient computation."""
    return np.maximum(0, x)


##############################
# Function softmax
##############################
def softmax(x):
    """A softmax activation function.  Accepts array-like input and normalizes
    each element relative to the others."""
    return np.exp(x)/np.sum(np.exp(x))


##############################
# Class SimpleNeuralNetworkDecoder
##############################
class SimpleNeuralNetworkDecoder():
    """Decode a real-vector genome into a neural network by treating it
    as a test_sequence of weight matrices.

    For example, say we have a linear real-valued made up of 29 values:

    >>> genome = list(range(0, 29))

    We can decode this into a neural network with 4 inputs, two hidden layers
    (of size 3 and 2), and 2 outputs like so:

    >>> from leap_ec.executable_rep import neural_network
    >>> dec = neural_network.SimpleNeuralNetworkDecoder([ 4, 3, 2, 2 ])
    >>> nn = dec.decode(genome)

    :param (int) shape: the size of each layer of the network, i.e. (inputs,
        hidden nodes, outputs).  The shape tuple must have at least two
        elements (inputs + bias weight and outputs): each additional value is treated as a hidden layer.
        Note also that we expect a bias weight to exist for the inputs of each layer,
        so the number of weights at each layer will be set to 1 greater
        than the number of inputs you specify for that layer.
    """

    def __init__(self, shape: Tuple[int], activation=sigmoid):
        assert(shape is not None)
        assert(len(shape) > 1)

        shape = [ x for x in shape if x != 0 ]  # Ignore layers of size zero

        # Pair the shapes into the dimensions of each weight matrix,
        # adding one row to each layer's input so they can accomodate
        # a biat unit.
        # ex. [a, b, c, d] —> [(a + 1, b), (b + 1, c), (c + 1, d)]
        shape = np.array(shape)
        self.dimensions = list(zip(1 + shape[:-1], shape[1:]))

        matrix_lengths = list(map(lambda x: x[0]*x[1], self.dimensions))
        self.length = sum(matrix_lengths)
        self.activation = activation

    def decode(self, genome, *args, **kwargs):
        """Decode a genome into a `SimpleNeuralNetworkExecutable`."""
        if len(genome) != self.length:
            raise ValueError(f"Expected a genome of length {self.length}, but received one of {len(genome)}.")

        # Extract each layer's weight matrix from the linear genome
        start = 0
        weight_matrices = []
        for num_inputs, num_outputs in self.dimensions:
            end = start + num_inputs*num_outputs
            layer_sequence = genome[start:end]
            layer_matrix = np.reshape(layer_sequence, (num_inputs, num_outputs))
            weight_matrices.append(layer_matrix)
            start = end

        return SimpleNeuralNetworkExecutable(weight_matrices, self.activation)


##############################
# Class SimpleNeuralNetworkExecutable
##############################
class SimpleNeuralNetworkExecutable(Executable):
    """A simple fixed-architecture neural network that can be executed on inputs.

    Takes a list of weight matrices and an activation function as arguments.  The
    weight matrices each must have 1 row more than the previous layer's outputs,
    to support a bias node that is implicitly connected to each layer.

    For example, here we build a network with 10 inputs, two hidden layers (with
    5 and 3 nodes, respectively), and 5 output nodes, and random weights:

    >>> import numpy as np
    >>> from leap_ec.executable_rep import neural_network
    >>> n_inputs = 10
    >>> n_hidden1, n_hidden2 = 5, 3
    >>> n_outputs = 5
    >>> weights = [ np.random.uniform((n_inputs + 1, n_hidden1)),
    ...             np.random.uniform((n_hidden1 + 1, n_hidden2)),
    ...             np.random.uniform((n_hidden2 + 1, n_outputs)) ]
    >>> nn = neural_network.SimpleNeuralNetworkExecutable(weights, neural_network.sigmoid)

    """
    def __init__(self, weight_matrices, activation):
        assert(weight_matrices is not None)
        assert(activation is not None)
        self.weight_matrices = weight_matrices
        self.activation = activation

    def __call__(self, input_):
        assert(input_ is not None)
        signal = np.array(input_)
        #print(f"\n\nINPUT\n{signal.tolist()}")
        for W in self.weight_matrices:
            signal = np.append(signal, 1.0) # Add a constant bias unit to the input
            #print(f"\n\n\nWEIGHTS\n{W.tolist()}")
            signal = self.activation(np.dot(signal, W))
            assert(len(signal) > 0)
            #print(f"\n\n\nOUTPUT\n{signal.tolist()}")

        return signal
