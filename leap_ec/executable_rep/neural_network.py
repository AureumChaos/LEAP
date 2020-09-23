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
    as a sequence of weight matrices.
    
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
        self.shape = shape
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
    """A simple fixed-architecture neural network that can be executed on inputs."""
    def __init__(self, weight_matrices, activation):
        assert(weight_matrices is not None)
        assert(activation is not None)
        self.weight_matrices = weight_matrices
        self.activation = activation

    def output(self, inputs):
        assert(inputs is not None)
        signal = np.array(inputs)
        
        for W in self.weight_matrices:
            signal = np.append(signal, 1.0) # Add a constant bias unit to the input
            signal = self.activation(np.dot(signal, W))

        if len(signal) > 1:
            return np.round(signal)  # Return a list of outputs if there are more than one
        else:
            return int(np.round(signal[0]))  # Return just the raw output if there is only one
