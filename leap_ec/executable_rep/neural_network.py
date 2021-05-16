"""Tools for decoding and executing a neural network from its genetic representation."""
from typing import Tuple

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np

from .executable import Executable
from leap_ec.global_vars import context


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

    @property
    def num_hidden_layers(self):
        """The number of hidden layers in this network."""
        return len(self.weight_matrices) - 1

    @property
    def num_inputs(self):
        """The number of inputs the network receives."""
        first_matrix = self.weight_matrices[0]
        num_rows = np.shape(first_matrix)[0]
        return num_rows - 1  # Exclude the bias input

    @property
    def num_outputs(self):
        """The number of outputs the network produces."""
        last_matrix = self.weight_matrices[-1]
        num_columns = np.shape(last_matrix)[1]
        return num_columns


    @property
    def graph(self):
        """Create a graph representation of this neural network (ex., for visualization)."""
        graph = nx.MultiDiGraph()
        graph.add_node('bias')

        input_ids = [ f"i{i}" for i in range(self.num_inputs) ]
        graph.add_nodes_from(input_ids)
        previous_ids = input_ids + ['bias']

        for l in range(self.num_hidden_layers):
            matrix = self.weight_matrices[l]
            assert(np.shape(matrix)[0] == len(previous_ids)), f"Expected {len(previous_ids)} inputs to hidden layer {l}, but got {np.shape(matrix)[0]}."
            num_hidden_nodes = np.shape(matrix)[1]  # Num columns
            hidden_ids = [ f"h{l}_{i}" for i in range(num_hidden_nodes)]
            graph.add_nodes_from(hidden_ids)

            for r, row in enumerate(matrix):
                source_id = previous_ids[r]
                edges = [(source_id, hidden_ids[i], w) for i, w in enumerate(row)]
                graph.add_weighted_edges_from(edges)

            previous_ids = hidden_ids + ['bias']


        output_ids = [ f"o{i}" for i in range(self.num_outputs) ]
        graph.add_nodes_from(output_ids)

        matrix = self.weight_matrices[self.num_hidden_layers]  # Matrix for last layer
        assert(np.shape(matrix)[1] == self.num_outputs)
        for r, row in enumerate(matrix):
            source_id = previous_ids[r]
            edges = [(source_id, output_ids[i], w) for i, w in enumerate(row)]
            graph.add_weighted_edges_from(edges)

        return graph


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


##############################
# Class GraphPhenotypeProbe
##############################
class GraphPhenotypeProbe():
    """Visualize the graph for the best individual in the population.
    
    This requires that the phenotypes of the individuals in the population
    have a `graph` attribute that provides a `networkx` graph object.
    """

    def __init__(self, modulo=1, ax=None, weights: bool=False, weight_multiplier: float=1.0, context=context):
        assert(modulo > 0)
        assert(context is not None)
        self.modulo = modulo
        self.weights = weights
        self.weight_multiplier = weight_multiplier
        if ax is None:
            _, ax = plt.subplots()
        self.ax = ax
        self.context = context

    def __call__(self, population: list) -> list:
        """Take a population, plot the best individual (if `step % modulo == 0`),
        and return the population unmodified.
        """
        assert(population is not None)
        assert ('leap' in self.context)
        assert ('generation' in self.context['leap'])
        step = self.context['leap']['generation']

        if step % self.modulo == 0:
            best = max(population)
            graph = best.decode().graph
            self.ax.clear()
            if self.weights:
                weights = list(nx.get_edge_attributes(graph,'weight').values())
                weights = [ self.weight_multiplier*w for w in weights ]
                nx.draw_shell(graph,
                        width=weights,
                        with_labels=True,
                        ax=self.ax)
            else:
                nx.draw_shell(graph,
                        with_labels=True,
                        ax=self.ax)

        return population