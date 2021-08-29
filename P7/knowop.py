# Name:         Michael Long
# Course:       CSC 480
# Instructor:   Daniel Kauffman
# Assignment:   Know Op
# Term:         Summer 2021

import math
import itertools
import random
from typing import Callable, Dict, List, Tuple


class Math:
    """A collection of static methods for mathematical operations."""

    @staticmethod
    def dot(xs: List[float], ys: List[float]) -> float:
        """Return the dot product of the given vectors."""
        return sum(x * y for x, y in zip(xs, ys))

    @staticmethod
    def matmul(xs: List[List[float]],
               ys: List[List[float]]) -> List[List[float]]:
        """Multiply the given matrices and return the resulting matrix."""
        product = []
        for x_row in range(len(xs)):
            row = []
            for y_col in range(len(ys[0])):
                col = [ys[y_row][y_col] for y_row in range(len(ys))]
                row.append(Math.dot(xs[x_row], col))
            product.append(row)
        return product

    @staticmethod
    def transpose(matrix: List[List[float]]) -> List[List[float]]:
        """Return the transposition of the given matrix."""
        return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

    @staticmethod
    def relu(z: float) -> float:
        """
        The activation function for hidden layers.
        """
        return z if z > 0 else 0.01 * z

    @staticmethod
    def relu_prime(z: float) -> float:
        """
        Return the derivative of the ReLU function.
        """
        return 1.0 if z > 0 else 0.0

    @staticmethod
    def sigmoid(z: float) -> float:
        """
        The activation function for the output layer.
        """
        epsilon = 1e-5
        return min(max(1 / (1 + math.e ** -z), epsilon), 1 - epsilon)

    @staticmethod
    def sigmoid_prime(z: float) -> float:
        """
        The activation function for the output layer.
        """
        return Math.sigmoid(z) * (1 - Math.sigmoid(z))

    @staticmethod
    def loss(actual: float, expect: float) -> float:
        """
        Return the loss between the actual and expected values.
        """
        return -(expect * math.log10(actual)
                 + (1 - expect) * math.log10(1 - actual))

    @staticmethod
    def loss_prime(actual: float, expect: float) -> float:
        """
        Return the derivative of the loss.
        """
        return -expect / actual + (1 - expect) / (1 - actual)


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #


class Layer:  # do not modify class

    def __init__(self, size: Tuple[int, int], is_output: bool) -> None:
        """
        Create a network layer with size[0] levels and size[1] inputs at each
        level. If is_output is True, use the sigmoid activation function;
        otherwise, use the ReLU activation function.

        size[0] -> num levels
        size[1] -> num inputs

        g,w,b,z,a,dw,db,

        An instance of Layer has the following attributes.

            g: The activation function - sigmoid for the output layer and ReLU
               for the hidden layer(s).

            w: The weight matrix (randomly-initialized), where each inner list
               represents the incoming weights for one neuron in the layer.

            b: The bias vector (zero-initialized), where each value represents
               the bias for one neuron in the layer.

            z: The result of (wx + b) for each neuron in the layer.

            a: The activation g(z) for each neuron in the layer.

           dw: The derivative of the weights with respect to the loss.

           db: The derivative of the bias with respect to the loss.

        """
        self.g = Math.sigmoid if is_output else Math.relu
        self.w: List[List[float]] = \
            [[random.random() * 0.1 for _ in range(size[1])]
             for _ in range(size[0])]
        self.b: List[float] = [0.0] * size[0]

        # use of below attributes is optional but recommended
        self.z: List[float] = [0.0] * size[0]
        self.a: List[float] = [0.0] * size[0]
        self.dw: List[List[float]] = \
            [[0.0 for _ in range(size[1])] for _ in range(size[0])]
        self.db: List[float] = [0.0] * size[0]

    def __repr__(self) -> str:
        """
        Return a string representation of a network layer, with each level of
        the layer on a separate line, formatted as "W | B".
        """
        s = "\n"
        fmt = "{:7.3f}"
        for i in range(len(self.w)):
            s += " ".join(fmt.format(w) for w in self.w[i])
            s += " | " + fmt.format(self.b[i]) + "\n"
        return s

    def activate(self, inputs: Tuple[float, ...]) -> Tuple[float, ...]:
        """
        Given an input (x) of the same length as the number of columns in this
        layer's weight matrix, return g(wx + b).
        """
        self.z = [Math.dot(self.w[i], inputs) + self.b[i]
                   for i in range(len(self.w))]
        self.a = [self.g(real) for real in self.z]
        return tuple(self.a)


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #


def create_samples(f: Callable[..., int], n_args: int, n_bits: int,
) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
    """
    Return a dictionary that maps inputs to expected outputs.
    """
    samples = {}
    max_arg = 2 ** n_bits
    for inputs in itertools.product((0, 1), repeat=n_args * n_bits):
        ints = [int("".join(str(bit) for bit in inputs[i:i + n_bits]), 2)
                for i in range(0, len(inputs), n_bits)]
        try:
            output = f(*ints)
            if 0 <= output < max_arg:
                bit_string = ("{:0" + str(n_bits) + "b}").format(output)
                samples[inputs] = tuple(int(bit) for bit in bit_string)
        except ZeroDivisionError:
            pass
    return samples


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

class KnowOp:

    def __init__(self, samples: Dict[Tuple[int, ...], Tuple[int, ...]],
                  i_size: int, o_size: int):

        self.samples = samples
        self.i_size = i_size
        self.o_size = o_size

        #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

        self.learning_rate = 0.1
        self.num_batches = 100
        self.num_training_iterations = 1
        self.decay = 0.85

    def train_network(self) -> List[Layer]:

        o_layer = Layer((self.o_size, self.i_size), True)
        network: List[Layer] = [o_layer]

        for _ in range(self.num_training_iterations):

            cur_batch = random.sample(self.samples.items(), self.num_batches)

            ct = 0
            for inputs in cur_batch:
                ct += 1

                # pass input onto next layer (updates z and a)
                a = propagate_forward(network, inputs[0])
                # a is List[float]

                running_sum = 0
                len_output = len(a)
                da = []
                for j in range(len_output):
                    loss = Math.loss(a[j], inputs[1][j])
                    loss_prime = Math.loss_prime(a[j], inputs[1][j])
                    da.append(loss_prime)
                    running_sum += loss

                cost = running_sum / len_output # TODO save for updating w,b

                propagate_backward(network, inputs[0], da)

                # update w and b

            pass

        return network


def propagate_backward(network: List[Layer],\
                       inputs: Tuple[int, ...], da: List[float]):
    """

    input: dan (dal)
    output: dan-1, dwn, dbn
    """

    # TODO only accounting for output (sigmoid) here
    for i in range(len(network)):
        z_col = network[i].z
        z_col_sigmoid = []
        for j in range(len(z_col)):
            z = z_col[j]
            z_col_sigmoid.append(Math.sigmoid_prime(z))

        # first eqn: dz = da * g'(z)
        dz = [x * y for x, y in zip(da, z_col_sigmoid)]

        # second eqn: dW = dz . aTn-1

        trans = Math.transpose(network[i].a)

        # third eqn:










        pass


def train_network(samples: Dict[Tuple[int, ...], Tuple[int, ...]],
                  i_size: int, o_size: int) -> List[Layer]:
    """
    Given a training set (with labels) and the sizes of the input and output
    layers, create and train a network by iteratively propagating inputs
    (forward) and their losses (backward) to update its weights and biases.
    Return the resulting trained network.
    """

    know_op = KnowOp(samples, i_size, o_size)
    layers = know_op.train_network()
    return layers


def propagate_forward(network: List[Layer], inputs: Tuple[int, ...]) \
        -> List[float]:

    i = 0
    for i in range(len(network)):
        a = network[i].activate(inputs)

    return list(a)


def old_propagate_forward(network: List[Layer], inputs: Tuple[int, ...])\
        -> List[float]:

    pre_x = list(inputs)
    x = []
    x.append(pre_x)
    x = Math.transpose(x)

    # num_neurons = len(network[0].b)

    # for each layer in network (only 1 rn...)
    for i in range(len(network)):

        w = network[i].w
        b = network[i].b
        wx = Math.matmul(w, x)

        for j in range(len(wx)):
            network[i].z[j] = wx[j][0] + b[j]
            network[i].a[j] = network[i].g(network[i].z[j])

    # compute loss? with dict? or is z y hat?

    # TODO: i?
    return network[i].z


def main() -> None:
    test()
    random.seed(0)
    f = lambda x, y: x + y  # operation to learn
    # f = lambda x: x // 2
    n_args = 2              # arity of operation
    n_bits = 8              # size of each operand

    samples = create_samples(f, n_args, n_bits)
    train_pct = 0.95
    train_set = {inputs: samples[inputs]
               for inputs in random.sample(list(samples),
                                           k=int(len(samples) * train_pct))}
    test_set = {inputs: samples[inputs]
               for inputs in samples if inputs not in train_set}
    print("Train Size:", len(train_set), "Test Size:", len(test_set))

    network = train_network(train_set, n_args * n_bits, n_bits)
    for inputs in test_set:
        output = tuple(round(n, 2) for n in propagate_forward(network, inputs))
        bits = tuple(round(n) for n in output)
        print("OUTPUT:", output)
        print("BITACT:", bits)
        print("BITEXP:", samples[inputs], end="\n\n")

def test() -> None:
    one = [[2, 3, 4],
           [2, 3, 4]]
    two = [[2, 3, 4],
           [2, 3, 4]]
    three = Math.matmul(one, two)
    print()

if __name__ == "__main__":
    main()