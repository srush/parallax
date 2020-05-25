import jax
import jax.nn.initializers as init

from .core import *


class Dense(Module):
    # All parameter-holders are explicitly declared.
    weight : Parameter
    bias : Parameter

    # Setup replace __init__ and creates shapes and binds lazy initializers.
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = ParamInit((out_size, in_size), init.xavier_normal())
        self.bias = ParamInit((out_size,), init.normal())

    # Forward is just like standard pytorch.
    def forward(self, input):
        return self.weight @ input + self.bias

    # Hook for pretty printing
    def extra_repr(self):
        return "%d, %d"%(self.weight.shape[1], self.weight.shape[0])


class Dropout(Module):
    # Arbitrary constants allowed.
    rate : float
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        # RNG state is use-once or split. Attached to tree.
        state = self.rng

        if self.mode == "train":
            keep = jax.random.bernoulli(state, self.rate, input.shape)
            return jax.numpy.where(keep, input / self.rate, 0)
        else:
            return input


class BinaryNetwork(Module):

    # No difference between modules and parameters
    dense1 : Dense
    dense2 : Dense
    dense3 : Dense
    dropout : Dropout

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.dense1 = Dense(input_size, hidden_size)
        self.dense2 = Dense(hidden_size, hidden_size)
        self.dense3 = Dense(hidden_size, 1)
        self.dropout = Dropout(0.2)

    def forward(self, input):

        # Standard usage works out of the box.
        x = jax.numpy.tanh(self.dense1(input))

        # Stochastic modules (have random seed already)
        x = self.dropout(x)

        # Shared params / recurrence only requires split to change RNG
        x = jax.numpy.tanh(self.dense2(x))
        x = jax.numpy.tanh(self.dense2(x))

        return jax.nn.sigmoid(self.dense3(jax.numpy.tanh(x)))[0]


class LSTMCell(Module):
    weight_ih : Parameter
    linear_hh : Dense

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weight_ih = ParamInit((input_size, 4 * hidden_size), init.normal())
        self.linear_hh = Dense(input_size, 4 * hidden_size)

    def forward(self, input, h, c):
        ifgo = self.weight_ih.T @ input + self.linear_hh(h)
        i, f, g, o = jax.numpy.split(ifgo, indices_or_sections=4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jax.numpy.tanh(g)
        o = jax.nn.sigmoid(o)
        new_c = f * c + i * g
        new_h = o * jax.numpy.tanh(new_c)
        return (new_h, new_c)


class MultiLayerLSTM(Module):
    # Dynamic number of parameters and modules
    cells : ModuleTuple
    c_0s : ParameterTuple

    def __init__(self, n_layers, n_hidden):
        """For simplicity, have everything have the same dimension."""
        super().__init__()
        self.cells = ModuleTuple([
            LSTMCell(n_hidden, n_hidden)
            for _ in range(n_layers)
        ])
        self.c_0s = ParameterTuple([
            ParamInit((n_hidden,), init.normal())
            for _ in range(n_layers)
        ])
    
    @property
    def hc_0s(self):
        return tuple((jax.numpy.tanh(c_0), c_0) for c_0 in self.c_0s)

    @jax.jit  # a lot faster (try it without!)
    def forward(self, input, hcs):
        new_hcs = []
        for cell, hc in zip(self.cells, hcs):
            input, c = cell(input, *hc)
            new_hcs.append((input, c))
        return tuple(new_hcs)
