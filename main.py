from parallax import Module, Parameter
import torch
import torch.nn.init as init


class Dense(Module):
    # All parameter-holders are explicitly declared.
    weight : Parameter
    bias : Parameter

    # Setup replace __init__ and creates shapes and binds lazy initializers.
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weight = Parameter((out_size, in_size), init.xavier_normal_)
        self.bias = Parameter((out_size,), init.normal_)

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

        # Pretend torch is pure.
        torch.random.set_rng_state(self.rng)
        out = torch.nn.functional.dropout(input, p=self.rate,
                                          training=self.mode == "train")
        #
        return out

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
        x = torch.tanh(self.dense1(input))

        # Stochastic modules (have random seed already)
        x = self.dropout(x)

        # Shared params / recurrence requires split (like RNG)
        dense2_a, dense2_b = self.dense2.split(2)
        x = torch.tanh(dense2_a(x))
        x = torch.tanh(dense2_b(x))

        return torch.sigmoid(self.dense3(
               torch.tanh(x)))

# Setup paramm tree -> declarative, immutable
layer = BinaryNetwork(5, 10)
print(layer)

# Initialize parameters -> stateful, hidden
rng = torch.random.get_rng_state()
layer = layer.initialize(rng)
print(layer)

for i in range(10):
    # Thread state through parameters -> functor, hidden
    rng = torch.random.get_rng_state()
    layer = layer.init_state(rng, mode="train")

    # Jax style grad compute -> tree-shaped immutable
    x = torch.zeros(5, requires_grad=True)
    def mock_grad():
        out = layer.forward(x)
        out.backward()
        return layer.grad()
    grad = mock_grad()

    # Grad Update -> tree-shaped
    new_layer = layer.update(lambda a, b: a + b, grad)
