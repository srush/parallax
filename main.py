from parallax import module, Module, Parameter
import torch
import torch.nn.init as init

# Everything is immutable @module =  dataclass(frozen=True, repr=False)
@module
class Dense(Module):

    # All parameter-holders are explicitly declared.
    weight : Parameter
    bias : Parameter

    # Setup is creates shapes and binds init methods.
    @staticmethod
    def setup(in_size, out_size):
        return Dense.init(
            weight = Parameter.setup((out_size, in_size),
                                     init.xavier_normal_),
            bias = Parameter.setup((out_size,),
                                   init.normal_))

    # Forward is associated with parameters
    def forward(self, input):
        return self.weight @ input + self.bias

    def extra_repr(self):
        return "%d, %d"%(self.weight.shape[1], self.weight.shape[0])

@module
class Dropout(Module):
    # Other constants allowed.
    rate : float

    @staticmethod
    def setup(rate):
        return Dropout.init(rate = rate)

    def forward(self, input):
        # RNG state is use-once or split. Attached to tree.
        torch.random.set_rng_state(self.rng)
        return torch.nn.functional.dropout(input, p=self.rate,
                                           training=self.mode == "train")

@module
class BinaryNetwork(Module):

    # No difference between modules and parameters
    dense1 : Dense
    dense2 : Dense
    dense3 : Dense
    dropout : Dropout

    @staticmethod
    def setup(input_size, hidden_size):
        return BinaryNetwork.init(
            dense1 = Dense.setup(input_size, hidden_size),
            dense2 = Dense.setup(hidden_size, hidden_size),
            dense3 = Dense.setup(hidden_size, 1),
            dropout = Dropout.setup(rate=0.2)
        )

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
layer = BinaryNetwork.setup(5, 10)

print(layer)
assert(layer.dense1.weight.shape == (10, 5))


# Initialize parameters -> stateful, hidden
rng = torch.random.get_rng_state()
layer = layer.initialize(rng)

assert(layer.dense1.weight.shape == (10, 5))
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
