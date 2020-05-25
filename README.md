# Parallax - Immutable Torch Modules for JAX


<img width=450px src="https://developers.google.com/web/updates/images/2016/12/performant-parallaxing/parallax.jpg">


Parallax is a prototype for a pure module system for JAX implemented by Sabrina Mielke (@sjmielke) and Sasha Rush (@srush).

Main ideas:

* Make param modules immutable trees.
* Replace all imperative style coding and init.
* Avoid tracking state for most applications by first distributing seeds / globals through tree.

```python

from parallax import Module, Parameter, ParamInit

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

# Setup param tree -> declarative, immutable
layer = BinaryNetwork(5, 10)
print(layer)
print(layer.dense1)

# Initialize parameters -> stateful, hidden
rng = jax.random.PRNGKey(0)
layer = layer.initialized(rng)
print(layer)
print(layer.dense1)

initial_loss = None
for i in range(10):
    # Thread state through parameters -> functor, hidden
    rng, iter_rng = jax.random.split(rng)
    layer = layer.new_state(iter_rng, mode="train")
    
    # Jax style grad compute -> tree-shaped immutable
    x = jax.numpy.zeros(5)
    loss = layer(x)
    if initial_loss is None:
        initial_loss = loss
    print(loss)
    grad = layer.grad(x)
    
    # Grad Update -> tree-shaped
    layer = jax.tree_util.tree_multimap(lambda p, g: p - 0.3 * g, layer, grad)
```
