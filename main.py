import jax

from parallax import Module, Parameter, ModuleTuple, ParameterTuple, ParamInit, Dense, BinaryNetwork, MultiLayerLSTM


def test_paraminit_instantiation():
    rng = jax.random.PRNGKey(0)
    p1 = ParamInit((2, 3), jax.nn.initializers.normal()).instantiate(rng)
    p2 = ParamInit((3, 5), jax.nn.initializers.normal()).instantiate(rng)
    assert (p1 @ p2).shape == (2, 5)


def test_parametertuple_pytree():
    t = ParameterTuple([
        ParamInit((2, 3), jax.nn.initializers.normal()),
        ParamInit((3, 5), jax.nn.initializers.normal()),
    ])
    _ = t.instantiate(jax.random.PRNGKey(0))
    t2 = ParameterTuple([t, t, t]).instantiate(jax.random.PRNGKey(0))
    l, a = t2.tree_flatten()
    assert t2 == ParameterTuple.tree_unflatten(a, l)


def test_moduletuple_pytree():
    t = ModuleTuple([Dense(2, 3), Dense(3, 5)])
    t2 = ModuleTuple([t, t, t])
    l, a = t2.initialized(jax.random.PRNGKey(0)).tree_flatten()
    _ = ModuleTuple.tree_unflatten(a, l)
    # TODO: test for equality...


def test_feedforward_network_learns():
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
    assert loss < .5 * initial_loss


@jax.jit
def continuous_lstm_model(lstm, all_data):
    hcs = lstm.hc_0s
    losses = jax.numpy.zeros(all_data.shape[-1])
    all_data = jax.numpy.split(all_data, all_data.shape[-2], axis=-2)
    for in_data, out_data in zip(all_data[:-1], all_data[1:]):
        hcs = lstm(jax.numpy.squeeze(in_data, axis=-2), hcs)
        losses += jax.numpy.sum((hcs[-1][0] - out_data) ** 2, axis=-1)
    return jax.numpy.sum(losses, axis=-1)


def test_lstm_single_example():
    lstm = MultiLayerLSTM(n_layers=3, n_hidden=7).initialized(jax.random.PRNGKey(0))

    # 5x7 matrix of rank 2
    data = jax.random.normal(jax.random.PRNGKey(0), (5, 2)) @ jax.random.normal(jax.random.PRNGKey(0), (2, 7))

    initial_loss = None
    for i, data_rng in enumerate(range(101)):
        if i % 10 == 0:
            loss = continuous_lstm_model(lstm, data)
            if initial_loss is None:
                initial_loss = loss
            print(loss)
        grad = jax.grad(continuous_lstm_model)(lstm, data)
        lstm = jax.tree_util.tree_multimap(lambda p, g: p - 0.005 * g, lstm, grad)
    assert loss < .5 * initial_loss


def test_lstm_vmapped():
    # let's make this one a bit more beefy, it has a harder task after all
    lstm = MultiLayerLSTM(n_layers=3, n_hidden=7).initialized(jax.random.PRNGKey(0))

    vmapped_fn = jax.vmap(continuous_lstm_model, in_axes=(None, 0))
    grad_fn = jax.jit(jax.grad(lambda l, d: jax.numpy.sum(vmapped_fn(l, d))))

    data_batch = jax.random.normal(jax.random.PRNGKey(0), (3, 5, 2)) @ jax.random.normal(jax.random.PRNGKey(0), (2, 7))

    initial_loss = None
    for i, data_rng in enumerate(range(3001)):
        if i % 300 == 0:
            loss = jax.numpy.array([
                continuous_lstm_model(lstm, data_batch[i, :, :])
                for i in range(data_batch.shape[0])
            ])
            if initial_loss is None:
                initial_loss = loss
            print(loss)
        grad = grad_fn(lstm, data_batch)
        lstm = jax.tree_util.tree_multimap(lambda p, g: p - 0.003 * g, lstm, grad)
    
    # TODO curiously there are slight differences that propagate between running this in Colab and on my machine :(
    assert jax.numpy.all(loss < .6 * initial_loss)


if __name__ == "__main__":
    test_paraminit_instantiation()
    test_parametertuple_pytree()
    test_moduletuple_pytree()
    test_feedforward_network_learns()
    test_lstm_single_example()
    test_lstm_vmapped()
