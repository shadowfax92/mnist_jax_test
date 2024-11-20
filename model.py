import equinox as eqx
import jax
import jax.numpy as jnp

class Model(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear

    def __init__(self, key):
        key1, key2 = jax.random.split(key, 2)
        self.layer1 = eqx.nn.Linear(in_features=784, out_features=16, key=key1, use_bias=False)
        self.layer2 = eqx.nn.Linear(in_features=16, out_features=10, key=key2, use_bias=False)

    def __call__(self, x):
        x = jax.nn.relu(self.layer1(x))
        x = self.layer2(x)
        return x