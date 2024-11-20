import jax
import equinox as eqx
import optax
import functools
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as PS

def forward_pass(model, batch):
    inputs, labels = batch
    logits = jax.vmap(model, in_axes=(0), out_axes=(0))(inputs)
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels)
    return loss.mean(), logits

def backward_pass(model, opt, opt_state, batch):
    grad_fn = jax.value_and_grad(forward_pass, argnums=(0), has_aux=True)
    (loss, _), grads = grad_fn(model, batch)
    updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

@functools.partial(jax.jit, static_argnums=(1, 2))
def train_step(params, static, opt, opt_state, batch):
    model = eqx.combine(params, static)
    return backward_pass(model, opt, opt_state, batch)

def setup_training(model, learning_rate=0.001):
    params, static = eqx.partition(model, eqx.is_array)
    opt = optax.sgd(learning_rate=learning_rate)
    opt_state = opt.init(params)
    return params, static, opt, opt_state

def setup_mesh_sharding(params):
    devices = mesh_utils.create_device_mesh((2, 2, 2))
    mesh = Mesh(devices, axis_names=('x', 'y', 'z'))
    
    params = eqx.tree_at(
        lambda tree: tree.layer1.weight,
        params,
        replace_fn=lambda node: jax.device_put(node, NamedSharding(mesh, PS('x', 'y')))
    )
    params = eqx.tree_at(
        lambda tree: tree.layer2.weight,
        params,
        replace_fn=lambda node: jax.device_put(node, NamedSharding(mesh, PS('x', 'y')))
    )
    
    return params, mesh