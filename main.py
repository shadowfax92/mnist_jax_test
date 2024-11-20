import jax
import jax.random as jr
from config import configure_xla
from data import load_mnist, create_data_iterators, get_batch
from model import Model
from training import setup_training, setup_mesh_sharding, train_step

def main():
    # Configure XLA
    # configure_xla()
    
    # Print available devices
    print("Available devices:")
    print(jax.devices())
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    train_iterator, test_iterator = create_data_iterators(x_train, y_train, x_test, y_test)
    
    # Initialize model
    model = Model(jr.PRNGKey(99))
    
    # Setup training
    params, static, opt, opt_state = setup_training(model)
    
    # Setup mesh sharding
    params, mesh = setup_mesh_sharding(params)
    
    # Get a batch and shard it
    images, labels = get_batch(train_iterator)
    images_sharded = jax.device_put(images, NamedSharding(mesh, PS('x', 'y')))
    labels_sharded = jax.device_put(labels, NamedSharding(mesh, PS('x', 'y')))
    
    # Test a training step
    try:
        model, opt_state, loss = train_step(params, static, opt, opt_state, 
                                          (images_sharded, labels_sharded))
        print(f"Initial training loss: {loss}")
        print("JAX training test successful!")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()