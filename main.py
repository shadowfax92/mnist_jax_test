import jax
import jax.random as jr
from config import configure_xla
from data import load_mnist, create_data_iterators, get_batch
from model import Model
from training import setup_training, setup_mesh_sharding, train_step
from jax.sharding import NamedSharding, PartitionSpec as PS

def main():
    # Configure XLA
    # configure_xla()
    
    # Print available devices
    print("Available devices:")
    print(jax.devices())
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    batch_size = 32
    train_iterator, test_iterator = create_data_iterators(x_train, y_train, x_test, y_test)
    
    # Initialize model
    model = Model(jr.PRNGKey(99))
    
    # Setup training
    params, static, opt, opt_state = setup_training(model)
    
    # Setup mesh sharding
    params, mesh = setup_mesh_sharding(params)
    
    # Training hyperparameters
    num_epochs = 10
    steps_per_epoch = len(x_train) // batch_size  # Use batch_size variable instead of accessing it from iterator

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            images, labels = get_batch(train_iterator)
            images_sharded = jax.device_put(images, NamedSharding(mesh, PS('x')))
            labels_sharded = jax.device_put(labels, NamedSharding(mesh, PS('x')))
            
            params, opt_state, loss = train_step(params, static, opt, opt_state, 
                                               (images_sharded, labels_sharded))
            epoch_loss += loss
        
        # Print epoch results
        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    print("Training completed!")

if __name__ == "__main__":
    main()