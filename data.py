import tensorflow as tf
import jax.numpy as jnp

def load_mnist():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize and flatten
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0

    # Convert to one-hot
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

def create_data_iterators(x_train, y_train, x_test, y_test, batch_size=64):
    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_train, y_train))
        .repeat()
        .shuffle(buffer_size=5000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

    test_dataset = (
        tf.data.Dataset
        .from_tensor_slices((x_test, y_test))
        .shuffle(buffer_size=1000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

    return train_dataset, test_dataset

def get_batch(iterator):
    images, labels = next(iterator)
    return jnp.array(images), jnp.array(labels, dtype=jnp.float32)