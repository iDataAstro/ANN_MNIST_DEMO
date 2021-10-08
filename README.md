# ANN_MNIST_DEMO
* ANN using MNIST dataset
  * Creates ANN model using following layers:
  ```python
      LAYERS = [
          tf.keras.layers.Flatten(input_shape=input_shape, name='input_layer'),
          tf.keras.layers.Dense(units=392, activation='relu', name='hidden1'),
          tf.keras.layers.Dense(units=196, activation='relu', name='hidden2'),
          tf.keras.layers.Dense(units=no_classes, activation='softmax', name='output_layer')
    ]
  ```
  * Train on MNIST dataset for 30 epochs and save trained model.
