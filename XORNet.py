import tensorflow as tf


def xor_net(layer_size, layer_count, verbose=True):
  # Network according to Table I in the original paper:
  #IMAGE_SIZE = [32, 32]#list(train_pipe.target_size)#
  # CLASS_NUM = 100#len(train_pipe.class_indices)
  assert layer_count >= 1
  x = tf.keras.layers.Input(shape=[2], name='2_bit_input')
  # input is 192x192 pixels RGB (3 channels)
  y = tf.keras.layers.Dense(layer_size, activation="relu", name="layer1")(x)
  for i in range(layer_count - 1):
    y = tf.keras.layers.Dense(layer_size,
                              activation="relu",
                              name="layer" + str(i + 2))(y)
  y = tf.keras.layers.Dense(1, activation="sigmoid", name="last_layer")(y)
  model = tf.keras.Model(x, y, name='XorNet')
  if verbose:
    model.summary()
  return model
