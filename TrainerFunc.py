import sys
import gc
from numba import cuda
import pickle
import tensorflow as tf
from XORNet import xor_net
from sklearn.preprocessing import OneHotEncoder


def TrainerFunction(batch_size, lr_exp, momentum, layer_size, layer_count):
  # parameter initialize
  batch_size = batch_size
  lr = 10**lr_exp  # float(sys.argv[3])
  momentum = momentum
  layer_size = layer_size
  layer_count = layer_count

  op = tf.keras.optimizers.SGD(
      lr=lr,
      momentum=momentum,
      nesterov=True
  )  # , decay=1e-6, momentum=0.9)
  model = xor_net(layer_size, layer_count, verbose=False)
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  model.compile(loss=loss,
                optimizer=op,
                metrics=['acc'])
  stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_acc',
                                                patience=5,
                                                verbose=1)
  validation_size = 1000
  xor_data_generator = get_xor_generator(batch_size)
  xor_validation_data = get_xor_valid_data(validation_size)
  history = model.fit_generator(xor_data_generator,
                                steps_per_epoch=100,  # this is a virtual parameter
                                epochs=10000,
                                batch_size=batch_size,
                                validation_data=xor_validation_data,
                                verbose=2,
                                callbacks=[stop_early],
                                multiprocessing=True
                                )
  # make sure the input of each pixel is between 0 and 1
  # get final val acc
  final_val_acc = history.history['val_acc'][-1]
  #print("ANS:", scale, small_filter_rate, final_val_acc, model.count_params())
  del model
  gc.collect()
  tf.keras.backend.clear_session()
  cuda.select_device(0)
  cuda.close()
  return final_val_acc
