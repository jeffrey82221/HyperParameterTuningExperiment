import gc
from numba import cuda
import tensorflow as tf
from XORNet import xor_net
from XORUtil import get_xor_generator, get_xor_data
from XORStopper import GoalReachedStopper, RandomAccuracyStopper
import time
import numpy as np
'''
from XORTrainerFunc import xor_trainer_function
val_acc, training_time = xor_trainer_function(40, -1, 0.95, 10, 2)
'''


class TimeHistory(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.times = []

  def on_epoch_begin(self, batch, logs={}):
    self.epoch_time_start = time.time()

  def on_epoch_end(self, batch, logs={}):
    self.times.append(time.time() - self.epoch_time_start)


def xor_trainer_function(batch_size_continuous, lr_exp, momentum, layer_size_continuous,
                         layer_count_continuous):
  # parameter initialize
  batch_size = int(batch_size_continuous / 4) * 4
  lr = 10**lr_exp  # float(sys.argv[3])
  momentum = momentum
  layer_size = int(layer_size_continuous)
  layer_count = int(layer_count_continuous)

  op = tf.keras.optimizers.SGD(lr=lr, momentum=momentum,
                               nesterov=True)
  model = xor_net(layer_size, layer_count, verbose=False)
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  model.compile(loss=loss, optimizer=op, metrics=['acc'])

  stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_acc',
                                                patience=1000,
                                                verbose=1)
  stop_if_goal_reached = GoalReachedStopper()
  stop_if_no_improve = RandomAccuracyStopper(patience=100)
  time_callback = TimeHistory()

  validation_size = 4
  xor_data_generator = get_xor_generator(batch_size)
  xor_validation_data = get_xor_data(validation_size)
  history = model.fit_generator(
      xor_data_generator,
      steps_per_epoch=1,  # this is a virtual parameter
      epochs=1000000,
      validation_data=xor_validation_data,
      verbose=0,
      callbacks=[stop_early, stop_if_goal_reached, stop_if_no_improve, time_callback],
      # use_multiprocessing=True,
      # workers=2,
      # max_queue_size=100
  )
  epoch_computation_time = np.mean(time_callback.times)

  # make sure the input of each pixel is between 0 and 1
  # get final val acc
  final_val_acc = history.history['val_acc'][-1]
  #print("ANS:", scale, small_filter_rate, final_val_acc, model.count_params())
  del model
  gc.collect()
  tf.keras.backend.clear_session()
  cuda.select_device(0)
  cuda.close()
  return final_val_acc, len(history.history['val_acc']) * epoch_computation_time
