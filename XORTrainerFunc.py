import gc
#from numba import cuda
import tensorflow as tf
from XORNet import xor_net
from XORUtil import get_xor_generator, get_xor_data
from XORStopper import OneSecondStopper, GoalReachedStopper  # GoalReachedStopper, RandomAccuracyStopper
from Util import TimeHistory, get_model_memory_usage
import time
import numpy as np
'''
from XORTrainerFunc import initial_slop
val_acc, training_time = initial_slop(40, -1, 0.95, 10, 2)

model = xor_net(100, 1, verbose=False)
'''


def initial_slop(batch_size_continuous, lr_exp, momentum,
                 layer_size_continuous, layer_count_continuous):
  # parameter initialize
  batch_size = int(batch_size_continuous / 4) * 4
  lr = 10**lr_exp  # float(sys.argv[3])
  momentum = momentum
  layer_size = int(layer_size_continuous)
  layer_count = int(layer_count_continuous)

  op = tf.keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=True)
  model = xor_net(layer_size, layer_count, verbose=False)
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  model.compile(loss=loss, optimizer=op, metrics=['acc'])

  time_callback = TimeHistory()

  validation_size = 4
  xor_data_generator = get_xor_generator(batch_size)
  xor_validation_data = get_xor_data(validation_size)
  history = model.fit_generator(xor_data_generator,
                                steps_per_epoch=1,
                                epochs=10,
                                validation_data=xor_validation_data,
                                verbose=2,
                                callbacks=[time_callback])
  computation_time = np.sum(time_callback.times)
  slop = (history.history['val_loss'][0] -
          history.history['val_loss'][-1]) / computation_time
  del model
  gc.collect()
  tf.keras.backend.clear_session()

  return slop


def initial_acc(batch_size_continuous, lr_exp, momentum, layer_size_continuous,
                layer_count_continuous):
  # parameter initialize
  batch_size = int(batch_size_continuous / 4) * 4
  lr = 10**lr_exp
  momentum = momentum
  layer_size = int(layer_size_continuous)
  layer_count = int(layer_count_continuous)

  op = tf.keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=True)
  model = xor_net(layer_size, layer_count, verbose=False)
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  model.compile(loss=loss, optimizer=op, metrics=['acc'])

  one_second_stop_callback = OneSecondStopper()

  validation_size = 4
  xor_data_generator = get_xor_generator(batch_size)
  xor_validation_data = get_xor_data(validation_size)
  history = model.fit_generator(
      xor_data_generator,
      steps_per_epoch=1,  # this is a virtual parameter
      epochs=10,
      validation_data=xor_validation_data,
      verbose=2,
      callbacks=[one_second_stop_callback])
  final_acc = history.history['val_acc'][-1]
  del model
  gc.collect()
  tf.keras.backend.clear_session()
  return final_acc


def perfect_acc_time(batch_size_continuous, lr_exp, momentum,
                     layer_size_continuous, layer_count_continuous):
  # parameter initialize
  batch_size = int(batch_size_continuous / 4) * 4
  lr = 10**lr_exp  # float(sys.argv[3])
  momentum = momentum
  layer_size = int(layer_size_continuous)
  layer_count = int(layer_count_continuous)

  op = tf.keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=True)
  model = xor_net(layer_size, layer_count, verbose=False)
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  model.compile(loss=loss, optimizer=op, metrics=['acc'])

  stop_if_goal_reached = GoalReachedStopper()
  time_callback = TimeHistory()
  validation_size = 4
  xor_data_generator = get_xor_generator(batch_size)
  xor_validation_data = get_xor_data(validation_size)
  history = model.fit_generator(
      xor_data_generator,
      steps_per_epoch=1,
      epochs=100,
      validation_data=xor_validation_data,
      verbose=0,
      callbacks=[stop_if_goal_reached, time_callback])
  computation_time = np.sum(time_callback.times)
  epoch_usage = len(history.history['val_acc'])
  final_acc = history.history['val_acc'][-1]
  print(
      "computation_time:",
      computation_time,
      "epoch size:",
      epoch_usage,
      "acc:",
      final_acc
  )
  del model
  gc.collect()
  tf.keras.backend.clear_session()
  if history.history['val_acc'][-1] < 1.0:
    return 0.
  else:
    return 1. / computation_time


def model_contruction_time(batch_size_continuous, lr_exp, momentum, layer_size_continuous,
                           layer_count_continuous):
  construct_start_time = time.time()
  batch_size = int(batch_size_continuous / 4) * 4
  lr = 10**lr_exp
  momentum = momentum
  layer_size = int(layer_size_continuous)
  layer_count = int(layer_count_continuous)

  op = tf.keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=True)
  model = xor_net(layer_size, layer_count, verbose=False)
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  model.compile(loss=loss, optimizer=op, metrics=['acc'])

  validation_size = 4
  xor_data_generator = get_xor_generator(batch_size)
  xor_validation_data = get_xor_data(validation_size)
  model.fit_generator(
      xor_data_generator,
      steps_per_epoch=1,  # this is a virtual parameter
      epochs=1,
      validation_data=xor_validation_data,
      verbose=2)
  del model
  gc.collect()
  tf.keras.backend.clear_session()
  construction_time = time.time() - construct_start_time
  return 1. / construction_time


def memory_efficiency(batch_size_continuous, lr_exp, momentum,
                      layer_size_continuous, layer_count_continuous):
  # parameter initialize
  batch_size = int(batch_size_continuous / 4) * 4
  lr = 10**lr_exp  # float(sys.argv[3])
  momentum = momentum
  layer_size = int(layer_size_continuous)
  layer_count = int(layer_count_continuous)

  op = tf.keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=True)
  model = xor_net(layer_size, layer_count, verbose=False)
  loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  model.compile(loss=loss, optimizer=op, metrics=['acc'])

  stop_if_goal_reached = GoalReachedStopper()
  time_callback = TimeHistory()
  validation_size = 4
  xor_data_generator = get_xor_generator(batch_size)
  xor_validation_data = get_xor_data(validation_size)
  history = model.fit_generator(
      xor_data_generator,
      steps_per_epoch=1,
      epochs=5,
      validation_data=xor_validation_data,
      verbose=0,
      callbacks=[stop_if_goal_reached, time_callback])
  memory_usage = get_model_memory_usage(batch_size, model)
  epoch_usage = len(history.history['val_acc'])
  final_acc = history.history['val_acc'][-1]
  print(
      "memory_usage:",
      memory_usage,
      "epoch size:",
      epoch_usage,
      "acc:",
      final_acc
  )
  memory_efficiency = 1. / memory_usage

  del model
  gc.collect()
  tf.keras.backend.clear_session()
  if final_acc < 1.0 or epoch_usage > 3:
    # The scenario we want to avoid:
    # val_acc < 1.0 or training epoch size > 3
    return 0. - epoch_usage + final_acc
  else:
    return memory_efficiency
