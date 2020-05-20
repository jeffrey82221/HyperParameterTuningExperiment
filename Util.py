import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
import time


class TimeHistory(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.times = []

  def on_epoch_begin(self, batch, logs={}):
    self.epoch_time_start = time.time()

  def on_epoch_end(self, batch, logs={}):
    self.times.append(time.time() - self.epoch_time_start)


'''
from XORNet import xor_net
model = xor_net(100, 1, verbose=False)
from Util import get_model_memory_usage
get_model_memory_usage(4, model)
from tensorflow.keras import backend as K
import numpy as np
trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])


shapes_mem_count = 0
internal_model_mem_count = 0
for l in model.layers:
  layer_type = l.__class__.__name__
  single_layer_mem = 1
  if type(l.output_shape) == list:
    for s in l.output_shape[0]:
      if s is None:
        continue
      single_layer_mem *= s
  else:
    for s in l.output_shape:
      if s is None:
        continue
      single_layer_mem *= s
  shapes_mem_count += single_layer_mem
'''


def get_model_memory_usage(batch_size, model):
  shapes_mem_count = 0
  internal_model_mem_count = 0
  for l in model.layers:
    layer_type = l.__class__.__name__
    if layer_type == 'Model':
      internal_model_mem_count += get_model_memory_usage(batch_size, l)
    single_layer_mem = 1
    if type(l.output_shape) == list:
      for s in l.output_shape[0]:
        if s is None:
          continue
        single_layer_mem *= s
    else:
      for s in l.output_shape:
        if s is None:
          continue
        single_layer_mem *= s
    shapes_mem_count += single_layer_mem

  trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
  non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

  number_size = 4.0
  if K.floatx() == 'float16':
    number_size = 2.0
  if K.floatx() == 'float64':
    number_size = 8.0

  total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
  gbytes = total_memory / (1024.0 ** 3) + internal_model_mem_count
  return gbytes
