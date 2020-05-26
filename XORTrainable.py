import gc
#from numba import cuda
import tensorflow as tf
from XORNet import xor_net
from XORUtil import get_xor_generator, get_xor_data
from XORStopper import OneSecondStopper, GoalReachedStopper  # GoalReachedStopper, RandomAccuracyStopper
from Util import TimeHistory, get_model_memory_usage
import time
import numpy as np
from ray.tune import Trainable
from ray.tune.integration.keras import TuneReporterCallback


'''
from XORNet import xor_net
from XORTrainerFunc import initial_slop
initial_slop(40, -1, 0.95, 10, 2)

model = xor_net(100, 1, verbose=False)
'''

class XORTrainable(Trainable):
  def _build_model(self):
    model = xor_net(
      self.layer_size, 
      self.layer_count, verbose=False)
    return model 
  def _setup(self, config):
    self.batch_size = int(config["batch_size_continuous"] / 4) * 4
    self.lr = 10**config["lr_exp"]  # float(sys.argv[3])
    self.momentum = config["momentum"]
    self.layer_size = int(config["layer_size_continuous"])
    self.layer_count = int(config["layer_count_continuous"])
    self.epochs = int(config['epochs'])
    model = self._build_model()
    op = tf.keras.optimizers.SGD(lr=self.lr, 
      momentum=self.momentum, nesterov=True)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=op, metrics=['acc'])
    self.model = model 
  def _callbacks(self):
    tune_reporter_callback = TuneReporterCallback()
    self.callbacks = {
      "tune_reporter_callback": tune_reporter_callback
    }
    return self.callbacks  
  def _train(self):
    xor_data_generator = get_xor_generator(self.batch_size)
    validation_size = 4
    xor_validation_data = get_xor_data(validation_size)
    self.history = model.fit_generator(xor_data_generator,
                                  steps_per_epoch=1,
                                  epochs=self.epochs,
                                  validation_data=xor_validation_data,
                                  verbose=0,
                                  callbacks=self._callbacks().values())
    return self._get_result()
  def _get_result(self):
    return {
      "mean_accuracy" : self.history.history['val_acc'][-1],
      "mean_loss": self.history.history['val_loss'][-1] 
    }
      
    
  def _save(self, checkpoint_dir):
    file_path = checkpoint_dir + '/model'
    self.model.save_weights(file_path)
    return file_path
  def _restore(self, path):
    self.model.load_weights(path) 
  def _stop(self):
    # If need, save your model when exit.
    # saved_path = self.model.save(self.logdir)
    # print('save model at: ', saved_path)
    pass

    
'''
def initial_slop(batch_size_continuous,
                 lr_exp,
                 momentum,
                 layer_size_continuous,
                 layer_count_continuous,
                 tune_report_call_back=None):
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
  if tune_report_call_back == None:
    callbacks = [time_callback]
  else:
    callbacks = [time_callback, tune_report_call_back]
  history = model.fit_generator(xor_data_generator,
                                steps_per_epoch=1,
                                epochs=10,
                                validation_data=xor_validation_data,
                                verbose=0,
                                callbacks=callbacks)
  computation_time = np.sum(time_callback.times)
  slop = (history.history['val_loss'][0] -
          history.history['val_loss'][-1]) / computation_time
  del model
  gc.collect()
  tf.keras.backend.clear_session()

  return slop
'''
class InitialSlopTrainable(XORTrainable):
  def _callbacks(self):
    time_callback = TimeHistory()
    tune_reporter_callback = TuneReporterCallback()
    self.callbacks = {
      "time_callback":time_callback,
      "tune_reporter_callback":tune_reporter_callback
    }
    return self.callbacks
  def _get_result(self):
    computation_time = np.sum(self.callbacks["time_callback"].times)
    slop = (self.history.history['val_loss'][0] -
            self.history.history['val_loss'][-1]) / computation_time 
    return {
      "slop": slop, 
      "mean_accuracy" : self.history.history['val_acc'][-1],
      "mean_loss": self.history.history['val_loss'][-1] 
    }



'''
def initial_acc(batch_size_continuous,
                lr_exp,
                momentum,
                layer_size_continuous,
                layer_count_continuous,
                tune_report_call_back=None):
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
  if tune_report_call_back == None:
    callbacks = [one_second_stop_callback]
  else:
    callbacks = [one_second_stop_callback, tune_report_call_back]
  history = model.fit_generator(
      xor_data_generator,
      steps_per_epoch=1,  # this is a virtual parameter
      epochs=10,
      validation_data=xor_validation_data,
      verbose=0,
      callbacks=callbacks)
  final_acc = history.history['val_acc'][-1]
  del model
  gc.collect()
  tf.keras.backend.clear_session()
  return final_acc
'''
class InitialAccuracyTrainable(XORTrainable):
  def _callbacks(self):
    tune_reporter_callback = TuneReporterCallback()
    one_second_stop_callback = OneSecondStopper()
    self.callbacks = {
      "one_second_stop_callback":one_second_stop_callback,
      "tune_reporter_callback":tune_reporter_callback
    }
    return self.callbacks


'''
def perfect_acc_time(batch_size_continuous, lr_exp, momentum,
                     layer_size_continuous, layer_count_continuous,
                     tune_report_call_back):
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
  if tune_report_call_back == None:
    callbacks = [stop_if_goal_reached, time_callback]
  else:
    callbacks = [stop_if_goal_reached, time_callback, tune_report_call_back]
  history = model.fit_generator(
      xor_data_generator,
      steps_per_epoch=1,
      epochs=10,
      validation_data=xor_validation_data,
      verbose=0,
      callbacks=callbacks)
  computation_time = np.sum(time_callback.times)
  epoch_usage = len(history.history['val_acc'])
  final_acc = history.history['val_acc'][-1]
  print("computation_time:", computation_time, "epoch size:", epoch_usage,
        "acc:", final_acc)
  del model
  gc.collect()
  tf.keras.backend.clear_session()
  if final_acc < 1.0:
    return -1. + final_acc
  else:
    return 1. / computation_time
'''
class IdealAccuracySpeedTrainable(XORTrainable):
  def _callbacks(self):
    time_callback = TimeHistory()
    tune_reporter_callback = TuneReporterCallback()
    stop_if_goal_reached = GoalReachedStopper()
    self.callbacks = {
      "stop_if_goal_reached":stop_if_goal_reached,
      "time_callback":time_callback,
      "tune_reporter_callback":tune_reporter_callback
    }
    return self.callbacks
  def _get_result(self):
    computation_time = np.sum(self.callbacks["time_callback"].times)
    final_acc = self.history.history['val_acc'][-1]
    if final_acc < 1.0:
      acc_feasibility_guided_speed = -1. + final_acc
    else:
      acc_feasibility_guided_speed = 1. / computation_time
    return {
      "acc_feasibility_guided_speed":acc_feasibility_guided_speed,
      "mean_accuracy" : self.history.history['val_acc'][-1],
      "mean_loss": self.history.history['val_loss'][-1] 
    }

'''
def memory_efficiency(batch_size_continuous,
                      lr_exp,
                      momentum,
                      layer_size_continuous,
                      layer_count_continuous,
                      tune_report_call_back=None):
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
  if tune_report_call_back == None:
    callbacks = [stop_if_goal_reached, time_callback]
  else:
    callbacks = [stop_if_goal_reached, time_callback, tune_report_call_back]
  history = model.fit_generator(
      xor_data_generator,
      steps_per_epoch=1,
      epochs=5,
      validation_data=xor_validation_data,
      verbose=0,
      callbacks=callbacks)
  memory_usage = get_model_memory_usage(batch_size, model)
  epoch_usage = len(history.history['val_acc'])
  final_acc = history.history['val_acc'][-1]
  print("memory_usage:", memory_usage, "epoch size:", epoch_usage, "acc:",
        final_acc)
  memory_efficiency = 1. / memory_usage

  del model
  gc.collect()
  tf.keras.backend.clear_session()
  if final_acc < 1.0 or epoch_usage > 3:
    return 0. - epoch_usage + final_acc
  else:
    return memory_efficiency
'''


class MemoryEfficiencyTrainable(XORTrainable):
  def _callbacks(self):
    time_callback = TimeHistory()
    tune_reporter_callback = TuneReporterCallback()
    stop_if_goal_reached = GoalReachedStopper()
    self.callbacks = {
      "stop_if_goal_reached":stop_if_goal_reached,
      "time_callback":time_callback,
      "tune_reporter_callback":tune_reporter_callback
    }
    return self.callbacks
  def _get_result(self):
    memory_usage = get_model_memory_usage(self.batch_size, self.model)
    epoch_usage = len(self.history.history['val_acc'])
    final_acc = self.history.history['val_acc'][-1]
    memory_efficiency = 1. / memory_usage
    if final_acc < 1.0 or epoch_usage > 3:
      speed_feasibility_guided_memory_efficiency =  0. - epoch_usage + final_acc
    else:
      speed_feasibility_guided_memory_efficiency = memory_efficiency
    return {
      "speed_feasibility_guided_memory_efficiency":speed_feasibility_guided_memory_efficiency,
      "mean_accuracy" : self.history.history['val_acc'][-1],
      "mean_loss": self.history.history['val_loss'][-1] 
    }
