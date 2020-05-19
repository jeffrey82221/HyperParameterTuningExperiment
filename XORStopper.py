from tensorflow.keras.callbacks import Callback
import time
'''
from XORUtil import get_xor_generator
gen = get_xor_generator(8)
next(gen)
from XORUtil import get_xor_data
get_xor_data(16)
'''


class GoalReachedStopper(Callback):
  def on_epoch_end(self, epoch, logs=None):
    if logs.get('val_acc') == 1.0:
      self.model.stop_training = True
      print("Training Stop because Acc == 1.0")


class RandomAccuracyStopper(Callback):
  def __init__(self, patience=100):
    self.patience = patience
    self.count = 0

  def on_epoch_end(self, epoch, logs=None):
    if logs.get('val_acc') <= 0.5:
      self.count += 1
    else:
      self.count = 0
    if self.count >= self.patience:
      self.model.stop_training = True
      print("Training Stop because Acc == 0.5 for " +
            str(self.patience) + " times")


class OneSecondStopper(Callback):
  def on_train_begin(self, logs={}):
    self.times = []

  def on_epoch_begin(self, batch, logs={}):
    self.epoch_time_start = time.time()

  def on_epoch_end(self, batch, logs={}):
    self.times.append(time.time() - self.epoch_time_start)
    if sum(self.times) > 1.0:
      self.model.stop_training = True
