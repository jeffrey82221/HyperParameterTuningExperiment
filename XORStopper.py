from tensorflow.keras.callbacks import Callback
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
