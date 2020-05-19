from tensorflow.keras.callbacks import Callback
'''
from XORUtil import get_xor_generator
gen = get_xor_generator(8)
next(gen)
from XORUtil import get_xor_data
get_xor_data(16)
'''


class GoalReachedStopper(Callback):
  def on_train_epoch_end(self, epoch, logs=None):
    if logs.get('val_acc') == 1.0:
      self.model.stop_training = True
      print("Training Stop because Acc == 1.0")
