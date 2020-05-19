import random
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
'''
from XOR_Util import get_xor_generator
gen = get_xor_generator(8)
next(gen)
from XOR_Util import get_xor_data
get_xor_data(16)
'''


class get_xor_generator(Sequence):
  def __init__(self, batch_size):
    assert batch_size % 4 == 0
    # make sure the number of the following instances are the same:
    # (0,0), (0,1), (1,0), (1,1).
    self.batch_size = batch_size

  def __len__(self):
    return 100

  @property
  def shape(self):
    return (2, )

  def __getitem__(self, item):
    return self.__next__()

  def __next__(self):
    zero_zero_instances = np.zeros((int(self.batch_size / 4), 2))
    one_one_instances = np.ones((int(self.batch_size / 4), 2))
    one_zero_instances = np.vstack(
        [zero_zero_instances[:, 0], one_one_instances[:, 0]]).T
    zero_one_instances = np.vstack(
        [one_one_instances[:, 0], zero_zero_instances[:, 0]]).T
    instances = np.vstack([
        zero_zero_instances, one_one_instances, one_zero_instances,
        zero_one_instances
    ])
    target = np.logical_xor(instances[:, 0], instances[:, 1])
    return instances, target.astype(float)


def get_xor_data(data_size):
  gen = get_xor_generator(data_size)
  return next(gen)
