import os
import tensorflow as tf
from abc import *


class AbsDBHelper(metaclass=ABCMeta):
    def __init__(self, db_path, batch_size):
        self.db_path = db_path
        self.batch_size = batch_size

        x_train = self.load_data_list()
        self.train_ds = self.make_random_batch_dataset(x_train)

    @abstractmethod
    def load_data_list(self):
        pass

    @abstractmethod
    def get_data(self, x_batch):
        pass

    def make_random_batch_dataset(self, x_train, num_train=None):
        if num_train is None:
            num_train = x_train.shape[0]
        train_ds = tf.data.Dataset.from_tensor_slices((x_train)).shuffle(num_train).batch(self.batch_size)
        return train_ds
