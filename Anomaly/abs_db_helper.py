import os
import tensorflow as tf
from abc import *
import numpy as np
import math
import matplotlib.pyplot as plt


class AbsDBHelper(metaclass=ABCMeta):
    def __init__(self, db_path, batch_size):
        self.db_path = db_path
        self.batch_size = batch_size

        x_train, y_train, x_test, y_test = self.load_data_list()
        self.train_ds, self.test_ds = self.make_random_batch_dataset(x_train, y_train, x_test, y_test)

    @abstractmethod
    def load_data_list(self):
        pass

    @abstractmethod
    def get_data(self, x_batch, y_batch):
        pass

    def make_random_batch_dataset(self, x_train, y_train, x_test, y_test, num_train=None):
        if num_train is None:
            num_train = x_train.shape[0]
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(num_train).batch(self.batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.batch_size)
        return train_ds, test_ds

