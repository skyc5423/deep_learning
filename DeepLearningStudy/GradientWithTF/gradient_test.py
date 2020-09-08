import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def gradient_test():

    a = tf.constant(1., shape=[100])
    b = tf.constant(- 1., shape=[100])
    c = tf.constant(- 5., shape=[100])
    d = tf.constant(3., shape=[100])

    x = tf.constant(np.linspace(-5, 5, 100), dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = tf.multiply(a, x ** 3) + tf.multiply(b, x ** 2) + tf.multiply(c, x) + d
        # y = x^3 - x^2 - 5x + 3
        # y'= 3x^2 - 2x - 5

    grads = tape.gradient(y, x)

    y_numpy = y.numpy()
    grads_numpy = grads.numpy()

    plt.plot(x.numpy(), y_numpy, color='black')
    plt.plot(x.numpy(), grads_numpy, color='red')

    plt.show()
