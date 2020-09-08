import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


def gradient_descent(with_square=False, dir='./result'):
    input_x = tf.Variable(5.0)
    ground_truth_y = tf.constant(-3.)
    learning_rate = 0.1

    if with_square:
        input_x2 = tf.Variable(5.0)

    for iter_idx in range(200):
        with tf.GradientTape() as tape:
            tape.watch(input_x)
            distance = tf.abs(input_x - ground_truth_y)
            loss = distance

        ##################### Visualize #####################
        fig, ax = plt.subplots(1, 1)
        ax.scatter(ground_truth_y, 0, color='blue')

        x = np.linspace(-5, 10, 100)
        distance_total = np.abs(x - (-2))
        ax.plot(x, distance_total, color='black')

        ax.scatter(input_x, distance, color='red')
        ###############################################################

        grads = tape.gradient(loss, input_x)

        input_x = tf.subtract(input_x, learning_rate * grads)

        if with_square:
            with tf.GradientTape() as tape:
                tape.watch(input_x2)
                distance2 = tf.square(input_x2 - ground_truth_y)
                loss2 = distance2

            x2 = np.linspace(-5, 10, 100)
            distance_total2 = np.square(x2 - (-2))
            ax.plot(x2, distance_total2, color='green')

            ax.scatter(input_x2, distance2, color='green')
            # (x - (-) 2)^2
            # 2x + 4 = 14
            grads2 = tape.gradient(loss2, input_x2)

            input_x2 = tf.subtract(input_x2, learning_rate * grads2)

        if not os.path.exists(dir):
            os.mkdir(dir)
        fig.savefig(os.path.join(dir, 'result_%d.png' % iter_idx))
        plt.close(fig)
