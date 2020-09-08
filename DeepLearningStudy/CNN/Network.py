import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, AveragePooling2D, Dense


class Network(tf.keras.Model):

    def __init__(self, name):
        super().__init__(name=name)

        self.conv_1 = Conv2D(filters=16, kernel_size=5, strides=(2, 2), padding='same', dilation_rate=(1, 1), activation=None)
        self.relu_1 = ReLU()

        self.conv_2 = Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same', dilation_rate=(1, 1), activation=None)
        self.relu_2 = ReLU()

        self.conv_3 = Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', dilation_rate=(1, 1), activation=None)
        self.relu_3 = ReLU()

        self.conv_4 = Conv2D(filters=128, kernel_size=3, strides=(2, 2), padding='same', dilation_rate=(1, 1), activation=None)
        self.relu_4 = ReLU()

        self.pool_1 = AveragePooling2D(pool_size=(8, 8), strides=None, padding='valid')

        self.out = Dense(2, activation='softmax')

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            name='Adam'
        )

    def __call__(self, input_batch):

        feature_1 = self.relu_1(self.conv_1(input_batch))
        feature_2 = self.relu_2(self.conv_2(feature_1))
        feature_3 = self.relu_3(self.conv_3(feature_2))
        feature_4 = self.relu_4(self.conv_4(feature_3))
        feature_5 = tf.squeeze(self.pool_1(feature_4))
        out = self.out(feature_5)
        return out

    def loss_(self, pred, label):
        loss_tmp = tf.keras.losses.categorical_crossentropy(label, pred)
        return tf.reduce_mean(loss_tmp)

    # @tf.function
    def train_(self, input_batch, label_batch):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            out_batch = self(input_batch)
            step_loss = self.loss_(out_batch, label_batch)

        grads = tape.gradient(step_loss, self.trainable_variables)
        grads_vars = zip(grads, self.trainable_variables)

        self.optimizer.apply_gradients(grads_vars)
        return step_loss
