import tensorflow as tf
from tensorflow.keras.layers import Dense


class Network(tf.keras.Model):

    def __init__(self, name):
        super().__init__(name=name)

        self.dense_1 = Dense(128, activation='relu', use_bias=True)
        self.dense_2 = Dense(128, activation='relu', use_bias=True)
        self.dense_3 = Dense(128, activation='relu', use_bias=True)
        self.dense_4 = Dense(128, activation='relu', use_bias=True)
        self.out = Dense(4, activation='softmax', use_bias=True)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            name='Adam'
        )

    def __call__(self, input_batch):
        feature_1 = self.dense_1(input_batch)
        feature_2 = self.dense_2(feature_1)
        feature_3 = self.dense_3(feature_2)
        feature_4 = self.dense_4(feature_3)
        out = self.out(feature_4)
        return out

    def loss_(self, pred, label):
        loss_tmp = tf.keras.losses.categorical_crossentropy(label, pred)
        # return loss_tmp
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
