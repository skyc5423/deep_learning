import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, ReLU, BatchNormalization


class GANTrainer(object):

    def __init__(self):
        self.discriminator = self.Discriminator('discriminator')
        self.generator = self.Generator('generator')

    class Discriminator(tf.keras.Model):
        def __init__(self, name):
            super().__init__(name=name)

            self.dense_1 = Dense(64, activation=None, use_bias=True)
            # self.bn_1 = BatchNormalization()
            self.relu_1 = ReLU()
            self.dense_2 = Dense(64, activation=None, use_bias=True)
            # self.bn_2 = BatchNormalization()
            self.relu_2 = ReLU()
            self.dense_3 = Dense(64, activation=None, use_bias=True)
            # self.bn_3 = BatchNormalization()
            self.relu_3 = ReLU()
            self.dense_4 = Dense(64, activation=None, use_bias=True)
            # self.bn_4 = BatchNormalization()
            self.relu_4 = ReLU()
            self.out = Dense(1, activation='sigmoid', use_bias=True)

            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            )

        def __call__(self, input_batch, training=True):
            feature_1 = self.relu_1(self.dense_1(input_batch))
            # feature_1 = self.bn_1(feature_1, training=training)
            feature_2 = self.relu_2(self.dense_2(feature_1))
            # feature_2 = self.bn_2(feature_2, training=training)
            feature_3 = self.relu_3(self.dense_3(feature_2))
            # feature_3 = self.bn_3(feature_3, training=training)
            feature_4 = self.relu_4(self.dense_4(feature_3))
            # feature_4 = self.bn_4(feature_4, training=training)
            out = self.out(feature_4)
            return out

    class Generator(tf.keras.Model):
        def __init__(self, name):
            super().__init__(name=name)

            self.dense_1 = Dense(256, activation=None, use_bias=True)
            self.bn_1 = BatchNormalization()
            self.relu_1 = LeakyReLU()
            self.dense_2 = Dense(256, activation=None, use_bias=True)
            self.bn_2 = BatchNormalization()
            self.relu_2 = LeakyReLU()
            self.dense_3 = Dense(256, activation=None, use_bias=True)
            self.bn_3 = BatchNormalization()
            self.relu_3 = LeakyReLU()
            self.dense_4 = Dense(256, activation=None, use_bias=True)
            self.bn_4 = BatchNormalization()
            self.relu_4 = LeakyReLU()
            self.out = Dense(2, activation=None, use_bias=True)

            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
            )

        def __call__(self, input_batch, training=True):
            feature_1 = self.relu_1(self.dense_1(input_batch))
            feature_1 = self.bn_1(feature_1, training=training)
            feature_2 = self.relu_2(self.dense_2(feature_1))
            feature_2 = self.bn_2(feature_2, training=training)
            feature_3 = self.relu_3(self.dense_3(feature_2))
            feature_3 = self.bn_3(feature_3, training=training)
            feature_4 = self.relu_4(self.dense_4(feature_3))
            feature_4 = self.bn_4(feature_4, training=training)
            out = self.out(feature_4)
            return out

    @staticmethod
    def loss_discriminator(dis_real, dis_fake):
        loss_fake = - tf.math.log(1 - dis_fake + 1E-7)
        loss_real = - tf.math.log(dis_real + 1E-7)

        return tf.reduce_mean(loss_fake + loss_real)

    @staticmethod
    def loss_generator(dis_fake):
        loss_fake = - tf.math.log(dis_fake + 1E-7)

        return tf.reduce_mean(loss_fake)

    def train_discriminator(self, input_batch, input_label, input_noise):
        with tf.GradientTape() as tape:
            tape.watch(self.discriminator.trainable_variables)

            concat_input_noise = tf.concat([input_noise, input_label], axis=1)
            concat_input_batch = tf.concat([input_batch, input_label], axis=1)

            generated_image = self.generator(concat_input_noise)
            concat_generated_image = tf.concat([generated_image, input_label], axis=1)
            dis_real = self.discriminator(concat_input_batch)
            dis_fake = self.discriminator(concat_generated_image)

            loss_discriminator = self.loss_discriminator(dis_real, dis_fake)

        grads = tape.gradient(loss_discriminator, self.discriminator.trainable_variables)
        grads_vars = zip(grads, self.discriminator.trainable_variables)

        self.discriminator.optimizer.apply_gradients(grads_vars)
        return loss_discriminator

    def train_generator(self, input_label, input_noise):
        with tf.GradientTape() as tape:
            tape.watch(self.discriminator.trainable_variables)

            concat_input_noise = tf.concat([input_noise, input_label], axis=1)

            generated_image = self.generator(concat_input_noise)

            concat_generated_image = tf.concat([generated_image, input_label], axis=1)

            dis_fake = self.discriminator(concat_generated_image)

            loss_generator = self.loss_generator(dis_fake)

        grads = tape.gradient(loss_generator, self.generator.trainable_variables)
        grads_vars = zip(grads, self.generator.trainable_variables)

        self.generator.optimizer.apply_gradients(grads_vars)

        return loss_generator

    def train_(self, input_batch, input_label, input_noise):
        loss_generator = self.train_generator(input_label, input_noise)
        loss_discriminator = self.train_discriminator(input_batch, input_label, input_noise)

        return loss_generator, loss_discriminator
