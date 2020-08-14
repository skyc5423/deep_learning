import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, AveragePooling2D, ReLU, MaxPooling2D, UpSampling2D


class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__(name='generator')

        self.fc_1 = Dense(128)
        self.bn_1 = BatchNormalization()
        self.relu_1 = LeakyReLU()

        self.fc_2 = Dense(128)
        self.bn_2 = BatchNormalization()
        self.relu_2 = LeakyReLU()

        self.fc_3 = Dense(128)
        self.bn_3 = BatchNormalization()
        self.relu_3 = LeakyReLU()

        self.out = Dense(2, activation='tanh')

    def __call__(self, input_tensor, training):
        feature_1 = self.relu_1(self.bn_1(self.fc_1(input_tensor)), training=training)
        feature_2 = self.relu_2(self.bn_2(self.fc_2(feature_1)), training=training)
        feature_3 = self.relu_3(self.bn_3(self.fc_3(feature_2)), training=training)
        out = self.out(feature_3)

        return out

    def gen_loss(self, gen_sample, discriminator):
        gen_score = discriminator(gen_sample, training=False)
        return tf.reduce_mean(-tf.math.log(gen_score + 1E-9))

    @tf.function
    def train(self, train_input, discriminator, optimizer):
        with tf.GradientTape() as gen_tape:
            gen_sample = self(train_input, training=True)
            gen_loss = self.gen_loss(gen_sample, discriminator)
            train_var = self.trainable_variables
            grad_of_generator = gen_tape.gradient(gen_loss, train_var)
            optimizer.apply_gradients(zip(grad_of_generator, train_var))
        return gen_loss

    @tf.function
    def test(self, train_input):
        gen_sample = self(train_input, training=False)
        return gen_sample


class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__(name='discriminator')

        self.fc_1 = Dense(128)
        self.relu_1 = LeakyReLU()

        self.fc_2 = Dense(128)
        self.relu_2 = LeakyReLU()

        self.fc_3 = Dense(128)
        self.relu_3 = LeakyReLU()

        self.out = Dense(1, activation='sigmoid')

    def __call__(self, input_tensor, training):
        feature_1 = self.relu_1(self.fc_1(input_tensor))
        feature_2 = self.relu_2(self.fc_2(feature_1))
        feature_3 = self.relu_3(self.fc_3(feature_2))
        out = self.out(feature_3)

        return out

    def dis_loss(self, score_real, score_fake):
        real_loss = tf.reduce_mean(-tf.math.log(score_real + 1E-9))
        fake_loss = tf.reduce_mean(-tf.math.log(1 - score_fake + 1E-9))
        return real_loss + fake_loss

    @tf.function
    def train(self, train_input_real, train_input_fake, optimizer):
        with tf.GradientTape() as gen_tape:
            dis_score_real = self(train_input_real, training=True)
            dis_score_fake = self(train_input_fake, training=True)
            dis_loss = self.dis_loss(dis_score_real, dis_score_fake)
            train_var = self.trainable_variables
            grad_of_discriminator = gen_tape.gradient(dis_loss, train_var)
            optimizer.apply_gradients(zip(grad_of_discriminator, train_var))
        return dis_loss

    @tf.function
    def test(self, train_input):
        dis_score = self(train_input, training=False)
        return dis_score
