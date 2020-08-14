import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, Reshape, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, AveragePooling2D, ReLU, MaxPooling2D, UpSampling2D
import tensorflow_addons as tfa


class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__(name='generator')

        self.conv_1 = Conv2D(64, 7, strides=(2, 2), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.ins_norm_1 = tfa.layers.InstanceNormalization(axis=-1)
        self.relu_1 = LeakyReLU()

        self.conv_2 = Conv2D(128, 3, strides=(2, 2), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.ins_norm_2 = tfa.layers.InstanceNormalization(axis=-1)
        self.relu_2 = LeakyReLU()

        self.conv_3 = Conv2D(256, 3, strides=(2, 2), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.ins_norm_3 = tfa.layers.InstanceNormalization(axis=-1)
        self.relu_3 = ReLU()

        self.conv_4 = Conv2D(512, 3, strides=(2, 2), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.ins_norm_4 = tfa.layers.InstanceNormalization(axis=-1)
        self.relu_4 = ReLU()

        self.conv_5 = Conv2D(512, 3, strides=(2, 2), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.ins_norm_5 = tfa.layers.InstanceNormalization(axis=-1)
        self.relu_5 = ReLU()


    def __call__(self, input_tensor, training):
        feature_1 = self.relu_1(self.bn_1(self.convt_1(input_tensor)), training=training)
        feature_2 = self.relu_2(self.bn_2(self.convt_2(feature_1)), training=training)
        feature_3 = self.convt_3(feature_2)
        out = feature_3

        return out

    def gen_loss(self, gen_sample, discriminator):
        gen_score = discriminator(gen_sample, training=False)
        return tf.reduce_mean(-tf.math.log(gen_score + 1E-9))

    @tf.function
    def train(self, random_z, discriminator, optimizer):
        with tf.GradientTape() as gen_tape:
            gen_sample = self(random_z, training=True)
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

        self.conv_1 = Conv2D(32, 3, strides=(1, 1), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.relu_1 = LeakyReLU()

        self.conv_2 = Conv2D(64, 3, strides=(2, 2), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.ins_norm_2 = tfa.layers.InstanceNormalization(axis=-1)
        self.relu_2 = LeakyReLU()

        self.conv_3 = Conv2D(128, 3, strides=(2, 2), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.ins_norm_3 = tfa.layers.InstanceNormalization(axis=-1)
        self.relu_3 = LeakyReLU()

        self.flat = Flatten()

        self.fc_1 = Dense(1, activation='sigmoid')

    def __call__(self, input_tensor, training):
        feature_1 = self.relu_1(self.conv_1(input_tensor))
        feature_2 = self.relu_2(self.ins_norm_2(self.conv_2(feature_1)))
        feature_3 = self.relu_3(self.ins_norm_3(self.conv_3(feature_2)))
        out = self.fc_1(self.flat(feature_3))

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
