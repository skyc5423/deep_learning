import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Flatten, Dense, Reshape, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, AveragePooling2D, ReLU, MaxPooling2D, UpSampling2D, Concatenate
import tensorflow_addons as tfa
from diff_aug import *


class AutoEncoderTrainer(object):
    def __init__(self, generator):
        self.generator = generator
        self.lr_generator = 0.002
        self.lr_encoder = 0.002

        self.generator_optimizer = tf.keras.optimizers.Adam(self.lr_generator, beta_1=0.5)

    @tf.function
    def train(self, input_img, z_loss=True):
        with tf.GradientTape(persistent=True) as tape:
            recon_img = self.generator(input_img, training=True)

            ae_loss = tf.reduce_mean(tf.square(recon_img - input_img))
            # ae_loss = tf.reduce_mean(tf.abs(recon_img - input_img))

        tv = self.generator.trainable_variables
        generator_ae_gradients = tape.gradient(ae_loss, tv)

        self.generator_optimizer.apply_gradients(zip(generator_ae_gradients, tv))

        return ae_loss

    @tf.function
    def test(self, input_img):
        recon_img = self.generator(input_img, training=False)

        return recon_img


class CycleGanTrainer(object):
    def __init__(self, generator_f, generator_g, discriminator_f, discriminator_g):
        self.generator_f_to_g = generator_f
        self.generator_g_to_f = generator_g
        self.discriminator_f = discriminator_f
        self.discriminator_g = discriminator_g
        self.lr_generator = 0.002
        self.lr_discriminator = 0.002
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generator_optimizer_f = tf.keras.optimizers.Adam(self.lr_generator, beta_1=0.5)
        self.generator_optimizer_g = tf.keras.optimizers.Adam(self.lr_generator, beta_1=0.5)
        self.discriminator_optimizer_f = tf.keras.optimizers.Adam(self.lr_discriminator, beta_1=0.5)
        self.discriminator_optimizer_g = tf.keras.optimizers.Adam(self.lr_discriminator, beta_1=0.5)

    @staticmethod
    def rand_aug(img_data, p=0.8):
        prob = np.random.uniform(0, 1, 5)

        if prob[0] <= p:
            img_data = rand_brightness(img_data)

        if prob[1] <= p:
            img_data = rand_contrast(img_data)

        if prob[2] <= p:
            img_data = rand_cutout(img_data)

        if prob[3] <= p:
            img_data = rand_saturation(img_data)

        if prob[4] <= p:
            img_data = rand_translation(img_data)

        return img_data

    @tf.function
    def train(self, real_img_f, real_img_g, z_loss=True):
        with tf.GradientTape(persistent=True) as tape:
            recon_img_g = self.generator_f_to_g(real_img_f, training=True)
            cycle_img_f = self.generator_g_to_f(recon_img_g, training=True)
            identity_img_f = self.generator_g_to_f(real_img_f, training=True)

            recon_img_f = self.generator_g_to_f(real_img_g, training=True)
            cycle_img_g = self.generator_f_to_g(recon_img_f, training=True)
            identity_img_g = self.generator_f_to_g(real_img_g, training=True)

            aug_real_img_f = self.rand_aug(real_img_f)
            aug_real_img_g = self.rand_aug(real_img_g)
            aug_recon_img_f = self.rand_aug(recon_img_f)
            aug_recon_img_g = self.rand_aug(recon_img_g)
            score_real_f = self.discriminator_f(aug_real_img_f, training=True)
            score_real_g = self.discriminator_g(aug_real_img_g, training=True)
            score_recon_f = self.discriminator_f(aug_recon_img_f, training=True)
            score_recon_g = self.discriminator_g(aug_recon_img_g, training=True)

            gan_loss_f = self.cross_entropy(tf.ones_like(score_recon_f), score_recon_f)
            gan_loss_g = self.cross_entropy(tf.ones_like(score_recon_g), score_recon_g)

            cycle_loss_f = tf.reduce_mean(tf.abs(real_img_f - cycle_img_f))
            cycle_loss_g = tf.reduce_mean(tf.abs(real_img_g - cycle_img_g))
            cycle_loss = 30 * (cycle_loss_f + cycle_loss_g)

            identity_loss_f = 5 * tf.reduce_mean(tf.abs(real_img_f - identity_img_f))
            identity_loss_g = 5 * tf.reduce_mean(tf.abs(real_img_g - identity_img_g))

            gen_loss_f = gan_loss_f + cycle_loss + identity_loss_f
            gen_loss_g = gan_loss_g + cycle_loss + identity_loss_g

            dis_real_loss_f = self.cross_entropy(tf.ones_like(score_real_f), score_real_f)
            dis_fake_loss_f = self.cross_entropy(tf.zeros_like(score_recon_f), score_recon_f)
            dis_real_loss_g = self.cross_entropy(tf.ones_like(score_real_g), score_real_g)
            dis_fake_loss_g = self.cross_entropy(tf.zeros_like(score_recon_g), score_recon_g)
            dis_loss_f = 1. * (dis_real_loss_f + dis_fake_loss_f)
            dis_loss_g = 1. * (dis_real_loss_g + dis_fake_loss_g)

        # Calculate the gradients for generator and discriminator
        generator_f_gradients = tape.gradient(gen_loss_f,
                                              self.generator_f_to_g.trainable_variables)
        generator_g_gradients = tape.gradient(gen_loss_g,
                                              self.generator_g_to_f.trainable_variables)

        discriminator_f_gradients = tape.gradient(dis_loss_f,
                                                  self.discriminator_f.trainable_variables)
        discriminator_g_gradients = tape.gradient(dis_loss_g,
                                                  self.discriminator_g.trainable_variables)

        # Apply the gradients to the optimizer
        self.generator_optimizer_g.apply_gradients(zip(generator_g_gradients,
                                                       self.generator_g_to_f.trainable_variables))

        self.generator_optimizer_f.apply_gradients(zip(generator_f_gradients,
                                                       self.generator_f_to_g.trainable_variables))

        self.discriminator_optimizer_f.apply_gradients(zip(discriminator_f_gradients,
                                                           self.discriminator_f.trainable_variables))

        self.discriminator_optimizer_g.apply_gradients(zip(discriminator_g_gradients,
                                                           self.discriminator_g.trainable_variables))

        return gen_loss_f, gen_loss_g, dis_loss_f, dis_loss_g


class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__(name='generator')
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

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
        self.relu_3 = LeakyReLU()

        self.conv_4 = Conv2D(512, 3, strides=(2, 2), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.ins_norm_4 = tfa.layers.InstanceNormalization(axis=-1)
        self.relu_4 = LeakyReLU()

        self.conv_5 = Conv2D(512, 3, strides=(1, 1), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.ins_norm_5 = tfa.layers.InstanceNormalization(axis=-1)
        self.relu_5 = LeakyReLU()

        self.conv_6 = Conv2D(256, 3, strides=(1, 1), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.ins_norm_6 = tfa.layers.InstanceNormalization(axis=-1)
        self.relu_6 = LeakyReLU()

        self.conv_7 = Conv2D(128, 3, strides=(1, 1), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.ins_norm_7 = tfa.layers.InstanceNormalization(axis=-1)
        self.relu_7 = LeakyReLU()

        self.conv_8 = Conv2D(64, 3, strides=(1, 1), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.ins_norm_8 = tfa.layers.InstanceNormalization(axis=-1)
        self.relu_8 = LeakyReLU()

        self.conv_9 = Conv2D(3, 5, strides=(1, 1), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation='tanh', use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.ins_norm_9 = tfa.layers.InstanceNormalization(axis=-1)
        self.relu_9 = LeakyReLU()

    def __call__(self, input_tensor, training):
        feature_1 = self.relu_1(self.conv_1(input_tensor))
        feature_2 = self.relu_2(self.ins_norm_2(self.conv_2(feature_1)))
        feature_3 = self.relu_3(self.ins_norm_3(self.conv_3(feature_2)))
        feature_4 = self.relu_4(self.ins_norm_4(self.conv_4(feature_3)))

        feature_5 = UpSampling2D()(feature_4)
        feature_5 = Concatenate()([feature_5, feature_3])
        feature_5 = self.relu_5(self.ins_norm_5(self.conv_5(feature_5)))
        feature_6 = UpSampling2D()(feature_5)
        feature_6 = Concatenate()([feature_6, feature_2])
        feature_6 = self.relu_6(self.ins_norm_6(self.conv_6(feature_6)))
        feature_7 = UpSampling2D()(feature_6)
        feature_7 = Concatenate()([feature_7, feature_1])
        feature_7 = self.relu_7(self.ins_norm_7(self.conv_7(feature_7)))
        feature_8 = UpSampling2D()(feature_7)
        # feature_8 = Concatenate()([feature_8, input_tensor])
        feature_8 = self.relu_8(self.ins_norm_8(self.conv_8(feature_8)))

        feature_9 = self.conv_9(feature_8)

        out = feature_9

        return out

    def gen_loss(self, real_image, cycled_image, same_image, discriminator):
        gen_score = discriminator(cycled_image, training=False)
        gan_loss = self.cross_entropy(tf.ones_like(gen_score), gen_score)

        cycle_loss = 60 * tf.reduce_mean(tf.abs(real_image - cycled_image))
        identity_loss = 5 * tf.reduce_mean(tf.abs(real_image - same_image))

        return

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

        self.conv_1 = Conv2D(64, 5, strides=(4, 4), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
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
        self.relu_3 = LeakyReLU()

        self.conv_4 = Conv2D(512, 3, strides=(2, 2), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.ins_norm_4 = tfa.layers.InstanceNormalization(axis=-1)
        self.relu_4 = LeakyReLU()

        self.conv_5 = Conv2D(512, 3, strides=(2, 2), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.ins_norm_5 = tfa.layers.InstanceNormalization(axis=-1)
        self.relu_5 = LeakyReLU()

        self.flat = Flatten()

        self.fc_1 = Dense(1, activation='sigmoid')

    def __call__(self, input_tensor, training):
        feature_1 = self.relu_1(self.conv_1(input_tensor))
        feature_2 = self.relu_2(self.ins_norm_2(self.conv_2(feature_1)))
        feature_3 = self.relu_3(self.ins_norm_3(self.conv_3(feature_2)))
        feature_4 = self.relu_4(self.ins_norm_4(self.conv_4(feature_3)))
        feature_5 = self.relu_5(self.ins_norm_5(self.conv_5(feature_4)))
        feature_6 = self.fc_1(self.flat(feature_5))

        return feature_6

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
