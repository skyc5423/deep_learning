import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, LeakyReLU, ReLU, BatchNormalization, Conv2D, AveragePooling2D, UpSampling2D, Concatenate, Flatten


class GANTrainer(object):

    def __init__(self, batch_size):
        self.discriminator_x = self.Discriminator('discriminator_x')
        self.generator_x_to_y = self.Generator('generator_x_to_y')
        self.discriminator_y = self.Discriminator('discriminator_y')
        self.generator_y_to_x = self.Generator('generator_y_to_x')
        self.batch_size = batch_size
        self.optimizer_g = tf.keras.optimizers.Adam(
                learning_rate=0.00005, beta_1=0.5, epsilon=1e-07, amsgrad=False,
            )
        self.optimizer_d = tf.keras.optimizers.Adam(
                learning_rate=0.00005, beta_1=0.5, epsilon=1e-07, amsgrad=False,
            )

    class Discriminator(tf.keras.Model):
        def __init__(self, name):
            super().__init__(name=name)

            self.conv_1 = Conv2D(filters=8, kernel_size=5, strides=(2, 2), padding='valid', dilation_rate=(1, 1), activation=None)
            self.ins_norm_1 = tfa.layers.InstanceNormalization(axis=-1)
            self.relu_1 = LeakyReLU()

            self.conv_2 = Conv2D(filters=16, kernel_size=3, strides=(2, 2), padding='valid', dilation_rate=(1, 1), activation=None)
            self.ins_norm_2 = tfa.layers.InstanceNormalization(axis=-1)
            self.relu_2 = LeakyReLU()

            self.conv_3 = Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='valid', dilation_rate=(1, 1), activation=None)
            self.ins_norm_3 = tfa.layers.InstanceNormalization(axis=-1)
            self.relu_3 = LeakyReLU()

            self.conv_4 = Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='valid', dilation_rate=(1, 1), activation=None)
            self.ins_norm_4 = tfa.layers.InstanceNormalization(axis=-1)
            self.relu_4 = LeakyReLU()

            self.conv_5 = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid', dilation_rate=(1, 1), activation=None)
            self.relu_5 = LeakyReLU()

            self.flat = Flatten()

            # self.pool_1 = AveragePooling2D(pool_size=(8, 8), strides=None, padding='valid')

            self.fc_1 = Dense(256, activation='relu')

            self.out = Dense(1, activation='sigmoid')

            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.0002, beta_1=0.5, epsilon=1e-07, amsgrad=False,
            )

        def __call__(self, input_batch, training=True):
            feature_1 = self.relu_1(self.ins_norm_1(self.conv_1(input_batch)))
            feature_2 = self.relu_2(self.ins_norm_2(self.conv_2(feature_1)))
            feature_3 = self.relu_3(self.ins_norm_3(self.conv_3(feature_2)))
            feature_4 = self.relu_4(self.ins_norm_4(self.conv_4(feature_3)))
            feature_5 = self.relu_5(self.conv_5(feature_4))
            feature_6 = self.fc_1(self.flat(feature_5))
            # feature_5 = tf.squeeze(self.pool_1(feature_4))
            out = self.out(feature_6)
            return out

    class Generator(tf.keras.Model):
        def __init__(self, name):
            super().__init__(name=name)

            self.conv_1 = Conv2D(16, 3, strides=(2, 2), padding='same', data_format=None,
                                 dilation_rate=(1, 1), activation=None, use_bias=True,
                                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)
            self.ins_norm_1 = tfa.layers.InstanceNormalization(axis=-1)
            self.relu_1 = LeakyReLU()

            self.conv_2 = Conv2D(32, 3, strides=(2, 2), padding='same', data_format=None,
                                 dilation_rate=(1, 1), activation=None, use_bias=True,
                                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)
            self.ins_norm_2 = tfa.layers.InstanceNormalization(axis=-1)
            self.relu_2 = LeakyReLU()

            self.conv_3 = Conv2D(64, 3, strides=(2, 2), padding='same', data_format=None,
                                 dilation_rate=(1, 1), activation=None, use_bias=True,
                                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)
            self.ins_norm_3 = tfa.layers.InstanceNormalization(axis=-1)
            self.relu_3 = LeakyReLU()

            self.conv_4 = Conv2D(128, 3, strides=(2, 2), padding='same', data_format=None,
                                 dilation_rate=(1, 1), activation=None, use_bias=True,
                                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)
            self.ins_norm_4 = tfa.layers.InstanceNormalization(axis=-1)
            self.relu_4 = LeakyReLU()

            self.conv_5 = Conv2D(128, 3, strides=(1, 1), padding='same', data_format=None,
                                 dilation_rate=(1, 1), activation=None, use_bias=True,
                                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)
            self.ins_norm_5 = tfa.layers.InstanceNormalization(axis=-1)
            self.relu_5 = LeakyReLU()

            self.conv_6 = Conv2D(64, 3, strides=(1, 1), padding='same', data_format=None,
                                 dilation_rate=(1, 1), activation=None, use_bias=True,
                                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)
            self.ins_norm_6 = tfa.layers.InstanceNormalization(axis=-1)
            self.relu_6 = LeakyReLU()

            self.conv_7 = Conv2D(32, 3, strides=(1, 1), padding='same', data_format=None,
                                 dilation_rate=(1, 1), activation=None, use_bias=True,
                                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)
            self.ins_norm_7 = tfa.layers.InstanceNormalization(axis=-1)
            self.relu_7 = LeakyReLU()

            self.conv_8 = Conv2D(16, 3, strides=(1, 1), padding='same', data_format=None,
                                 dilation_rate=(1, 1), activation=None, use_bias=True,
                                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)
            self.ins_norm_8 = tfa.layers.InstanceNormalization(axis=-1)
            self.relu_8 = LeakyReLU()

            self.conv_9 = Conv2D(3, 3, strides=(1, 1), padding='same', data_format=None,
                                 dilation_rate=(1, 1), activation='tanh', use_bias=True,
                                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                                 kernel_constraint=None, bias_constraint=None)
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.0002, beta_1=0.5, epsilon=1e-07, amsgrad=False,
            )

        def __call__(self, input_batch, training=True):
            feature_1 = self.relu_1(self.conv_1(input_batch))
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
            feature_8 = self.relu_8(self.ins_norm_8(self.conv_8(feature_8)))

            feature_9 = self.conv_9(feature_8)

            out = feature_9

            return out

    @staticmethod
    def loss_discriminator(dis_input_real, dis_generated_real):
        # loss_input_fake = - tf.reduce_sum(tf.math.log(1 - dis_input_fake + 1E-7))
        # loss_cycle_fake = - tf.reduce_sum(tf.math.log(1 - dis_cycle_fake + 1E-7))
        # loss_identity_fake = - tf.reduce_sum(tf.math.log(1 - dis_identity_fake + 1E-7))
        # loss_generated_fake = - tf.reduce_sum(tf.math.log(1 - dis_generated_fake + 1E-7))

        loss_input_real = - tf.reduce_sum(tf.math.log(dis_input_real + 1E-7))
        # loss_cycle_real = - tf.reduce_sum(tf.math.log(dis_cycle_real + 1E-7))
        # loss_identity_real = - tf.reduce_sum(tf.math.log(dis_identity_real + 1E-7))
        loss_generated_real = - tf.reduce_sum(tf.math.log(1 - dis_generated_real + 1E-7))

        return 2 * loss_input_real + loss_generated_real

    @staticmethod
    def loss_generator_dis(dis_generated_fake):
        # loss_cycle_real = - tf.reduce_sum(tf.math.log(1 - dis_cycle_real + 1E-7))
        # loss_identity_real = - tf.reduce_sum(tf.math.log(1 - dis_identity_real + 1E-7))
        # loss_generated_real = - tf.reduce_sum(tf.math.log(1 - dis_generated_real + 1E-7))

        # loss_cycle_fake = - tf.reduce_sum(tf.math.log(dis_cycle_fake + 1E-7))
        # loss_identity_fake = - tf.reduce_sum(tf.math.log(dis_identity_fake + 1E-7))
        loss_generated_fake = - tf.reduce_sum(tf.math.log(dis_generated_fake + 1E-7))

        # return loss_cycle_real + loss_identity_real + loss_generated_real + loss_cycle_fake + loss_identity_fake + loss_generated_fake
        return loss_generated_fake

    @staticmethod
    def loss_generator_cycle(input_image, cycle_image):
        return tf.reduce_mean(tf.reduce_sum(tf.abs(input_image - cycle_image), axis=0))

    def train_discriminator(self, input_x, input_y):
        with tf.GradientTape() as tape_x:
            tape_x.watch(self.discriminator_x.trainable_variables + self.discriminator_y.trainable_variables)

            generated_image_y = self.generator_x_to_y(input_x)
            # cycled_image_x = self.generator_y_to_x(generated_image_y)
            # identity_image_x = self.generator_y_to_x(input_x)

            generated_image_x = self.generator_y_to_x(input_y)
            # cycled_image_y = self.generator_x_to_y(generated_image_x)
            # identity_image_y = self.generator_x_to_y(input_y)

            dis_x_input_x = self.discriminator_x(input_x)
            # dis_x_cycle_x = self.discriminator_x(cycled_image_x)
            dis_x_generated_x = self.discriminator_x(generated_image_x)

            # dis_x_input_y = self.discriminator_x(input_y)
            # dis_x_cycle_y = self.discriminator_x(cycled_image_y)
            # dis_x_generated_y = self.discriminator_x(generated_image_y)

            # dis_y_input_x = self.discriminator_y(input_x)
            # dis_y_cycle_x = self.discriminator_y(cycled_image_x)
            # dis_y_identity_x = self.discriminator_y(identity_image_x)
            # dis_y_generated_x = self.discriminator_y(generated_image_x)

            dis_y_input_y = self.discriminator_y(input_y)
            # dis_y_cycle_y = self.discriminator_y(cycled_image_y)
            # dis_y_identity_y = self.discriminator_y(identity_image_y)
            dis_y_generated_y = self.discriminator_y(generated_image_y)

            # loss_discriminator_x = self.loss_discriminator(dis_x_input_x, dis_x_cycle_x, dis_x_identity_x, dis_x_generated_x,
            #                                                dis_x_input_y, dis_x_cycle_y, dis_x_identity_y, dis_x_generated_y)

            loss_discriminator_x = self.loss_discriminator(dis_x_input_x, dis_x_generated_x) / self.batch_size

            loss_discriminator_y = self.loss_discriminator(dis_y_input_y, dis_y_generated_y) / self.batch_size

            loss_discriminator = loss_discriminator_x + loss_discriminator_y

        grads_x = tape_x.gradient(loss_discriminator, self.discriminator_x.trainable_variables + self.discriminator_y.trainable_variables)
        grads_vars_x = zip(grads_x, self.discriminator_x.trainable_variables + self.discriminator_y.trainable_variables)
        # grads_vars_y = zip(grads_x, self.discriminator_y.trainable_variables)

        self.optimizer_d.apply_gradients(grads_vars_x)
        # self.discriminator_y.optimizer.apply_gradients(grads_vars_y)

        # with tf.GradientTape() as tape_y:
        #     tape_y.watch(self.discriminator_y.trainable_variables)
        #
        #     generated_image_y = self.generator_x_to_y(input_x)
        #     cycled_image_x = self.generator_y_to_x(generated_image_y)
        #     identity_image_x = self.generator_y_to_x(input_x)
        #
        #     generated_image_x = self.generator_y_to_x(input_y)
        #     cycled_image_y = self.generator_x_to_y(generated_image_x)
        #     identity_image_y = self.generator_x_to_y(input_y)
        #
        #
        #     loss_discriminator_y = self.loss_discriminator(dis_y_input_y, dis_y_cycle_y, dis_y_identity_y, dis_y_generated_y,
        #                                                    dis_y_input_x, dis_y_cycle_x, dis_y_identity_x, dis_y_generated_x)
        #
        # grads_y = tape_y.gradient(loss_discriminator_y, self.discriminator_y.trainable_variables)
        # grads_vars_y = zip(grads_y, self.discriminator_y.trainable_variables)
        #
        # self.discriminator_y.optimizer.apply_gradients(grads_vars_y)

        return loss_discriminator_x, loss_discriminator_y

    def train_generator(self, input_x, input_y):
        with tf.GradientTape() as tape_x_to_y:
            tape_x_to_y.watch(self.generator_x_to_y.trainable_variables + self.generator_y_to_x.trainable_variables)

            generated_image_y = self.generator_x_to_y(input_x)
            cycled_image_x = self.generator_y_to_x(generated_image_y)
            identity_image_x = self.generator_y_to_x(input_x)

            generated_image_x = self.generator_y_to_x(input_y)
            cycled_image_y = self.generator_x_to_y(generated_image_x)
            identity_image_y = self.generator_x_to_y(input_y)

            # dis_x_cycle_x = self.discriminator_x(cycled_image_x)
            # dis_x_identity_x = self.discriminator_x(identity_image_x)
            dis_x_generated_x = self.discriminator_x(generated_image_x)

            # dis_x_cycle_y = self.discriminator_x(cycled_image_y)
            # dis_x_identity_y = self.discriminator_x(identity_image_y)
            # dis_x_generated_y = self.discriminator_x(generated_image_y)

            # dis_y_cycle_x = self.discriminator_y(cycled_image_x)
            # dis_y_identity_x = self.discriminator_y(identity_image_x)
            # dis_y_generated_x = self.discriminator_y(generated_image_x)

            # dis_y_cycle_y = self.discriminator_y(cycled_image_y)
            # dis_y_identity_y = self.discriminator_y(identity_image_y)
            dis_y_generated_y = self.discriminator_y(generated_image_y)

            loss_gen_dis_y_to_x = self.loss_generator_dis(dis_x_generated_x) / self.batch_size
            loss_gen_dis_x_to_y = self.loss_generator_dis(dis_y_generated_y) / self.batch_size
            loss_gen_cycle_x = 10 * self.loss_generator_cycle(input_x, cycled_image_x)
            loss_gen_cycle_y = 10 * self.loss_generator_cycle(input_y, cycled_image_y)
            loss_gen_ident_x = 0 * self.loss_generator_cycle(input_x, identity_image_x)
            loss_gen_ident_y = 0 * self.loss_generator_cycle(input_y, identity_image_y)
            loss_gen_cycle = (loss_gen_cycle_x + loss_gen_cycle_y) / self.batch_size
            loss_gen_ident = (loss_gen_ident_x + loss_gen_ident_y) / self.batch_size
            loss_total = loss_gen_dis_y_to_x + loss_gen_dis_x_to_y + loss_gen_cycle + loss_gen_ident

            # loss_gen_dis_y_to_x = self.loss_generator_dis(dis_x_cycle_x, dis_x_generated_x, dis_y_cycle_x, dis_y_generated_x) / self.batch_size
            # loss_gen_dis_x_to_y = self.loss_generator_dis(dis_y_cycle_y, dis_y_generated_y, dis_x_cycle_y, dis_x_generated_y) / self.batch_size
            # loss_gen_cycle_x = 20 * self.loss_generator_cycle(input_x, cycled_image_x)
            # loss_gen_cycle_y = 20 * self.loss_generator_cycle(input_y, cycled_image_y)
            # loss_gen_identity_x = self.loss_generator_cycle(input_x, identity_image_x)
            # loss_gen_identity_y = self.loss_generator_cycle(input_y, identity_image_y)
            # loss_gen_cycle = (loss_gen_cycle_x + loss_gen_cycle_y) / self.batch_size
            # loss_gen_identity = (loss_gen_identity_x + loss_gen_identity_y) / self.batch_size
            # loss_total = loss_gen_dis_y_to_x + loss_gen_dis_x_to_y + loss_gen_cycle

        grads_x_to_y = tape_x_to_y.gradient(loss_total, self.generator_x_to_y.trainable_variables + self.generator_y_to_x.trainable_variables)
        grads_vars_x_to_y = zip(grads_x_to_y, self.generator_x_to_y.trainable_variables + self.generator_y_to_x.trainable_variables)
        # grads_vars_y_to_x = zip(grads_x_to_y, self.generator_y_to_x.trainable_variables)

        self.optimizer_g.apply_gradients(grads_vars_x_to_y)

        # with tf.GradientTape() as tape_y_to_x:
        #     tape_y_to_x.watch(self.generator_y_to_x.trainable_variables)
        #
        #     generated_image_y = self.generator_x_to_y(input_x)
        #     cycled_image_x = self.generator_y_to_x(generated_image_y)
        #     identity_image_x = self.generator_y_to_x(input_x)
        #
        #     generated_image_x = self.generator_y_to_x(input_y)
        #     cycled_image_y = self.generator_x_to_y(generated_image_x)
        #     identity_image_y = self.generator_x_to_y(input_y)
        #
        #     dis_x_cycle_x = self.discriminator_x(cycled_image_x)
        #     dis_x_identity_x = self.discriminator_x(identity_image_x)
        #     dis_x_generated_x = self.discriminator_x(generated_image_x)
        #
        #     dis_x_cycle_y = self.discriminator_x(cycled_image_y)
        #     dis_x_identity_y = self.discriminator_x(identity_image_y)
        #     dis_x_generated_y = self.discriminator_x(generated_image_y)
        #
        #     dis_y_cycle_x = self.discriminator_y(cycled_image_x)
        #     dis_y_identity_x = self.discriminator_y(identity_image_x)
        #     dis_y_generated_x = self.discriminator_y(generated_image_x)
        #
        #     dis_y_cycle_y = self.discriminator_y(cycled_image_y)
        #     dis_y_identity_y = self.discriminator_y(identity_image_y)
        #     dis_y_generated_y = self.discriminator_y(generated_image_y)

        return loss_gen_dis_y_to_x, loss_gen_dis_x_to_y, loss_gen_cycle_x, loss_gen_cycle_y

    def train_(self, input_data, input_label):
        x_idx = np.where(input_label[:, 0] == 1)
        y_idx = np.where(input_label[:, 1] == 1)

        if x_idx[0].shape[0] == 0 or y_idx[0].shape[0] == 0:
            return None, None, None, None, None, None

        input_x = input_data[x_idx]
        input_y = input_data[y_idx]

        if input_x.shape[0] == 1 or input_y.shape[0] == 1:
            return None, None, None, None, None, None

        loss_gen_dis_y_to_x, loss_gen_dis_x_to_y, loss_gen_cycle_x, loss_gen_cycle_y = self.train_generator(input_x, input_y)
        # loss_discriminator_x, loss_discriminator_y = self.train_discriminator(input_x, input_y)
        # loss_discriminator_x, loss_discriminator_y = self.train_discriminator(input_x, input_y)
        # loss_discriminator_x, loss_discriminator_y = self.train_discriminator(input_x, input_y)
        # loss_discriminator_x, loss_discriminator_y = self.train_discriminator(input_x, input_y)
        # loss_discriminator_x, loss_discriminator_y = self.train_discriminator(input_x, input_y)
        # loss_discriminator_x, loss_discriminator_y = self.train_discriminator(input_x, input_y)
        # loss_discriminator_x, loss_discriminator_y = self.train_discriminator(input_x, input_y)
        loss_discriminator_x, loss_discriminator_y = self.train_discriminator(input_x, input_y)

        return loss_gen_dis_y_to_x, loss_gen_dis_x_to_y, loss_gen_cycle_x, loss_gen_cycle_y, loss_discriminator_x, loss_discriminator_y
