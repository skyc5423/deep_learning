import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, LeakyReLU, ReLU, BatchNormalization, Conv2D, AveragePooling2D, UpSampling2D, Concatenate, Flatten, Reshape


class AutoEncoder(tf.keras.Model):
    def __init__(self, name, dims=64):
        super().__init__(name=name)
        self.dims = dims

        self.conv_1 = Conv2D(8, 3, strides=(2, 2), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.relu_1 = LeakyReLU()

        self.conv_2 = Conv2D(16, 3, strides=(2, 2), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.bn_2 = BatchNormalization()
        self.relu_2 = LeakyReLU()

        self.conv_3 = Conv2D(32, 3, strides=(2, 2), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.bn_3 = BatchNormalization()
        self.relu_3 = LeakyReLU()

        self.conv_4 = Conv2D(64, 3, strides=(2, 2), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.bn_4 = BatchNormalization()
        self.relu_4 = LeakyReLU()

        self.reshape = Reshape([-1])

        self.fc_1 = Dense(self.dims)

        self.fc_2 = Dense(8 * 8 * 64)
        self.relu_fc_2 = LeakyReLU()

        self.reshape_conv = Reshape([8, 8, 64])

        self.conv_5 = Conv2D(64, 3, strides=(1, 1), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.bn_5 = BatchNormalization()
        self.relu_5 = LeakyReLU()

        self.conv_6 = Conv2D(32, 3, strides=(1, 1), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.bn_6 = BatchNormalization()
        self.relu_6 = LeakyReLU()

        self.conv_7 = Conv2D(16, 3, strides=(1, 1), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.bn_7 = BatchNormalization()
        self.relu_7 = LeakyReLU()

        self.conv_8 = Conv2D(8, 3, strides=(1, 1), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation=None, use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.bn_8 = BatchNormalization()
        self.relu_8 = LeakyReLU()

        self.conv_9 = Conv2D(3, 3, strides=(1, 1), padding='same', data_format=None,
                             dilation_rate=(1, 1), activation='tanh', use_bias=True,
                             kernel_initializer='glorot_uniform', bias_initializer='zeros',
                             kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                             kernel_constraint=None, bias_constraint=None)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0002, beta_1=0.9, epsilon=1e-07, amsgrad=False,
        )

    def __call__(self, input_batch, training=True):
        feature_1 = self.relu_1(self.conv_1(input_batch))
        # feature_2 = self.relu_2(self.bn_2(self.conv_2(feature_1)))
        # feature_3 = self.relu_3(self.bn_3(self.conv_3(feature_2)))
        # feature_4 = self.relu_4(self.bn_4(self.conv_4(feature_3)))
        feature_2 = self.relu_2(self.conv_2(feature_1))
        feature_3 = self.relu_3(self.conv_3(feature_2))
        feature_4 = self.relu_4(self.conv_4(feature_3))

        reshape = self.reshape(feature_4)

        feature_fc_1 = self.fc_1(reshape)
        self.feature_fc_1 = feature_fc_1
        mean_feature_fc = feature_fc_1[:, :int(self.dims / 2)]
        log_std_feature_fc = feature_fc_1[:, int(self.dims / 2):]
        z = self.reparameterize(mean_feature_fc, log_std_feature_fc)

        feature_fc_2 = self.relu_fc_2(self.fc_2(z))

        reshape_conv = self.reshape_conv(feature_fc_2)

        feature_5 = UpSampling2D()(reshape_conv)
        feature_5 = self.relu_5(self.conv_5(feature_5))
        # feature_5 = self.relu_5(self.bn_5(self.conv_5(feature_5)))
        feature_6 = UpSampling2D()(feature_5)
        feature_6 = self.relu_6(self.conv_6(feature_6))
        # feature_6 = self.relu_6(self.bn_6(self.conv_6(feature_6)))
        feature_7 = UpSampling2D()(feature_6)
        feature_7 = self.relu_7(self.conv_7(feature_7))
        # feature_7 = self.relu_7(self.bn_7(self.conv_7(feature_7)))
        feature_8 = UpSampling2D()(feature_7)
        feature_8 = self.relu_8(self.conv_8(feature_8))
        # feature_8 = self.relu_8(self.bn_8(self.conv_8(feature_8)))

        feature_9 = self.conv_9(feature_8)

        out = feature_9

        return out, mean_feature_fc, log_std_feature_fc, z

    def reparameterize(self, mean, log_std):
        eps = tf.random.normal([mean.shape[0], mean.shape[1]])
        sigma = tf.exp(log_std)
        z = tf.sqrt(sigma) * eps + mean
        return z

    def encoder_loss(self, z, mean, log_std):
        std = tf.exp(log_std)

        loss = tf.square(mean) + std - log_std - 1

        return tf.reduce_mean(loss)

    @staticmethod
    def loss_l2_norm(input_image, cycle_image):
        return tf.reduce_mean(tf.square(input_image - cycle_image))

    def train_autoencoder(self, input_image, epoch):
        with tf.GradientTape() as tape:
            weight = tf.maximum(0, (epoch - 50))
            weight = tf.cast(weight, tf.float32)

            tape.watch(self.trainable_variables)

            generated_image, mean, log_std, z = self(input_image)

            encoder_loss = self.encoder_loss(z, mean, log_std)

            decoder_loss = self.loss_l2_norm(input_image, generated_image)

            loss_total = 0.0001 * weight * encoder_loss + decoder_loss

        grads = tape.gradient(loss_total, self.trainable_variables)
        grads_vars = zip(grads, self.trainable_variables)

        self.optimizer.apply_gradients(grads_vars)

        return loss_total

    def train_(self, input_data, epoch):
        loss_total = self.train_autoencoder(input_data, epoch)

        return loss_total
