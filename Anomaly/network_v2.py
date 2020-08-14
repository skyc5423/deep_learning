import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, AveragePooling2D, ReLU, MaxPooling2D, UpSampling2D


class Generator(tf.keras.Model):
    def __init__(self, num_layer=4):
        super().__init__(name='generator')

        self.up_samp = UpSampling2D(size=(2, 2), interpolation='bilinear')

        self.conv_1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1), kernel_initializer='random_normal')
        self.relu_1 = LeakyReLU()

        # self.conv_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1))
        # self.relu_2 = LeakyReLU()

        self.conv_3 = Conv2D(16, (3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1), kernel_initializer='random_normal')
        self.relu_3 = LeakyReLU()

        self.conv_4 = Conv2D(8, (3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1), kernel_initializer='random_normal')
        self.relu_4 = LeakyReLU()

        self.conv_5 = Conv2D(1, (3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1), activation='tanh', kernel_initializer='random_normal')

    def __call__(self, input_tensor, training):
        input_feature_out = input_tensor['out']
        up_samp = self.up_samp(input_feature_out)
        feature_1 = self.relu_1(self.conv_1(up_samp))
        # feature_2 = self.relu_2(self.conv_2(feature_1))
        feature_3 = self.relu_3(self.conv_3(feature_1))
        feature_4 = self.relu_4(self.conv_4(feature_3))
        feature_5 = self.conv_5(feature_4)

        rtn = {}
        # rtn['up_samp'] = up_samp
        # rtn['feature_1'] = feature_1
        # rtn['feature_3'] = feature_3
        # rtn['feature_4'] = feature_4
        rtn['out'] = feature_5

        return rtn


def vgg_layers(layer_names):
    # vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    # vgg.trainable = False
    #
    # outputs = [vgg.get_layer(name).output for name in layer_names]
    #
    # model = tf.keras.Model([vgg.input], outputs)
    return


class Discriminator(tf.keras.Model):
    def __init__(self, num_layer=4):
        super().__init__(name='discriminator')

        self.conv_1 = Conv2D(16, (5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1), kernel_initializer='random_normal')
        self.relu_1 = LeakyReLU()

        self.conv_2 = Conv2D(32, (5, 5), strides=(1, 1), padding='same', dilation_rate=(1, 1))
        self.relu_2 = LeakyReLU()

        self.conv_3 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1), kernel_initializer='random_normal')
        self.relu_3 = LeakyReLU()

        self.conv_4 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', dilation_rate=(1, 1), kernel_initializer='random_normal')
        self.relu_4 = LeakyReLU()

        self.conv_5 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1))
        self.relu_5 = LeakyReLU()

        self.conv_6 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1), kernel_initializer='random_normal')
        self.relu_6 = LeakyReLU()

        self.conv_7 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=(2, 2), kernel_initializer='random_normal')
        self.relu_7 = LeakyReLU()

        self.conv_8 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=(4, 4), kernel_initializer='random_normal')
        self.relu_8 = LeakyReLU()

        self.conv_9 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=(8, 8))
        self.relu_9 = LeakyReLU()

        self.conv_10 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=(16, 16))
        self.relu_10 = LeakyReLU()

        self.conv_11 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=(1, 1), kernel_initializer='random_normal')
        self.relu_11 = LeakyReLU()

    def __call__(self, input_tensor, training):
        feature_1 = self.relu_1(self.conv_1(input_tensor))
        feature_2 = self.relu_2(self.conv_2(feature_1))
        feature_3 = self.relu_3(self.conv_3(feature_2))
        feature_4 = self.relu_4(self.conv_4(feature_3))
        feature_5 = self.relu_5(self.conv_5(feature_4))
        feature_6 = self.relu_6(self.conv_6(feature_5))
        feature_7 = self.relu_7(self.conv_7(feature_6))
        feature_8 = self.relu_8(self.conv_8(feature_7))
        feature_9 = self.relu_9(self.conv_9(feature_8))
        feature_10 = self.relu_10(self.conv_10(feature_9))
        feature_11 = self.relu_11(self.conv_11(feature_10))

        rtn = {}
        # rtn['feature_1'] = feature_1
        # rtn['feature_3'] = feature_3
        # rtn['feature_4'] = feature_4
        # rtn['feature_6'] = feature_6
        # rtn['feature_7'] = feature_7
        # rtn['feature_8'] = feature_8
        rtn['out'] = feature_11

        return rtn
