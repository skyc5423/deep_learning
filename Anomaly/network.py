import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, AveragePooling2D, ReLU, MaxPooling2D


class Generator(tf.keras.Model):
    def __init__(self, num_layer=4):
        super().__init__(name='generator')

        # self.input_layer = Dense(units=7 * 7 * 512, activation='tanh')
        # self.bn_input = BatchNormalization()
        # self.relu_input = ReLU()
        # self.reshape_input = Reshape((7, 7, 512))

        use_bias = False
        self.num_layer = num_layer

        # self.convt_1_1 = Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=use_bias)
        # self.bn_1_1 = BatchNormalization()
        # self.relu_1_1 = ReLU()
        # self.convt_1_2 = Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=use_bias)
        # self.bn_1_2 = BatchNormalization()
        # self.relu_1_2 = ReLU()
        #
        self.convt_2_1 = Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=use_bias)
        self.bn_2_1 = BatchNormalization()
        self.relu_2_1 = ReLU()
        self.convt_2_2 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=use_bias)
        self.bn_2_2 = BatchNormalization()
        self.relu_2_2 = ReLU()

        self.convt_3_1 = Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', use_bias=use_bias)
        self.bn_3_1 = BatchNormalization()
        self.relu_3_1 = ReLU()
        self.convt_3_2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=use_bias)
        self.bn_3_2 = BatchNormalization()
        self.relu_3_2 = ReLU()

        self.convt_4_1 = Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same', use_bias=use_bias)
        self.bn_4_1 = BatchNormalization()
        self.relu_4_1 = ReLU()
        self.convt_4_2 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', use_bias=use_bias)
        self.bn_4_2 = BatchNormalization()
        self.relu_4_2 = ReLU()

        self.convt_5_1 = Conv2DTranspose(8, (3, 3), strides=(1, 1), padding='same', use_bias=use_bias)
        self.bn_5_1 = BatchNormalization()
        self.relu_5_1 = ReLU()
        self.convt_5_2 = Conv2DTranspose(8, (5, 5), strides=(2, 2), padding='same', use_bias=use_bias)
        self.bn_5_2 = BatchNormalization()
        self.relu_5_2 = ReLU()

        self.convt_6_1 = Conv2DTranspose(8, (3, 3), strides=(1, 1), padding='same', use_bias=use_bias)
        self.relu_6_1 = ReLU()
        self.convt_6_2 = Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same', use_bias=use_bias, activation='tanh')

    def __call__(self, input_tensor, training):
        input_feature_out = input_tensor['out']  # 64x64
        # feature_2 = self.relu_2_1(self.bn_2_1(self.convt_2_2(input_feature_out), training=training))
        feature_3 = self.relu_3_1(self.bn_3_1(self.convt_3_2(input_feature_out), training=training))  # 64x64
        feature_4 = self.relu_4_1(self.bn_4_1(self.convt_4_2(feature_3), training=training))  # 128x128
        feature_5 = self.relu_5_1(self.bn_5_1(self.convt_5_2(feature_4), training=training))

        if self.num_layer == 4:
            feature_6 = self.convt_6_2(self.relu_6_1(self.convt_6_1(feature_5)))
        elif self.num_layer == 3:
            feature_6 = self.convt_6_2(self.relu_6_1(self.convt_6_1(feature_4)))
        elif self.num_layer == 2:
            feature_6 = self.convt_6_2(self.relu_6_1(self.convt_6_1(feature_3)))



        rtn = {}
        rtn['out'] = feature_6

        return rtn


def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


class Discriminator(tf.keras.Model):
    def __init__(self, num_layer=4):
        super().__init__(name='discriminator')

        self.conv_1_1 = Conv2D(8, (3, 3), strides=(1, 1), padding='same')
        self.relu_1_1 = LeakyReLU()
        self.conv_1_2 = Conv2D(8, (3, 3), strides=(1, 1), padding='same')
        self.relu_1_2 = LeakyReLU()

        self.pool_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')

        self.conv_2_1 = Conv2D(16, (3, 3), strides=(1, 1), padding='same')
        self.relu_2_1 = LeakyReLU()
        self.pool_2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')

        self.conv_3_1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')
        self.relu_3_1 = LeakyReLU()
        self.pool_3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')

        self.conv_4_1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')
        self.relu_4_1 = LeakyReLU()
        self.conv_4_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')

        self.num_layer=num_layer
        # self.relu_5_1 = LeakyReLU()
        # self.pool_5 = AveragePooling2D((2, 2), strides=(2, 2), padding='same')
        #
        # self.conv_6_1 = Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        # self.relu_6_1 = LeakyReLU()
        # self.conv_6_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same')
        # self.relu_6_2 = LeakyReLU()

        # self.pool_out = AveragePooling2D((7, 7), strides=(7, 7), padding='same')
        # self.fc_out = Dense(units=1)

    def __call__(self, input_tensor, training):
        feature_1 = self.relu_1_2(self.conv_1_2(self.relu_1_1(self.conv_1_1(input_tensor))))
        feature_2 = self.relu_2_1(self.conv_2_1(self.pool_1(feature_1)))
        feature_3 = self.relu_3_1(self.conv_3_1(self.pool_2(feature_2)))
        # feature_4 = self.relu_4_1(self.conv_4_1(self.pool_3(feature_3)))
        # feature_5 = self.relu_5_1(self.conv_5_1(self.pool_4(feature_4)))
        # out = self.conv_6_1(self.pool_5(feature_5))
        feature_4 = self.relu_4_1(self.conv_4_1(self.pool_3(feature_3)))
        if self.num_layer == 4:
            out = self.conv_4_2(feature_4)
        elif self.num_layer == 3:
            out = self.conv_4_2(feature_3)
        elif self.num_layer == 2:
            out = self.conv_4_2(feature_2)
        else:
            out = self.conv_4_2(feature_1)

        rtn = {}
        # rtn['feature_1'] = feature_1  # 256x256
        # rtn['feature_2'] = feature_2  # 128x128
        # rtn['feature_3'] = feature_3  # 64x64
        # rtn['feature_4'] = feature_4
        # rtn['feature_5'] = feature_5
        # rtn['feature_6'] = feature_6
        rtn['out'] = out  # 32x32

        return rtn
