import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, AveragePooling2D, ReLU, MaxPooling2D, Flatten


class CVAE(tf.keras.Model):
    def __init__(self, input_size, hidden_dim):
        super().__init__()

        use_bias = False

        self.enc_conv_1_1 = Conv2D(4, (3, 3), strides=(1, 1), padding='same')
        self.enc_relu_1_1 = LeakyReLU()
        self.enc_pool_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')

        self.enc_conv_2_1 = Conv2D(8, (3, 3), strides=(1, 1), padding='same')
        self.enc_relu_2_1 = LeakyReLU()
        self.enc_pool_2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')

        self.enc_conv_3_1 = Conv2D(16, (3, 3), strides=(1, 1), padding='same')
        self.enc_relu_3_1 = LeakyReLU()
        self.enc_pool_3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')

        self.enc_conv_4_1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')
        self.enc_relu_4_1 = LeakyReLU()
        #
        self.enc_flat = Flatten()
        self.enc_dense = Dense(hidden_dim, activation='relu')

        resize = (int(input_size[0] / 8), int(input_size[1] / 8), 32)

        self.dec_dense = Dense(units=resize[0] * resize[1] * resize[2], activation='relu')
        self.dec_reshape = Reshape(target_shape=resize)

        self.dec_convt_3_1 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', use_bias=use_bias)
        self.dec_bn_3_1 = BatchNormalization()
        self.dec_relu_3_1 = ReLU()

        self.dec_convt_4_1 = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', use_bias=use_bias)
        self.dec_bn_4_1 = BatchNormalization()
        self.dec_relu_4_1 = ReLU()

        self.dec_convt_5_1 = Conv2DTranspose(4, (3, 3), strides=(2, 2), padding='same', use_bias=use_bias)
        self.dec_bn_5_1 = BatchNormalization()
        self.dec_relu_5_1 = ReLU()

        self.dec_convt_6_1 = Conv2DTranspose(4, (3, 3), strides=(1, 1), padding='same', use_bias=use_bias)
        self.dec_relu_6_1 = ReLU()
        self.dec_convt_6_2 = Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same', use_bias=use_bias, activation='tanh')

    def encoder(self, input_image):
        feature_1 = self.enc_relu_1_1(self.enc_conv_1_1(input_image))
        feature_2 = self.enc_relu_2_1(self.enc_conv_2_1(self.enc_pool_1(feature_1)))
        feature_3 = self.enc_relu_3_1(self.enc_conv_3_1(self.enc_pool_2(feature_2)))
        feature_4 = self.enc_conv_4_1(self.enc_pool_3(feature_3))
        out = self.enc_dense(self.enc_flat(feature_4))

        rtn = {}
        rtn['out'] = out  # 32x32

        return rtn

    def decoder(self, input_feature, training):
        input_feature_out = input_feature
        feature_1 = self.dec_reshape(self.dec_dense(input_feature_out))
        feature_3 = self.dec_relu_3_1(self.dec_bn_3_1(self.dec_convt_3_1(feature_1), training=training))  # 64x64
        feature_4 = self.dec_relu_4_1(self.dec_bn_4_1(self.dec_convt_4_1(feature_3), training=training))  # 128x128
        feature_5 = self.dec_relu_5_1(self.dec_bn_5_1(self.dec_convt_5_1(feature_4), training=training))
        feature_6 = self.dec_convt_6_2(self.dec_relu_6_1(self.dec_convt_6_1(feature_5)))

        rtn = {}
        rtn['out'] = feature_6

        return rtn

    def __call__(self, input_image, training):
        return self.decoder(self.encoder(input_image)['out'], training)['out']

    def encode(self, x):
        # tmp = self.encoder(x)
        mean, logvar = tf.split(self.encoder(x)['out'], num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False, training=True):
        logits = self.decoder(z, training=training)['out']
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits
