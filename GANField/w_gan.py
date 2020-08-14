import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D, AveragePooling2D, ReLU, MaxPooling2D, UpSampling2D


class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__(name='generator')

        self.fc_1 = Dense(8)
        self.bn_1 = BatchNormalization()
        self.relu_1 = ReLU()

        self.fc_2 = Dense(16)
        self.bn_2 = BatchNormalization()
        self.relu_2 = ReLU()

        self.fc_3 = Dense(32)
        self.bn_3 = BatchNormalization()
        self.relu_3 = ReLU()

        self.fc_4 = Dense(16)
        self.bn_4 = BatchNormalization()
        self.relu_4 = ReLU()

        self.fc_5 = Dense(8)
        self.bn_5 = BatchNormalization()
        self.relu_5 = ReLU()

        self.out = Dense(2, activation='tanh')

    def __call__(self, input_tensor, training):
        # feature_1 = self.relu_1(self.bn_1(self.fc_1(input_tensor)), training=training)
        # feature_2 = self.relu_2(self.bn_2(self.fc_2(feature_1)), training=training)
        # feature_3 = self.relu_3(self.bn_3(self.fc_3(feature_2)), training=training)
        # feature_4 = self.relu_4(self.bn_4(self.fc_4(feature_3)), training=training)
        # feature_5 = self.relu_5(self.bn_5(self.fc_5(feature_4)), training=training)
        feature_1 = self.relu_1(self.fc_1(input_tensor))
        feature_2 = self.relu_2(self.fc_2(feature_1))
        feature_3 = self.relu_3(self.fc_3(feature_2))
        feature_4 = self.relu_4(self.fc_4(feature_3))
        feature_5 = self.relu_5(self.fc_5(feature_4))
        out = self.out(feature_5)

        return out

    def gen_loss(self, gen_sample, discriminator):
        gen_score = discriminator(gen_sample, training=False)
        return tf.reduce_mean(-gen_score)

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
        gen_sample = self(train_input, training=True)
        return gen_sample


class Discriminator(tf.keras.Model):
    def __init__(self, gp=True):
        super().__init__(name='discriminator')

        self.fc_1 = Dense(8)
        self.relu_1 = LeakyReLU()

        self.fc_2 = Dense(16)
        self.relu_2 = LeakyReLU()

        self.fc_3 = Dense(32)
        self.relu_3 = LeakyReLU()

        self.fc_4 = Dense(16)
        self.relu_4 = LeakyReLU()

        self.fc_5 = Dense(8)
        self.relu_5 = LeakyReLU()

        self.out = Dense(1, activation='sigmoid')
        self.gp = gp

    def __call__(self, input_tensor, training):
        feature_1 = self.relu_1(self.fc_1(input_tensor))
        feature_2 = self.relu_2(self.fc_2(feature_1))
        feature_3 = self.relu_3(self.fc_3(feature_2))
        feature_4 = self.relu_4(self.fc_4(feature_3))
        feature_5 = self.relu_5(self.fc_5(feature_4))
        out = self.out(feature_5)

        return out

    def dis_loss(self, score_real, score_fake, input_fake_intp):
        with tf.GradientTape() as tmp_tape:
            tmp_tape.watch(input_fake_intp)
            gp_dis = tmp_tape.gradient(self(input_fake_intp, training=False), input_fake_intp)
        tmp = tf.norm(gp_dis, axis=1)
        gp_loss = 10. * ((tmp - 1) ** 2)
        return tf.reduce_mean(-score_real + score_fake), tf.reduce_mean(gp_loss)

    @tf.function
    def train(self, train_input_real, train_input_fake, optimizer):
        if self.gp:
            k = np.tile(np.expand_dims(np.random.uniform(0., 1., train_input_fake.shape[0]), 1), [1, 2]).astype(np.float32)
            train_input_fake_intp = k * train_input_fake + (1 - k) * train_input_real

        with tf.GradientTape() as dis_tape:
            dis_score_real = self(train_input_real, training=True)
            dis_score_fake = self(train_input_fake, training=True)
            dis_loss_clf, dis_loss_gp = self.dis_loss(dis_score_real, dis_score_fake, train_input_fake_intp)
            train_var = self.trainable_variables
            grad_of_discriminator = dis_tape.gradient(dis_loss_clf + dis_loss_gp, train_var)
            optimizer.apply_gradients(zip(grad_of_discriminator, train_var))
        return dis_loss_clf, dis_loss_gp

    @tf.function
    def test(self, train_input):
        dis_score = self(train_input, training=False)
        return dis_score
