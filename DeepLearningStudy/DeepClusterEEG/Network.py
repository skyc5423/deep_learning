import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, LeakyReLU, ReLU, BatchNormalization, Conv2D, AveragePooling2D, UpSampling2D, Concatenate, Flatten, Reshape


class AutoEncoder(tf.keras.Model):
    def __init__(self, name, feature_num, k, beta, delta, gamma, noise_std):
        super().__init__(name=name)
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.min_max_array = None
        self.noise_std = noise_std
        # self.encoder_1_abs = Dense(8, 'elu')
        # self.encoder_1_rel = Dense(8, 'elu')
        # self.encoder_1_rat = Dense(8, 'elu')
        # self.encoder_1_etc = Dense(8, 'elu')
        self.encoder_1 = Dense(16, 'elu')
        self.encoder_2 = Dense(16, 'elu')
        self.encoder_3 = Dense(k, 'tanh')
        self.decoder_1 = Dense(16, 'elu')
        self.decoder_2 = Dense(16, 'elu')
        self.decoder_3 = Dense(feature_num, 'elu')
        self.out = Dense(feature_num, 'tanh')

        self.discriminator_1 = Dense(32, 'relu', trainable=True, name='discriminator_1')
        self.discriminator_2 = Dense(32, 'relu', trainable=True, name='discriminator_2')
        self.discriminator_3 = Dense(32, 'relu', trainable=True, name='discriminator_3')
        self.discriminator_4 = Dense(1, trainable=True, name='discriminator_4')

        self.cluster_network_1 = Dense(k, 'relu')
        self.cluster_network_2 = Dense(k, 'relu')
        self.cluster_network_3 = Dense(k)

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0002, epsilon=1e-07, amsgrad=False,
        )

    def set_normalize_factor(self, min_max_array):
        self.min_max_array = min_max_array

    def get_normalize_factor(self):
        return self.min_max_array

    def cluster_network(self, encode_f_batch):
        feature_1 = self.cluster_network_1(encode_f_batch)
        feature_2 = self.cluster_network_2(feature_1)
        feature_3 = self.cluster_network_3(feature_2)
        return feature_3

    def discriminate(self, input_batch, encode_f_batch):
        input_tmp = tf.concat([input_batch, encode_f_batch], axis=1)
        feature_1 = self.discriminator_1(input_tmp)
        feature_2 = self.discriminator_2(feature_1)
        feature_3 = self.discriminator_3(feature_2)
        out = self.discriminator_4(feature_3)
        return out

    def encode(self, input_batch):
        # abs_feature = tf.slice(input_batch, [0, 0 * 19 * 6], [-1, 1 * 19 * 6])
        # rel_feature = tf.slice(input_batch, [0, 1 * 19 * 6], [-1, 1 * 19 * 6])
        # rat_feature = tf.slice(input_batch, [0, 2 * 19 * 6], [-1, 19 * 3])
        # etc_feature = tf.slice(input_batch, [0, 2 * 19 * 6 + 19 * 3], [-1, -1])
        # abs_en_1 = self.encoder_1_abs(abs_feature)
        # rel_en_1 = self.encoder_1_rel(rel_feature)
        # rat_en_1 = self.encoder_1_rat(rat_feature)
        # etc_en_1 = self.encoder_1_etc(etc_feature)
        # combined_feature_1 = tf.concat([abs_en_1, rel_en_1, rat_en_1, etc_en_1], axis=1)
        combined_feature_1 = self.encoder_1(input_batch)
        feature_2 = self.encoder_2(combined_feature_1)
        feature_3 = self.encoder_3(feature_2)
        return feature_3

    def decode(self, encode_f_batch):
        feature_1 = self.decoder_1(encode_f_batch)
        feature_2 = self.decoder_2(feature_1)
        feature_3 = self.decoder_3(feature_2)
        return self.out(feature_3)

    def encode_cluster_net(self, input_batch):
        encode_batch = self.encode(input_batch)
        return self.cluster_network(encode_batch)

    def __call__(self, input_batch, training=True):
        encode_f_batch = self.encode(input_batch)
        decode_batch = self.decode(encode_f_batch)

        return encode_f_batch, decode_batch

    @staticmethod
    def loss_l2_norm(input_image, cycle_image):
        return tf.reduce_mean(tf.abs(input_image - cycle_image))

    def diverge_score(self, real_score, fake_score):
        real_dis_out = tf.math.log(2.) - tf.math.log(1 + tf.exp(-real_score))  # real score increases -> real_dis_out decreases
        fake_dis_out = tf.math.log(2 - tf.exp(tf.math.log(2.) - tf.math.log(1 + tf.exp(-fake_score))) + 1E-7)  # fake score increases -> fake_dis_out decreases
        return tf.sigmoid(real_dis_out), tf.sigmoid(fake_dis_out)

    def loss_encoder_autoencoder(self, input_image, encode_f_real, encode_f_fake):
        kl = tf.keras.losses.KLDivergence()
        loss_kld = kl(encode_f_real, tf.random.normal(encode_f_real.shape))

        discriminator_out_real = self.discriminate(input_image, encode_f_real)
        discriminator_out_fake = self.discriminate(input_image, encode_f_fake)

        real_dis_out = tf.sigmoid(discriminator_out_real)
        fake_dis_out = tf.sigmoid(discriminator_out_fake)

        loss_discriminator = tf.reduce_mean(-self.beta * (tf.math.log(real_dis_out + 1E-7) + tf.math.log(1 - fake_dis_out + 1E-7)))
        return loss_discriminator + self.gamma * loss_kld

    def loss_encoder_cluster(self, input_image, encode_f_real, encode_f_fake, y_cluster):
        kl = tf.keras.losses.KLDivergence()
        loss_kld = tf.reduce_sum(kl(tf.random.normal(encode_f_real.shape, np.mean(y_cluster)), y_cluster))

        discriminator_out_real = self.discriminate(input_image, encode_f_real)
        discriminator_out_fake = self.discriminate(input_image, encode_f_fake)

        # real_dis_out, fake_dis_out = self.diverge_score(discriminator_out_real, discriminator_out_fake)
        real_dis_out = tf.sigmoid(discriminator_out_real)
        fake_dis_out = tf.sigmoid(discriminator_out_fake)

        loss_discriminator = tf.reduce_mean(-self.beta * (tf.math.log(real_dis_out + 1E-7) + tf.math.log(1 - fake_dis_out + 1E-7)))
        return loss_discriminator + self.gamma * loss_kld

    def loss_decoder(self, input_image, decode_img_noise, decode_img_real):
        return 5 * tf.reduce_mean(tf.abs(decode_img_noise - decode_img_real)) + self.delta * tf.reduce_mean(tf.abs(input_image - decode_img_real))

    def affinity_mat(self, encode_f_real):
        zi = tf.tile(tf.expand_dims(encode_f_real, axis=0), [encode_f_real.shape[0], 1, 1])
        zj = tf.tile(tf.expand_dims(encode_f_real, axis=1), [1, encode_f_real.shape[0], 1])
        w = tf.exp(-tf.square(zi - zj) / 2)
        return w

    def loss_cluster(self, w, y):
        yi = tf.tile(tf.expand_dims(y, axis=0), [y.shape[0], 1, 1])
        yj = tf.tile(tf.expand_dims(y, axis=1), [1, y.shape[0], 1])
        yij = tf.square(yi - yj)

        return tf.reduce_mean(tf.reduce_sum(w * yij, axis=2))

    def compute_cholesky_if_possible(self, cluster_y_tilde):

        try:
            cholesky_l = tf.linalg.cholesky(tf.matmul(tf.transpose(cluster_y_tilde), cluster_y_tilde))
            return cholesky_l
        except:
            jitter = 1E-9
            while jitter < 1.0:
                try:
                    cholesky_l = tf.linalg.cholesky(tf.matmul(tf.transpose(cluster_y_tilde), cluster_y_tilde) + jitter * tf.eye(tf.transpose(cluster_y_tilde).shape[0]))
                    return cholesky_l
                except:
                    jitter *= 10

            return None

    def train_autoencoder(self, input_image, cluster=False, epoch_idx=None):

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            fake_image = np.array(input_image)
            np.random.shuffle(fake_image)

            encode_f_real = self.encode(input_image)
            encode_f_fake = self.encode(fake_image)

            noised_encode_f_real = tf.random.normal(encode_f_real.shape, 1, self.noise_std) * encode_f_real
            decode_img_noise = self.decode(noised_encode_f_real)
            decode_img_real = self.decode(encode_f_real)

            loss_decoder = self.loss_decoder(input_image, decode_img_noise, decode_img_real)

            if cluster:
                cluster_y_tilde = self.cluster_network(encode_f_real)  # M x D
                affinity_mat = self.affinity_mat(encode_f_real)  # M x M x D
                cholesky_l = self.compute_cholesky_if_possible(cluster_y_tilde)

                cluster_y = tf.matmul(cluster_y_tilde, tf.transpose(tf.linalg.inv(cholesky_l))) * np.sqrt(encode_f_real.shape[0])
                if epoch_idx is not None and epoch_idx % 50 == 0:
                    fig, ax = plt.subplots(2, 2, figsize=(8, 7))
                    ax[0, 0].pcolormesh(cholesky_l.numpy() + 1E-9, cmap='autumn')
                    ax[0, 1].pcolormesh(cluster_y.numpy() + 1E-9, cmap='autumn')
                    ax[1, 0].pcolormesh(encode_f_real.numpy() + 1E-9, cmap='autumn')
                    ax[1, 1].pcolormesh(cluster_y_tilde.numpy() + 1E-9, cmap='autumn')
                    fig.savefig('./tmp_%d.png' % epoch_idx)
                    plt.close(fig)
                loss_encoder = self.loss_encoder_cluster(input_image, encode_f_real, encode_f_fake, cluster_y)
                loss_cluster = self.loss_cluster(affinity_mat, cluster_y)
            else:
                loss_cluster = 0
                loss_encoder = self.loss_encoder_autoencoder(input_image, encode_f_real, encode_f_fake)

            loss_total = loss_encoder + loss_decoder + loss_cluster

        grads = tape.gradient(loss_total, self.trainable_variables)

        grads_vars = zip(grads, self.trainable_variables)

        self.optimizer.apply_gradients(grads_vars)

        return loss_total, loss_encoder, loss_decoder, loss_cluster

    def train_(self, input_data, epoch_idx, cluster=True):
        loss_total, loss_encoder, loss_decoder, loss_cluster = self.train_autoencoder(input_data, cluster, epoch_idx)

        return loss_total, loss_encoder, loss_decoder, loss_cluster
