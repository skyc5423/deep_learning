import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, random

from dc_gan import *

generator = Generator()
discriminator = Discriminator()

from make_sample import *
from w_gan import Generator, Discriminator

# from vanilla_gan import Generator, Discriminator

EPOCH_NUM = 300
BATCH_SIZE = 256
DIS_TRAIN_NUM = 5
GEN_TRAIN_NUM = 1
out_path = 'wgan_gp_10'

iter_in_epoch = 8
num_of_cluster = 1

cir_sam_1 = circular_sample(int(BATCH_SIZE * iter_in_epoch / num_of_cluster), 0.5, 0.5, 0.2, 0.15)
# cir_sam_2 = circular_sample(BATCH_SIZE * 8, -0.5, 0.5, 0.2, 0.15)
# cir_sam_3 = circular_sample(BATCH_SIZE * 8, 0.5, -0.5, 0.2, 0.15)
# cir_sam_4 = circular_sample(int(BATCH_SIZE * iter_in_epoch / num_of_cluster), -0.5, -0.5, 0.2, 0.15)
# cir_sam = np.concatenate([cir_sam_1, cir_sam_2, cir_sam_3, cir_sam_4], axis=0).astype(np.float32)
# cir_sam = np.concatenate([cir_sam_1, cir_sam_4], axis=0).astype(np.float32)
cir_sam = np.concatenate([cir_sam_1], axis=0).astype(np.float32)


def shuffle(a):
    rand_idx = random.sample(range(a.shape[0]), a.shape[0])
    return np.array(a[rand_idx])


no_train = False
load_path = 'model'

cir_sam = shuffle(cir_sam)

g = Generator()
d = Discriminator()

# optimizer_d = tf.keras.optimizers.RMSprop(learning_rate=0.001)
# optimizer_g = tf.keras.optimizers.RMSprop(learning_rate=0.001)

optimizer_d = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0., beta_2=0.9)
optimizer_g = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0., beta_2=0.9)

if load_path is not None:
    g.load_weights('./%s/model_g' % load_path)
    d.load_weights('./%s/model_d' % load_path)

for epoch in range(EPOCH_NUM):
    if no_train:
        break

    d_loss_clf = 0
    d_loss_gp = 0
    g_loss = 0

    random_z_gen_train = np.random.uniform(-1, 1, [GEN_TRAIN_NUM * BATCH_SIZE * iter_in_epoch, 2])
    random_z_dis_train = np.random.uniform(-1, 1, [DIS_TRAIN_NUM * BATCH_SIZE * iter_in_epoch, 2])

    for i in range(iter_in_epoch):

        for n in range(DIS_TRAIN_NUM):
            tmp_idx = i * DIS_TRAIN_NUM + n
            fake_sam = g.test(random_z_dis_train[tmp_idx * BATCH_SIZE:(tmp_idx + 1) * BATCH_SIZE])

            d_loss_clf_tmp, d_loss_gp_tmp = d.train(cir_sam[BATCH_SIZE * i:BATCH_SIZE * (i + 1)], fake_sam, optimizer_d)
            d_loss_clf = d_loss_clf_tmp
            d_loss_gp = d_loss_gp_tmp

        for n in range(GEN_TRAIN_NUM):
            tmp_idx = i * GEN_TRAIN_NUM + n
            g_loss = g.train(random_z_gen_train[tmp_idx * BATCH_SIZE:BATCH_SIZE * (tmp_idx + 1)], d, optimizer_g)

        dis_train_sample = np.array(g.test(random_z_gen_train))

    # random_z_gen = np.random.uniform(-1, 1, [1000, 2])
    # gen_test = g.test(random_z_gen)
    # gen_test = np.array(gen_test)

    # gen_train = np.array(g.test(random_z_gen_train))

    # print('epoch-%02d: discriminator loss (gp, clf)=(%.3f, %.3f), generator loss=%.3f' % (
    # epoch, d_loss_gp / DIS_TRAIN_NUM / iter_in_epoch, d_loss_clf / DIS_TRAIN_NUM / iter_in_epoch, g_loss / iter_in_epoch))
    print('epoch-%02d: discriminator loss (gp, clf)=(%.3f, %.3f), generator loss=%.3f' % (
        epoch, d_loss_gp, d_loss_clf, g_loss))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # for axis in ax:
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    xy_dim = 200

    x_mesh = np.linspace(-1, 1, xy_dim)
    y_mesh = np.linspace(-1, 1, xy_dim)
    xy = np.zeros([xy_dim ** 2, 2])
    for x in range(xy_dim):
        for y in range(xy_dim):
            xy[xy_dim * x + y, 0] = x_mesh[x]
            xy[xy_dim * x + y, 1] = y_mesh[y]

    xy = np.array(d.test(xy))
    xy_score = np.zeros([xy_dim, xy_dim])

    for x in range(xy_dim):
        for y in range(xy_dim):
            xy_score[x, y] = xy[xy_dim * x + y]

    xy_binary = np.where(xy_score < 0.499, 1., np.where(xy_score > 0.501, 1., 0.))

    ax.pcolormesh(x_mesh, y_mesh, xy_binary, alpha=1.0, cmap='gray', vmin=0., vmax=1.)

    ax.pcolormesh(x_mesh, y_mesh, xy_score, alpha=0.25, cmap='jet', vmin=0., vmax=1.)

    for n in range(cir_sam.shape[0]):
        ax.scatter(cir_sam[n, 0], cir_sam[n, 1], color='blue', s=1, alpha=0.5)

    # for n in range(gen_test.shape[0]):
    #     ax.scatter(gen_test[n, 0], gen_test[n, 1], color='yellow', s=1, alpha=1.)

    for n in range(dis_train_sample.shape[0]):
        ax.scatter(dis_train_sample[n, 0], dis_train_sample[n, 1], color='red', s=1, alpha=0.8)

    if not os.path.isdir('./%s' % out_path):
        os.mkdir('./%s' % out_path)
    fig.savefig('./%s/result_%d.png' % (out_path, epoch))
    plt.close(fig)

g.save_weights('./%s/model_g' % 'model')
d.save_weights('./%s/model_d' % 'model')

import imageio
from PIL import Image

paths = []
for i in range(EPOCH_NUM):
    img = Image.open('./%s/result_%d.png' % (out_path, i))
    paths.append(img)
for i in range(120):
    paths.append(img)
imageio.mimsave('./%s/result.gif' % out_path, paths, fps=30)

# rec_sam = rectangular_sample(1000, 0.8, 0.9, 0.1, 0.05)

# for i in range(cir_sam.shape[0]):
#     plt.scatter(cir_sam[i, 0], cir_sam[i, 1], color='red')
# plt.scatter(rec_sam[i, 0], rec_sam[i, 1], color='blue')
# plt.show()
