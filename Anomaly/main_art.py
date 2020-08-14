from db_helper import DBHelper
# from network import Generator, vgg_layers
# from network import Discriminator
from sklearn.metrics import roc_curve, auc
from network_patch import Generator
from network_patch import Discriminator
from scipy.misc import imresize
import datetime
import matplotlib
import numpy as np
import cv2

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import os, time

BATCH_SIZE = 1


def autoencoder_loss(input_img, output_img):
    return tf.nn.l2_loss(tf.dtypes.cast(input_img[:, :, :], tf.float32) - (output_img[:, :, :] + 1) / 2.)


def ssim_loss(input_img, output_img):
    ssim_size = [(3, 3), (7, 7), (11, 11)]

    img_size = input_img.shape[1]

    total_ssim = 0.
    for tmp_size in ssim_size:
        kernel_size = tmp_size[0]
        step_size = tmp_size[1]
        c1 = 0.01
        c2 = 0.03
        total_step_size = int((img_size - kernel_size + 1) / step_size)
        input_img = tf.cast(input_img, tf.float32)
        output_img = tf.cast(output_img, tf.float32)
        output_img = (output_img + 1.) / 2.
        for i in range(BATCH_SIZE):
            for n in range(total_step_size):
                for m in range(total_step_size):
                    patch_input = tf.slice(input_img, [0, m * step_size, n * step_size, 0], [1, kernel_size, kernel_size, 1])
                    patch_output = tf.slice(output_img, [0, m * step_size, n * step_size, 0], [1, kernel_size, kernel_size, 1])

                    mean_input = tf.reduce_mean(tf.squeeze(patch_input))
                    std_input = tf.math.reduce_std(tf.squeeze(patch_input))
                    var_input = tf.math.square(std_input)

                    mean_output = tf.reduce_mean(tf.squeeze(patch_output))
                    std_output = tf.math.reduce_std(tf.squeeze(patch_output))
                    var_output = tf.math.square(std_output)

                    # covar = tf.reduce_sum((patch_input - mean_input) * (patch_output - mean_output)) / (img_size * img_size - 1)
                    # ssim = (2 * mean_input * mean_output + c1) * (2 * covar + c2) / (tf.math.square(mean_input) + tf.math.square(mean_output) + c1) / (var_input + var_output + c2)

                    ssim_lumi = (2 * mean_input * mean_output + c1) / (tf.math.square(mean_input) + tf.math.square(mean_output) + c1)
                    ssim_cont = (2 * std_output * std_input + c2) / (var_input + var_output + c2)

                    total_ssim += (ssim_lumi + ssim_cont)
    return total_ssim


def ssim_valid(input_img, output_img):
    ssim_size = [(3, 3), (7, 7), (11, 11)]
    img_size = input_img.shape[1]

    output_map = []
    for tmp_size in ssim_size:
        kernel_size = tmp_size[0]
        step_size = tmp_size[1]
        c1 = 0.01
        c2 = 0.03
        total_step_size = int((img_size - kernel_size) / step_size)
        tmp_output_map = np.zeros([total_step_size, total_step_size])
        for i in range(input_img.shape[0]):
            for n in range(total_step_size):
                for m in range(total_step_size):
                    patch_input = input_img[0, m * step_size:m * step_size + kernel_size, n * step_size + kernel_size, 0]
                    patch_output = output_img[0, m * step_size:m * step_size + kernel_size, n * step_size + kernel_size, 0]

                    mean_input = np.mean(patch_input)
                    std_input = np.std(patch_input)
                    var_input = np.square(std_input)

                    mean_output = np.mean(patch_output)
                    std_output = np.std(patch_output)
                    var_output = np.square(std_output)

                    # covar = tf.reduce_sum((patch_input - mean_input) * (patch_output - mean_output)) / (img_size * img_size - 1)
                    # ssim = (2 * mean_input * mean_output + c1) * (2 * covar + c2) / (np.square(mean_input) + np.square(mean_output) + c1) / (var_input + var_output + c2)

                    ssim_lumi = (2 * mean_input * mean_output + c1) / (tf.math.square(mean_input) + tf.math.square(mean_output) + c1)
                    ssim_cont = (2 * std_output * std_input + c2) / (var_input + var_output + c2)

                    tmp_output_map[m, n] = ssim_lumi + ssim_cont
        output_map.append(tmp_output_map)
    return output_map


def pyramid_down(batch_image):
    for i in range(batch_image.shape[0]):
        if i == 0:
            rtn = np.expand_dims(cv2.pyrDown(batch_image[i, :, :]), 0)

        else:
            rtn = np.concatenate([rtn, np.expand_dims(cv2.pyrDown(batch_image[i, :, :]), 0)], axis=0)
    return np.expand_dims(rtn, 3)


def salt_and_pepper(batch_images):
    patch_size = batch_images.shape[1]
    for i in range(batch_images.shape[0]):
        ns = np.random.randint(patch_size * patch_size, size=int(patch_size * patch_size * 0.05))
        for n in ns:
            batch_images[i, int(n / patch_size), n % patch_size, 0] = 0.
        ns = np.random.randint(patch_size * patch_size, size=int(patch_size * patch_size * 0.05))
        for n in ns:
            batch_images[i, int(n / patch_size), n % patch_size, 0] = 1.

    np.zeros([])

    return batch_images


@tf.function
def train_step(noise_images, images,
               generator, discriminator, discriminator_optimizer):
    with tf.GradientTape() as disc_tape:
        lat_var = discriminator[0](noise_images, training=True)
        generated_images = generator[0](lat_var, training=True)['out']
        disc_loss = 0.1 * autoencoder_loss(images, generated_images) - ssim_loss(images, generated_images)
        tv = discriminator[0].trainable_variables + generator[0].trainable_variables
        gradients_of_discriminator = disc_tape.gradient(disc_loss, tv)
        discriminator_optimizer[0].apply_gradients(zip(gradients_of_discriminator, tv))

    return disc_loss, generated_images


@tf.function
def test_step(images, generator, discriminator):
    with tf.GradientTape() as disc_tape:
        lat_var = discriminator[0](images, training=True)
        tmp = generator[0](lat_var, training=True)
        generated_images = tmp['out']

    return generated_images


def make_patch_image(images, patch_size, div_num):
    rtn = np.zeros([div_num * div_num, patch_size, patch_size, 1])
    for n in range(div_num):
        for m in range(div_num):
            rtn[div_num * n + m, :, :, 0] = images[0, m * patch_size:m * patch_size + patch_size, n * patch_size:n * patch_size + patch_size, 0]
    return rtn


def train(dataset, epochs, generator, discriminator, discriminator_optimizer, class_num):
    best_iou = 0.
    best_epoch = -1
    for epoch in range(epochs):
        start = time.time()
        gen_loss, dis_loss, itr_num = 0, 0, 0

        for x_batch, y_batch in dataset.train_ds:
            if not x_batch.shape[0] == BATCH_SIZE:
                break
            data_batch, label_batch = db_helper.get_data(x_batch, y_batch)
            noise_data_batch = salt_and_pepper(data_batch)

            image_patch_0 = make_patch_image(data_batch, 64, 8)
            noise_image_patch_0 = make_patch_image(noise_data_batch, 64, 8)

            _, _ = train_step(noise_image_patch_0[:, :, :], image_patch_0[:, :, :],
                              generator, discriminator, discriminator_optimizer)
            itr_num += 1.

        test_itr_num = 0.

        no_figure = False

        total_iou = 0
        total_num = 0

        for x_batch, y_batch in dataset.test_ds:

            if not x_batch.shape[0] == BATCH_SIZE:
                break
            data_batch, label_batch = db_helper.get_data(x_batch, y_batch)

            if not no_figure:
                print(datetime.datetime.now())

            image_patch_0 = make_patch_image(data_batch, 64, 8)

            gen_img_0 = test_step(image_patch_0[:, :, :],
                                  generator, discriminator)
            test_itr_num += 1.

            for k in range(1):
                mask = np.zeros([512, 512])
                if not no_figure:
                    fig, ax = plt.subplots(3, 5, figsize=(16, 16))
                    in_img = data_batch
                    ge_img = np.zeros([1, 512, 512, 1])
                    for n in range(8):
                        for m in range(8):
                            ge_img[0, m * 64:m * 64 + 64, n * 64:n * 64 + 64, 0] = gen_img_0[8 * n + m, :, :, 0]

                    diff_img_tmp = ssim_valid(in_img, (ge_img + 1.) / 2.)
                    diff_img_1 = diff_img_tmp[0]
                    diff_img_2 = diff_img_tmp[1]
                    diff_img_3 = diff_img_tmp[2]
                    q_diff_img_1 = np.where(diff_img_1 < 1.0, 1, 0)
                    q_diff_img_2 = np.where(diff_img_2 < 1.0, 1, 0)
                    q_diff_img_3 = np.where(diff_img_3 < 1.0, 1, 0)

                    if not no_figure:
                        ax[0, 0].imshow(in_img[k, :, :, 0], cmap='gray')
                        ax[0, 1].imshow((ge_img[k, :, :, 0] + 1.) / 2., cmap='gray')
                        ax[0, 2].imshow(diff_img_1, cmap='jet')
                        ax[0, 3].imshow(q_diff_img_1, cmap='gray', vmin=0, vmax=1)
                        ax[1, 2].imshow(diff_img_2, cmap='jet')
                        ax[1, 3].imshow(q_diff_img_2, cmap='gray', vmin=0, vmax=1)
                        ax[2, 2].imshow(diff_img_3, cmap='jet')
                        ax[2, 3].imshow(q_diff_img_3, cmap='gray', vmin=0, vmax=1)
                        ax[0, 4].imshow(label_batch[k, :, :, 0], cmap='gray')

                    q_diff_img_1 = imresize(q_diff_img_1, (512, 512), 'nearest')
                    q_diff_img_2 = imresize(q_diff_img_2, (512, 512), 'nearest')
                    q_diff_img_3 = imresize(q_diff_img_3, (512, 512), 'nearest')
                    q_diff_img = np.logical_and(np.logical_and(q_diff_img_1, q_diff_img_2), q_diff_img_3)
                    ax[1, 4].imshow(q_diff_img, cmap='gray', vmin=0, vmax=1)

                total_num += 1

                if not no_figure:
                    fig.savefig('./class_%d_patch/epoch_%d_%d_%d_result.png' % (class_num, epoch + 1, test_itr_num, k))

                    plt.close(fig)

            if not no_figure:
                print(datetime.datetime.now())

            if test_itr_num == 12:
                no_figure = True

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print('Loss for Generator is {}'.format(gen_loss / itr_num))
        epoch_iou = total_iou / total_num
        if best_iou < epoch_iou:
            best_iou = epoch_iou
            best_epoch = epoch
        print('iou is %f' % epoch_iou)
        print('best iou is %f at epoch %d' % (best_iou, best_epoch))


i = 5

d_1 = Discriminator(4)
g_1 = Generator(4)

if not os.path.isdir('./class_%d_patch' % i):
    os.mkdir('./class_%d_patch' % i)

db_helper = DBHelper('/home/ybrain/sangmin/DAGM_KaggleUpload/Class%d/' % i, BATCH_SIZE)
# db_helper = DBHelper('/Users/sangminlee/Documents/Sangmin/DB/DAGM_KaggleUpload/Class%d/' % i, BATCH_SIZE)

discriminator_optimizer_0 = tf.keras.optimizers.Adam(2e-4)

discriminator_optimizer = [discriminator_optimizer_0]

EPOCHS = 50

g = [g_1]
d = [d_1]

train(db_helper, EPOCHS, g, d, discriminator_optimizer, i)

print('')
