from db_helper import DBHelper
# from network import Generator, vgg_layers
# from network import Discriminator
from sklearn.metrics import roc_curve, auc
from network_v2 import Generator, vgg_layers
from network_v2 import Discriminator
from scipy.misc import imresize
import matplotlib
import numpy as np
import cv2

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import os, time

BATCH_SIZE = 4


def autoencoder_loss(input_img, output_img):
    return tf.nn.l2_loss(tf.dtypes.cast(input_img[:, :, :], tf.float32) - (output_img[:, :, :] + 1) / 2.)


def ssim_loss(input_img, output_img):
    img_size = input_img.shape[1]
    kernel_size = 41
    step_size = 11
    c1 = 0.01
    c2 = 0.03
    total_step_size = int((img_size - kernel_size + 1) / step_size)
    total_ssim = 0.
    input_img = tf.cast(input_img, tf.float32)
    for i in range(BATCH_SIZE):
        for n in range(total_step_size):
            for m in range(total_step_size):
                patch_input = tf.slice(input_img, [0, m * step_size, m * step_size, 0], [1, kernel_size, kernel_size, 1])
                patch_output = tf.slice(output_img, [0, m * step_size, m * step_size, 0], [1, kernel_size, kernel_size, 1])

                mean_input = tf.reduce_mean(tf.squeeze(patch_input))
                var_input = tf.math.square(tf.math.reduce_std(tf.squeeze(patch_input)))

                mean_output = tf.reduce_mean(tf.squeeze(patch_output))
                var_output = tf.math.square(tf.math.reduce_std(tf.squeeze(patch_output)))

                covar = tf.reduce_sum((patch_input - mean_input) * (patch_output - mean_output)) / (img_size * img_size - 1)

                ssim = (2 * mean_input * mean_output + c1) * (2 * covar + c2) / (tf.math.square(mean_input) + tf.math.square(mean_output) + c1) / (var_input + var_output + c2)

                total_ssim += ssim
    return total_ssim


def iou(label, mask):
    return np.sum(np.logical_and(label, mask)) / np.sum(np.logical_or(label, mask))


def pyramid_down(batch_image):
    for i in range(batch_image.shape[0]):
        if i == 0:
            # ori = np.expand_dims(np.expand_dims(cv2.pyrDown(batch_image[i, :, :, 0]), 0), 3)
            # norm = np.expand_dims(np.expand_dims(cv2.pyrDown(batch_image[i, :, :, 1]), 0), 3)
            # rtn = np.concatenate([ori, norm], axis=3)
            rtn = np.expand_dims(cv2.pyrDown(batch_image[i, :, :]), 0)

        else:
            # ori = np.expand_dims(np.expand_dims(cv2.pyrDown(batch_image[i, :, :, 0]), 0), 3)
            # norm = np.expand_dims(np.expand_dims(cv2.pyrDown(batch_image[i, :, :, 1]), 0), 3)
            # tmp = np.concatenate([ori, norm], axis=3)
            # rtn = np.concatenate([rtn, tmp], axis=0)
            rtn = np.concatenate([rtn, np.expand_dims(cv2.pyrDown(batch_image[i, :, :]), 0)], axis=0)
    return np.expand_dims(rtn, 3)


def salt_and_pepper(batch_images):
    # np.random.randint(0, 51*512)
    for i in range(batch_images.shape[0]):
        ns = np.random.randint(512 * 512, size=int(512 * 512 * 0.05))
        for n in ns:
            batch_images[i, int(n / 512), n % 512, 0] = 0.
        ns = np.random.randint(512 * 512, size=int(512 * 512 * 0.05))
        for n in ns:
            batch_images[i, int(n / 512), n % 512, 0] = 1.

    np.zeros([])

    return batch_images


# def normalize_weber(batch_images):
#     tmp = np.log(batch_images + 1) / np.log(2)
#
#     return np.concatenate([batch_images, tmp], axis=3)


@tf.function
def train_step(noise_images, images,
               noise_image_1, image_1,
               noise_image_2, image_2,
               noise_image_3, image_3,
               generator, discriminator, discriminator_optimizer):
    with tf.GradientTape() as disc_tape:
        lat_var = discriminator[0](noise_images, training=True)
        generated_images = generator[0](lat_var, training=True)['out']
        disc_loss = autoencoder_loss(images, generated_images)
        tv = discriminator[0].trainable_variables + generator[0].trainable_variables
        gradients_of_discriminator = disc_tape.gradient(disc_loss, tv)
        discriminator_optimizer[0].apply_gradients(zip(gradients_of_discriminator, tv))
    with tf.GradientTape() as disc_tape:
        lat_var = discriminator[1](noise_image_1, training=True)
        generated_images = generator[1](lat_var, training=True)['out']
        disc_loss = autoencoder_loss(image_1, generated_images)
        tv = discriminator[1].trainable_variables + generator[1].trainable_variables
        gradients_of_discriminator = disc_tape.gradient(disc_loss, tv)
        discriminator_optimizer[1].apply_gradients(zip(gradients_of_discriminator, tv))
    with tf.GradientTape() as disc_tape:
        lat_var = discriminator[2](noise_image_2, training=True)
        generated_images = generator[2](lat_var, training=True)['out']
        disc_loss = autoencoder_loss(image_2, generated_images)
        tv = discriminator[2].trainable_variables + generator[2].trainable_variables
        gradients_of_discriminator = disc_tape.gradient(disc_loss, tv)
        discriminator_optimizer[2].apply_gradients(zip(gradients_of_discriminator, tv))
    with tf.GradientTape() as disc_tape:
        lat_var = discriminator[3](noise_image_3, training=True)
        generated_images = generator[3](lat_var, training=True)['out']
        disc_loss = autoencoder_loss(image_3, generated_images)
        tv = discriminator[3].trainable_variables + generator[3].trainable_variables
        gradients_of_discriminator = disc_tape.gradient(disc_loss, tv)
        discriminator_optimizer[3].apply_gradients(zip(gradients_of_discriminator, tv))

    return disc_loss, generated_images


@tf.function
def test_step(images, images_1, images_2, images_3, generator, discriminator):
    with tf.GradientTape() as disc_tape:
        lat_var = discriminator[0](images, training=True)
        tmp = generator[0](lat_var, training=True)
        generated_images = tmp['out']
    with tf.GradientTape() as disc_tape:
        lat_var = discriminator[1](images_1, training=True)
        generated_images_1 = generator[1](lat_var, training=True)['out']
    with tf.GradientTape() as disc_tape:
        lat_var = discriminator[2](images_2, training=True)
        generated_images_2 = generator[2](lat_var, training=True)['out']
    with tf.GradientTape() as disc_tape:
        lat_var = discriminator[3](images_3, training=True)
        generated_images_3 = generator[3](lat_var, training=True)['out']

    return generated_images, generated_images_1, generated_images_2, generated_images_3


def train(dataset, epochs, generator, discriminator, discriminator_optimizer, class_num):
    best_iou = 0.
    best_epoch = -1
    best_epoch = -1
    for epoch in range(epochs):
        start = time.time()
        gen_loss, dis_loss, itr_num = 0, 0, 0

        for x_batch, y_batch in dataset.train_ds:
            if not x_batch.shape[0] == BATCH_SIZE:
                break
            data_batch, label_batch = db_helper.get_data(x_batch, y_batch)
            noise_data_batch = salt_and_pepper(data_batch)
            # norm_noise_data_batch = normalize_weber(noise_data_batch)
            noise_image_1 = pyramid_down(noise_data_batch)
            noise_image_2 = pyramid_down(noise_image_1)
            noise_image_3 = pyramid_down(noise_image_2)
            # norm_data_batch = normalize_weber(data_batch)
            image_1 = pyramid_down(data_batch)
            image_2 = pyramid_down(image_1)
            image_3 = pyramid_down(image_2)
            loss_0, gen_img_0 = train_step(noise_data_batch[:, :, :], data_batch[:, :, :],
                                           noise_image_1[:, :, :], image_1[:, :, :],
                                           noise_image_2[:, :, :], image_2[:, :, :],
                                           noise_image_3[:, :, :], image_3[:, :, :],
                                           generator, discriminator, discriminator_optimizer)
            gen_loss += loss_0 / (512 * 512)
            # gen_loss += loss_1 / (256 * 256)
            # gen_loss += loss_2 / (128 * 128)
            # dis_loss += d_l
            itr_num += 1.

        test_itr_num = 0.

        # if epoch < 10:
        #     continue

        no_figure = False

        total_iou = 0
        total_num = 0

        for x_batch, y_batch in dataset.test_ds:

            if not x_batch.shape[0] == BATCH_SIZE:
                break
            data_batch, label_batch = db_helper.get_data(x_batch, y_batch)
            # norm_data_batch = normalize_weber(data_batch)
            data_batch_ds_1 = pyramid_down(data_batch)
            data_batch_ds_2 = pyramid_down(data_batch_ds_1)
            data_batch_ds_3 = pyramid_down(data_batch_ds_2)
            gen_img_0, gen_img_1, gen_img_2, gen_img_3 = test_step(data_batch[:, :, :],
                                                                   data_batch_ds_1[:, :, :],
                                                                   data_batch_ds_2[:, :, :],
                                                                   data_batch_ds_3[:, :, :],
                                                                   generator, discriminator)
            # g_l, d_l = train_step(data_batch, BATCH_SIZE, generator, discriminator, generator_optimizer, discriminator_optimizer)
            # gen_loss += loss
            # dis_loss += d_l
            test_itr_num += 1.

            # out = generator(seed, training=False)['feature_6']
            for m in range(4):
                mask = np.zeros([512, 512])
                mask_0_1 = np.ones([512, 512])
                mask_1_2 = np.ones([512, 512])
                mask_2_3 = np.ones([512, 512])
                if not no_figure:
                    fig, ax = plt.subplots(4, 6, figsize=(16, 16))
                for i in range(4):
                    if i == 0:
                        in_img = data_batch
                        ge_img = gen_img_0
                    elif i == 1:
                        in_img = data_batch_ds_1
                        ge_img = gen_img_1
                    elif i == 2:
                        in_img = data_batch_ds_2
                        ge_img = gen_img_2
                    elif i == 3:
                        in_img = data_batch_ds_3
                        ge_img = gen_img_3

                    diff_img = np.abs(in_img[m, :, :, 0] - (ge_img[m, :, :, 0] + 1.) / 2.)
                    threshold = 1 - (np.mean(diff_img) + 2.5 * np.std(diff_img))
                    threshold = 0.97
                    q_diff_img = ((diff_img + threshold).astype(np.int8)).astype(np.float32)

                    if not no_figure:
                        ax[i, 0].imshow(in_img[m, :, :, 0], cmap='gray')
                        ax[i, 1].imshow((ge_img[m, :, :, 0] + 1.) / 2., cmap='gray')
                        ax[i, 2].imshow(diff_img, cmap='jet', vmin=0, vmax=0.1)
                        ax[i, 3].imshow(q_diff_img, cmap='gray', vmin=0, vmax=1)

                    # for tmp in range(i):
                    #     q_diff_img = cv2.pyrUp(q_diff_img)

                    q_diff_img = imresize(q_diff_img, (512, 512), 'nearest')

                    if i == 0:
                        mask_0_1 *= q_diff_img
                    elif i == 1:
                        mask_0_1 *= q_diff_img
                        mask_1_2 *= q_diff_img
                    elif i == 2:
                        mask_1_2 *= q_diff_img
                        mask_2_3 *= q_diff_img
                    elif i == 3:
                        mask_2_3 *= q_diff_img

                mask = np.logical_or(mask_0_1, mask_1_2)
                mask = np.logical_or(mask, mask_2_3)

                if not no_figure:
                    ax[0, 5].imshow(label_batch[m, :, :, 0], cmap='gray', vmin=0, vmax=1)

                    ax[0, 4].imshow(mask, cmap='gray', vmin=0, vmax=1)
                    ax[1, 4].imshow(mask_0_1, cmap='gray', vmin=0, vmax=1)
                    ax[2, 4].imshow(mask_1_2, cmap='gray', vmin=0, vmax=1)
                    ax[3, 4].imshow(mask_1_2, cmap='gray', vmin=0, vmax=1)

                total_iou += iou(label_batch[m, :, :, 0], mask)
                total_num += 1

                # false_positive_rate_F, true_positive_rate_F, thresholds_F = roc_curve(np.reshape(label_batch, -1), np.reshape(mask, -1))
                # tmp = auc(false_positive_rate_F, true_positive_rate_F)
                if not no_figure:
                    fig.savefig('./class_%d_v2/epoch_%d_%d_%d_result.png' % (class_num, epoch + 1, test_itr_num, m))

                    plt.close(fig)

            if test_itr_num == 3:
                no_figure = True
                # break

        # 15 에포크가 지날 때마다 모델을 저장합니다.
        # if (epoch + 1) % 15 == 0:
        #     checkpoint.save(file_prefix=checkpoint_prefix)

        # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print('Loss for Generator is {}'.format(gen_loss / itr_num))
        epoch_iou = total_iou / total_num
        if best_iou < epoch_iou:
            best_iou = epoch_iou
            best_epoch = epoch
        print('iou is %f' % epoch_iou)
        print('best iou is %f at epoch %d' % (best_iou, best_epoch))


i = 7

d_1 = Discriminator(4)
g_1 = Generator(4)

d_2 = Discriminator(4)
g_2 = Generator(4)

d_3 = Discriminator(4)
g_3 = Generator(4)

d_4 = Discriminator(4)
g_4 = Generator(4)
# db_helper = FoodDBHelper('/home/ybrain/sangmin/food_dataset/preprocessed_224_224', BATCH_SIZE)
if not os.path.isdir('./class_%d_v2' % i):
    os.mkdir('./class_%d_v2' % i)

db_helper = DBHelper('/home/ybrain/sangmin/DAGM_KaggleUpload/Class%d/' % i, BATCH_SIZE)
# db_helper = DBHelper('/Users/sangminlee/Documents/Sangmin/DB/DAGM_KaggleUpload/Class%d/' % i, BATCH_SIZE)

discriminator_optimizer_0 = tf.keras.optimizers.Adam(2e-4)
discriminator_optimizer_1 = tf.keras.optimizers.Adam(2e-4)
discriminator_optimizer_2 = tf.keras.optimizers.Adam(2e-4)
discriminator_optimizer_3 = tf.keras.optimizers.Adam(2e-4)

discriminator_optimizer = [discriminator_optimizer_0, discriminator_optimizer_1, discriminator_optimizer_2, discriminator_optimizer_3]

# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer[0],
#                                  generator=g_1,
#                                  discriminator=d_1)

EPOCHS = 50

g = [g_1, g_2, g_3, g_4]
d = [d_1, d_2, d_3, d_4]

train(db_helper, EPOCHS, g, d, discriminator_optimizer, i)

print('')
