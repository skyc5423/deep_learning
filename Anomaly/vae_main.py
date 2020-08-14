from db_helper import DBHelper
# from network import Generator, vgg_layers
# from network import Discriminator
from vae_network import CVAE
import matplotlib
import numpy as np
import cv2

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import os, time

BATCH_SIZE = 16
seed = tf.random.normal([16, 100])

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


@tf.function
def compute_loss(model, x, noise_x):
    mean, logvar = model.encode(noise_x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=tf.cast(x, tf.float32))
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def compute_apply_gradients(model, x, noise_x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, noise_x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def pyramid_down(batch_image):
    for i in range(batch_image.shape[0]):
        if i == 0:
            ori = np.expand_dims(np.expand_dims(cv2.pyrDown(batch_image[i, :, :, 0]), 0), 3)
            norm = np.expand_dims(np.expand_dims(cv2.pyrDown(batch_image[i, :, :, 1]), 0), 3)
            rtn = np.concatenate([ori, norm], axis=3)
        else:
            ori = np.expand_dims(np.expand_dims(cv2.pyrDown(batch_image[i, :, :, 0]), 0), 3)
            norm = np.expand_dims(np.expand_dims(cv2.pyrDown(batch_image[i, :, :, 1]), 0), 3)
            tmp = np.concatenate([ori, norm], axis=3)
            rtn = np.concatenate([rtn, tmp], axis=0)
    return rtn


def salt_and_pepper(batch_images):
    # np.random.randint(0, 51*512)
    for i in range(batch_images.shape[0]):
        ns = np.random.randint(512 * 512, size=int(512 * 512 * 0.03))
        for n in ns:
            batch_images[i, int(n / 512), n % 512, 0] = 0.
        ns = np.random.randint(512 * 512, size=int(512 * 512 * 0.03))
        for n in ns:
            batch_images[i, int(n / 512), n % 512, 0] = 1.
    return batch_images


def normalize_weber(batch_images):
    tmp = np.log(batch_images + 1) / np.log(2)

    return np.concatenate([batch_images, tmp], axis=3)


@tf.function
def train_step(noise_images, images,
               noise_image_1, image_1,
               noise_image_2, image_2,
               noise_image_3, image_3,
               cvae, optimizer):
    with tf.GradientTape() as disc_tape:
        compute_apply_gradients(cvae[0], images, noise_images, optimizer[0])
    # with tf.GradientTape() as disc_tape:
    #     compute_apply_gradients(cvae[1], image_1, noise_image_1, optimizer[1])
    # with tf.GradientTape() as disc_tape:
    #     compute_apply_gradients(cvae[2], image_2, noise_image_2, optimizer[2])
    # with tf.GradientTape() as disc_tape:
    #     compute_apply_gradients(cvae[3], image_3, noise_image_3, optimizer[3])

    return


@tf.function
def test_step(images, images_1, images_2, images_3, cvae):
    with tf.GradientTape() as disc_tape:
        generated_images = cvae[0](images, training=False)
    with tf.GradientTape() as disc_tape:
        generated_images_1 = cvae[1](images_1, training=False)
    with tf.GradientTape() as disc_tape:
        generated_images_2 = cvae[2](images_2, training=False)
    with tf.GradientTape() as disc_tape:
        generated_images_3 = cvae[3](images_3, training=False)

    return generated_images, generated_images_1, generated_images_2, generated_images_3


def train(dataset, epochs, cvae, discriminator_optimizer, class_num):
    for epoch in range(epochs):
        start = time.time()
        gen_loss, dis_loss, itr_num = 0, 0, 0

        for x_batch, y_batch in dataset.train_ds:
            if not x_batch.shape[0] == BATCH_SIZE:
                break
            data_batch = db_helper.get_data(x_batch, y_batch)
            noise_data_batch = salt_and_pepper(data_batch)
            norm_noise_data_batch = normalize_weber(noise_data_batch)
            noise_image_1 = pyramid_down(norm_noise_data_batch)
            noise_image_2 = pyramid_down(noise_image_1)
            noise_image_3 = pyramid_down(noise_image_2)
            norm_data_batch = normalize_weber(data_batch)
            image_1 = pyramid_down(norm_data_batch)
            image_2 = pyramid_down(image_1)
            image_3 = pyramid_down(image_2)
            train_step(norm_noise_data_batch[:, :, :, 0:1], norm_data_batch[:, :, :, 0:1],
                       noise_image_1[:, :, :, 0:1], image_1[:, :, :, 0:1],
                       noise_image_2[:, :, :, 0:1], image_2[:, :, :, 0:1],
                       noise_image_3[:, :, :, 0:1], image_3[:, :, :, 0:1],
                       cvae, discriminator_optimizer)
            # gen_loss += loss_0 / (512 * 512)
            itr_num += 1.

        test_itr_num = 0.

        for x_batch, y_batch in dataset.test_ds:
            if not x_batch.shape[0] == BATCH_SIZE:
                break
            data_batch = db_helper.get_data(x_batch, y_batch)
            norm_data_batch = normalize_weber(data_batch)
            data_batch_ds_1 = pyramid_down(norm_data_batch)
            data_batch_ds_2 = pyramid_down(data_batch_ds_1)
            data_batch_ds_3 = pyramid_down(data_batch_ds_2)
            gen_img_0, gen_img_1, gen_img_2, gen_img_3 = test_step(norm_data_batch[:, :, :, 0:1],
                                                                   data_batch_ds_1[:, :, :, 0:1],
                                                                   data_batch_ds_2[:, :, :, 0:1],
                                                                   data_batch_ds_3[:, :, :, 0:1],
                                                                   cvae)
            # g_l, d_l = train_step(data_batch, BATCH_SIZE, generator, discriminator, generator_optimizer, discriminator_optimizer)
            # gen_loss += loss
            # dis_loss += d_l
            test_itr_num += 1.

            # out = generator(seed, training=False)['feature_6']
            for n in range(4):
                for m in range(4):
                    mask = np.ones([512, 512])
                    mask_0_1 = np.ones([512, 512])
                    mask_1_2 = np.ones([512, 512])
                    mask_2_3 = np.ones([512, 512])
                    fig, ax = plt.subplots(4, 5, figsize=(16, 16))
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

                        diff_img = np.abs(in_img[n * 4 + m, :, :, 0] - (ge_img[n * 4 + m, :, :, 0] + 1.) / 2.)
                        ax[i, 0].imshow(in_img[n * 4 + m, :, :, 0], cmap='gray')
                        ax[i, 1].imshow((ge_img[n * 4 + m, :, :, 0] + 1.) / 2., cmap='gray')
                        ax[i, 2].imshow(diff_img, cmap='jet', vmin=0, vmax=1)
                        q_diff_img = ((diff_img + 0.8).astype(np.int8)).astype(np.float32)
                        ax[i, 3].imshow(q_diff_img, cmap='gray', vmin=0, vmax=1)

                        for tmp in range(i):
                            q_diff_img = cv2.pyrUp(q_diff_img)

                        mask *= q_diff_img

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

                    ax[0, 4].imshow(mask, cmap='gray', vmin=0, vmax=1)
                    ax[1, 4].imshow(mask_0_1, cmap='gray', vmin=0, vmax=1)
                    ax[2, 4].imshow(mask_1_2, cmap='gray', vmin=0, vmax=1)
                    ax[3, 4].imshow(mask_2_3, cmap='gray', vmin=0, vmax=1)

                    fig.savefig('./class_%d/epoch_%d_%d_%d_result.png' % (class_num, epoch + 1, test_itr_num, 4 * n + m))

                    plt.close(fig)

            if test_itr_num == 2:
                break

        # 15 에포크가 지날 때마다 모델을 저장합니다.
        # if (epoch + 1) % 15 == 0:
        #     checkpoint.save(file_prefix=checkpoint_prefix)

        # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print('Loss for Generator is {}'.format(gen_loss / itr_num))
        print('Loss for Discriminator is {}'.format(dis_loss / itr_num))


for i in range(4, 11):
    cvae_1 = CVAE((512, 512, 1), 1000)
    cvae_2 = CVAE((256, 256, 1), 500)
    cvae_3 = CVAE((128, 128, 1), 250)
    cvae_4 = CVAE((64, 64, 1), 100)

    # db_helper = FoodDBHelper('/home/ybrain/sangmin/food_dataset/preprocessed_224_224', BATCH_SIZE)
    if not os.path.isdir('./class_%d' % i):
        os.mkdir('./class_%d' % i)

    # db_helper = DBHelper('/home/ybrain/sangmin/DAGM_KaggleUpload/Class%d/' % i, BATCH_SIZE)
    db_helper = DBHelper('/Users/sangminlee/Documents/Sangmin/DB/DAGM_KaggleUpload/Class%d/' % i, BATCH_SIZE)

    discriminator_optimizer_0 = tf.keras.optimizers.Adam(1e-3)
    discriminator_optimizer_1 = tf.keras.optimizers.Adam(1e-3)
    discriminator_optimizer_2 = tf.keras.optimizers.Adam(1e-3)
    discriminator_optimizer_3 = tf.keras.optimizers.Adam(1e-3)

    discriminator_optimizer = [discriminator_optimizer_0, discriminator_optimizer_1, discriminator_optimizer_2, discriminator_optimizer_3]

    EPOCHS = 100

    cvae = [cvae_1, cvae_2, cvae_3, cvae_4]

    train(db_helper, EPOCHS, cvae, discriminator_optimizer, i)

    print('')
