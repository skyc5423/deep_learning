from db_helper import DBHelper
from network import Generator, vgg_layers
from network import Discriminator
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import os, time

BATCH_SIZE = 32
seed = tf.random.normal([16, 100])

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

# vgg_output = vgg_layers(style_layers)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_image):
    # real_vgg = vgg_output(real_image)
    # fake_vgg = vgg_output(fake_image)

    # l1 = 0.
    # for i in range(real_vgg.__len__()):
    #     l1 += tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.abs(real_vgg[i] - fake_vgg[i]))))) * i
    #
    # return l1
    return cross_entropy(tf.ones_like(fake_image), fake_image)


# @tf.function
# def train_step(images, batch_size, generator, generator_optimizer):
#     noise = tf.random.normal([batch_size, 100])
#
#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         generated_images = generator(noise, training=True)
#         gen_loss = generator_loss(images, generated_images['feature_6'])
#         gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
#         generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#
#     return gen_loss


@tf.function
def train_step(images, batch_size, generator, discriminator, generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)['out']
        fake_output = discriminator(generated_images['feature_6'], training=True)['out']
        disc_loss = discriminator_loss(real_output, fake_output)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    for i in range(16):
        with tf.GradientTape() as gen_tape:
            generated_images = generator(noise, training=True)

            fake_output = discriminator(generated_images['feature_6'], training=True)['out']
            gen_loss = generator_loss(fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return gen_loss, disc_loss


def train(dataset, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint):
    for epoch in range(epochs):
        start = time.time()
        gen_loss, dis_loss, itr_num = 0, 0, 0

        for x_batch, y_batch in dataset.train_ds:
            if not x_batch.shape[0] == BATCH_SIZE:
                break
            data_batch = db_helper.get_data(x_batch, y_batch)
            g_l, d_l = train_step(data_batch, BATCH_SIZE, generator, discriminator, generator_optimizer, discriminator_optimizer)
            gen_loss += g_l
            dis_loss += d_l
            itr_num += 1.

        # 15 에포크가 지날 때마다 모델을 저장합니다.
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        fig, ax = plt.subplots(4, 4, figsize=(16, 16))

        out = generator(seed, training=False)['feature_6']

        for i in range(4):
            for j in range(4):
                ax[i, j].imshow((out[i * 4 + j] + 1.) / 2.)

        fig.savefig('./result/epoch_%d_result.png' % (epoch + 1))

        plt.close(fig)

        # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print('Loss for Generator is {}'.format(gen_loss / itr_num))
        print('Loss for Discriminator is {}'.format(dis_loss / itr_num))


d = Discriminator()

g = Generator()

# db_helper = FoodDBHelper('/home/ybrain/sangmin/food_dataset/preprocessed_224_224', BATCH_SIZE)
db_helper = DBHelper('/home/ybrain/sangmin/ybrain_db/topo_artifact_db/EEGArtifact/topomap', BATCH_SIZE)

generator_optimizer = tf.keras.optimizers.Adam(1e-3)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-3)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=g,
                                 discriminator=d)

EPOCHS = 100

train(db_helper, EPOCHS, g, d, generator_optimizer, discriminator_optimizer, checkpoint)

print('')
