import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, random


from dc_gan import *
from db_helper import DBHelper

generator_general = Generator()
generator_celeb = Generator()

discriminator_general = Discriminator()
discriminator_celeb = Discriminator()

trainer = CycleGanTrainer(generator_general, generator_celeb, discriminator_general, discriminator_celeb)

EPOCH_NUM = 100
BATCH_SIZE = 2
DIS_TRAIN_NUM = 1
GEN_TRAIN_NUM = 1
load_path = None
output_path = './result/cycle_gan'

dataset = DBHelper('./downloads/bing2/preprocessed_256', 2)
dataset_general = DBHelper('./downloads/bing3/preprocessed_256_general', 2)
dataset.train_ds.shuffle(50000)
dataset_general.train_ds.shuffle(1000)

# combined_dataset = dataset.train_ds.concatenate(dataset_general.train_ds)
# combined_dataset.shuffle(10000)

# if load_path is not None:
#     generator.load_weights('./%s/model_g' % load_path)
#     discriminator.load_weights('./%s/model_d' % load_path)

for epoch in range(EPOCH_NUM):

    gen_f_loss = 0
    gen_g_loss = 0
    dis_f_loss = 0
    dis_g_loss = 0
    iter_num = 0

    for general_batch in dataset_general.train_ds:
        celeb_batch = list(dataset.train_ds)[iter_num]
        general_data_batch = dataset_general.get_data(general_batch, augment=True)
        celeb_data_batch = dataset.get_data(celeb_batch, augment=True)
        cur_gen_f_loss, cur_gen_g_loss, cur_dis_f_loss, cur_dis_g_loss = trainer.train(general_data_batch, celeb_data_batch)
        gen_f_loss += cur_gen_f_loss
        gen_g_loss += cur_gen_g_loss
        dis_f_loss += cur_dis_f_loss
        dis_g_loss += cur_dis_g_loss
        iter_num += 1
        break

    for test_batch in dataset_general.train_ds:
        test_data_batch = dataset.get_data(test_batch)
        recon_data = generator_general.test(test_data_batch)
        recon_data = recon_data.numpy()
        cycled_data = generator_celeb.test(recon_data)
        cycled_data = cycled_data.numpy()
        break

    cycled_data = (cycled_data + 1.) / 2.
    recon_data = (recon_data + 1.) / 2.
    shown_x_batch = (test_data_batch + 1.) / 2.

    print('epoch-%02d: f_loss (gen, dis)=(%.3f, %.3f), g_loss (gen, dis)=(%.3f, %.3f)' % (
        epoch, gen_f_loss / iter_num, dis_f_loss / iter_num, gen_g_loss / iter_num, dis_g_loss / iter_num))

    fig, ax = plt.subplots(BATCH_SIZE, 3, figsize=(6, 6))
    for i in range(BATCH_SIZE):
        ax[i, 0].imshow(shown_x_batch[i])
        ax[i, 1].imshow(recon_data[i])
        ax[i, 2].imshow(cycled_data[i])

    if not os.path.isdir('./%s' % output_path):
        os.mkdir('./%s' % output_path)
    fig.savefig('./%s/result_%d.png' % (output_path, epoch))
    plt.close(fig)
