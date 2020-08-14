import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, random

from dc_gan import *
from db_helper import DBHelper

generator = Generator()
trainer = AutoEncoderTrainer(generator)

EPOCH_NUM = 100
BATCH_SIZE = 1
DIS_TRAIN_NUM = 1
GEN_TRAIN_NUM = 1
load_path = None
output_path = './result/auto_encoder'

dataset = DBHelper('./downloads/bing2/preprocessed_256', 2)

# if load_path is not None:
#     generator.load_weights('./%s/model_g' % load_path)
#     discriminator.load_weights('./%s/model_d' % load_path)

for epoch in range(EPOCH_NUM):

    ae_loss = 0
    iter_num = 0

    for x_batch in dataset.train_ds:
        data_batch = dataset.get_data(x_batch)
        ae_loss += trainer.train(data_batch)
        iter_num += 1
        break

    recon_data = trainer.test(data_batch)
    recon_data = (recon_data + 1.) / 2.
    shown_x_batch = (x_batch + 1.) / 2.

    print('epoch-%02d: auto_encoder loss=%.3f' % (epoch, ae_loss / iter_num))

    fig, ax = plt.subplots(BATCH_SIZE, 2, figsize=(6, 6))
    for i in range(BATCH_SIZE):
        ax[i, 0].imshow(shown_x_batch[i])
        ax[i, 1].imshow(recon_data[i])

    if not os.path.isdir('./%s' % output_path):
        os.mkdir('./%s' % output_path)
    fig.savefig('./%s/result_%d.png' % (output_path, epoch))
    plt.close(fig)
