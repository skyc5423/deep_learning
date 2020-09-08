import numpy as np
import matplotlib.pyplot as plt

from CycleGAN.GANTrainer import GANTrainer
from DataManager.db_helper import ImageDBHelper


def main():
    TRAIN_REC_DB_PATH = '/Users/sangminlee/Documents/Sangmin/DB/toy_data/rectangle'
    TRAIN_CIR_DB_PATH = '/Users/sangminlee/Documents/Sangmin/DB/toy_data/circle'
    TEST_REC_DB_PATH = '/Users/sangminlee/Documents/Sangmin/DB/toy_data/rectangle_test'
    TEST_CIR_DB_PATH = '/Users/sangminlee/Documents/Sangmin/DB/toy_data/circle_test'

    BATCH_SIZE = 4

    gan_trainer = GANTrainer(BATCH_SIZE)

    train_db = ImageDBHelper([TRAIN_REC_DB_PATH, TRAIN_CIR_DB_PATH], BATCH_SIZE)
    test_rec_db = ImageDBHelper(TEST_REC_DB_PATH, BATCH_SIZE)
    test_cir_db = ImageDBHelper(TEST_CIR_DB_PATH, BATCH_SIZE)

    for epoch_idx in range(1, 301):
        total_num = 0
        loss_gen_dis_y_to_x = 0
        loss_gen_dis_x_to_y = 0
        loss_gen_cycle_x = 0
        loss_gen_cycle_y = 0
        loss_discriminator_x = 0
        loss_discriminator_y = 0
        for data_name_batch, label_batch in train_db.train_ds:
            data_batch = train_db.get_data(data_name_batch, label_batch)

            tmp_loss_gen_dis_y_to_x, tmp_loss_gen_dis_x_to_y, tmp_loss_gen_cycle_x, tmp_loss_gen_cycle_y, tmp_loss_discriminator_x, tmp_loss_discriminator_y = \
                gan_trainer.train_(data_batch, label_batch)
            # total_loss_generator += tmp_loss_generator
            # total_loss_discriminator += tmp_loss_discriminator

            if tmp_loss_gen_dis_x_to_y is not None:
                loss_gen_dis_y_to_x += tmp_loss_gen_dis_y_to_x
                loss_gen_dis_x_to_y += tmp_loss_gen_dis_x_to_y
                loss_gen_cycle_x += tmp_loss_gen_cycle_x
                loss_gen_cycle_y += tmp_loss_gen_cycle_y
                loss_discriminator_x += tmp_loss_discriminator_x
                loss_discriminator_y += tmp_loss_discriminator_y

                total_num += data_batch.shape[0]
            # if total_num > 200:
            #     break

        print('epoch %02d, gen_dis_loss_y_to_x %.4f, gen_dis_loss_y_to_x %.4f' % (epoch_idx, (loss_gen_dis_y_to_x / total_num), (loss_gen_dis_x_to_y / total_num)))
        print('epoch %02d, gen_cycle_loss_x %.4f, gen_cycle_loss_y %.4f' % (epoch_idx, (loss_gen_cycle_x / total_num), (loss_gen_cycle_y / total_num)))
        print('epoch %02d, dis_loss_x %.4f, dis_loss_y %.4f' % (epoch_idx, (loss_discriminator_x / total_num), (loss_discriminator_y / total_num)))

        for test_data_name_batch in test_rec_db.train_ds:
            test_data_batch = test_rec_db.get_data(test_data_name_batch)

            gen_cir = gan_trainer.generator_x_to_y(test_data_batch)
            idt_rec = gan_trainer.generator_y_to_x(test_data_batch)
            cycle_rec = gan_trainer.generator_y_to_x(gen_cir)
            fig, ax = plt.subplots(4, 4)
            for i in range(4):
                ax[0, i].imshow((test_data_batch[i] + 1) / 2)
                ax[1, i].imshow((gen_cir[i] + 1) / 2)
                ax[2, i].imshow((idt_rec[i] + 1) / 2)
                ax[3, i].imshow((cycle_rec[i] + 1) / 2)
            fig.savefig('./result_x_to_y_%d.png' % epoch_idx)
            plt.close(fig)
            break

        for test_data_name_batch in test_cir_db.train_ds:
            test_data_batch = test_cir_db.get_data(test_data_name_batch)

            gen_rec = gan_trainer.generator_y_to_x(test_data_batch)
            idt_cir = gan_trainer.generator_x_to_y(test_data_batch)
            cycle_cir = gan_trainer.generator_x_to_y(gen_rec)
            fig, ax = plt.subplots(4, 4)
            for i in range(4):
                ax[0, i].imshow((test_data_batch[i] + 1) / 2)
                ax[1, i].imshow((gen_rec[i] + 1) / 2)
                ax[2, i].imshow((idt_cir[i] + 1) / 2)
                ax[3, i].imshow((cycle_cir[i] + 1) / 2)
            fig.savefig('./result_y_to_x_%d.png' % epoch_idx)
            plt.close(fig)
            break


def main_gif():
    import imageio
    from PIL import Image

    paths = []
    for i in range(1000):
        img = Image.open('./tmp_%d.png' % i)
        paths.append(img)
    imageio.mimsave('./result.gif', paths, fps=40)


if __name__ == "__main__":
    main()
    # main_gif()
