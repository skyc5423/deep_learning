import numpy as np
import matplotlib.pyplot as plt

from VAE.Network import AutoEncoder
from DataManager.db_helper import ImageDBHelper


def main():
    TRAIN_REC_DB_PATH = '/Users/sangminlee/Documents/Sangmin/DB/toy_data/rectangle_bgd'
    TRAIN_CIR_DB_PATH = '/Users/sangminlee/Documents/Sangmin/DB/toy_data/circle_bgd'
    TEST_REC_DB_PATH = '/Users/sangminlee/Documents/Sangmin/DB/toy_data/rectangle_bgd_test'
    TEST_CIR_DB_PATH = '/Users/sangminlee/Documents/Sangmin/DB/toy_data/circle_bgd_test'

    BATCH_SIZE = 4

    auto_encoder = AutoEncoder(BATCH_SIZE, 32)

    train_db = ImageDBHelper([TRAIN_REC_DB_PATH, TRAIN_CIR_DB_PATH], BATCH_SIZE)
    test_db = ImageDBHelper([TEST_REC_DB_PATH, TEST_CIR_DB_PATH], BATCH_SIZE)

    for epoch_idx in range(1, 301):
        total_num = 0
        loss_total = 0
        for data_name_batch, label_batch in train_db.train_ds:
            data_batch = train_db.get_data(data_name_batch, label_batch)

            loss_tmp = auto_encoder.train_(data_batch, epoch_idx)
            loss_total += loss_tmp

            total_num += data_batch.shape[0]
            # if total_num > 200:
            #     break

        print('epoch %02d, loss %.4f' % (epoch_idx, (loss_total / total_num)))

        for test_data_name_batch, test_label_batch in test_db.train_ds:
            test_data_batch = test_db.get_data(test_data_name_batch, test_label_batch)

            gen_cir, mean, std, z = auto_encoder(test_data_batch)
            fig, ax = plt.subplots(2, 4)
            for i in range(4):
                ax[0, i].imshow((test_data_batch[i] + 1) / 2, vmin=0, vmax=1)
                ax[1, i].imshow((gen_cir[i] + 1) / 2, vmin=0, vmax=1)
            fig.savefig('./result_x_to_y_%d.png' % epoch_idx)
            plt.close(fig)
            break

        # if epoch_idx % 50 == 0:
        #     auto_encoder.save('./epoch_%d'%epoch_idx)

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
