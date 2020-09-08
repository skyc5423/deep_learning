import numpy as np
import matplotlib.pyplot as plt

from CNN.Network import Network
from DataManager.db_helper import ImageDBHelper


def main():
    TRAIN_REC_DB_PATH = '/Users/sangminlee/Documents/Sangmin/DB/toy_data/rectangle'
    TRAIN_CIR_DB_PATH = '/Users/sangminlee/Documents/Sangmin/DB/toy_data/circle'
    TEST_REC_DB_PATH = '/Users/sangminlee/Documents/Sangmin/DB/toy_data/rectangle_test'
    TEST_CIR_DB_PATH = '/Users/sangminlee/Documents/Sangmin/DB/toy_data/circle_test'

    BATCH_SIZE = 16

    train_db = ImageDBHelper([TRAIN_REC_DB_PATH, TRAIN_CIR_DB_PATH], BATCH_SIZE)
    test_db = ImageDBHelper([TEST_REC_DB_PATH, TEST_CIR_DB_PATH], BATCH_SIZE)

    network = Network('name')

    for epoch_idx in range(1, 101):
        total_num = 0
        total_correct = 0
        total_loss = 0
        for data_name_batch, label_batch in train_db.train_ds:
            data_batch = train_db.get_data(data_name_batch, label_batch)

            total_loss += network.train_(data_batch, label_batch)
            train_pred = network(data_batch)

            total_correct += np.sum(np.where(np.argmax(train_pred, axis=1) == np.argmax(label_batch, axis=1), 1, 0))
            total_num += data_batch.shape[0]

        print('epoch %02d, loss %.4f, accr %.2f%%' % (epoch_idx, (total_loss / total_num), 100 * (total_correct / total_num)))

        test_total_num = 0
        test_total_correct = 0

        for test_data_name_batch, test_label_batch in test_db.train_ds:
            test_data_batch = test_db.get_data(test_data_name_batch, test_label_batch)

            test_pred = network(test_data_batch)
            test_total_correct += np.sum(np.where(np.argmax(test_pred, axis=1) == np.argmax(test_label_batch, axis=1), 1, 0))
            test_total_num += test_data_batch.shape[0]

        print('epoch %02d, test accr %.2f%%' % (epoch_idx, 100 * (test_total_correct / test_total_num)))


if __name__ == "__main__":
    main()
