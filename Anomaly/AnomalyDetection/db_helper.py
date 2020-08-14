import os
import numpy as np
from abs_db_helper import AbsDBHelper
from PIL import Image


class DBHelper(AbsDBHelper):
    def load_data_list(self):

        x_train = np.array([])
        y_train = np.array([])
        x_test = np.array([])
        y_test = np.array([])

        self.file_num = 0

        for fold_label in os.listdir(self.db_path):
            if not os.path.isdir(os.path.join(self.db_path, fold_label)):
                continue
            file_list = os.listdir(os.path.join(self.db_path, fold_label))
            for file_name in file_list:
                x_train = np.concatenate([x_train, np.array([os.path.join(fold_label, file_name)])])
                y_train = np.concatenate([y_train, np.array([fold_label])])
                # x_test = np.concatenate([x_test, np.array([os.path.join(fold_label, file_list[i])])])
                # y_test = np.concatenate([y_test, np.array([fold_label])])
                self.file_num += 1

            # if self.file_num > 1000:
            #     break

        return x_train, y_train, x_test, y_test

    def get_img_by_file_name(self, file_name):
        file_path = os.path.join(self.db_path, file_name)
        f_img = Image.open(file_path)
        # f_img = f_img.resize((28, 28))
        f_img = f_img.resize((224, 224))
        img_data = np.array(f_img)
        if img_data.shape.__len__() == 2:
            img_data = np.expand_dims(img_data, 2)
            img_data = np.tile(img_data, [1, 1, 3])

        # if not img_data.shape == (256, 256, 3):
        #     print(img_data.shape)

        return img_data / 255.

    @staticmethod
    def get_label_array_from_text(text_label):
        label_array = np.zeros([4])
        if text_label == 'EEG':
            label_array[0] = 1.
        elif text_label == 'EMG':
            label_array[1] = 1.
        elif text_label == 'HEOG':
            label_array[2] = 1.
        elif text_label == 'VEOG':
            label_array[3] = 1.
        else:
            print('Wrong Label: %s' % text_label)
            return

        return label_array

    def get_data(self, x_batch, y_batch):
        for batch_idx in range(x_batch.shape[0]):
            file_name = x_batch[batch_idx].numpy().decode('utf-8')
            tmp_data = self.get_img_by_file_name(file_name)
            tmp_data = np.expand_dims(tmp_data, axis=0)
            if batch_idx == 0:
                data_batch = tmp_data
            else:
                data_batch = np.concatenate([data_batch, tmp_data], axis=0)

            # text_label = y_batch[batch_idx].numpy().decode('utf-8')
            # tmp_label = np.expand_dims(self.get_label_array_from_text(text_label), axis=0)
            # if batch_idx == 0:
            #     label_batch = tmp_label
            # else:
            #     label_batch = np.concatenate([label_batch, tmp_label], axis=0)

        return data_batch
