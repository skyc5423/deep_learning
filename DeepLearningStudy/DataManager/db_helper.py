import os
import numpy as np
from .abs_db_helper import AbsDBHelper
from PIL import Image, ImageFilter


class ImageDBHelper(AbsDBHelper):
    def load_data_list(self):

        x_train = np.array([])

        if isinstance(self.db_path, list):
            y_train = []
            label_num = len(self.db_path)
            cur_label = 0
            for path_list in self.db_path:
                file_list = os.listdir(path_list)
                label = np.zeros([label_num])
                label[cur_label] = 1
                for file_name in file_list:
                    if not file_name.endswith('png'):
                        continue
                    x_train = np.concatenate([x_train, np.array([file_name])])
                    y_train.append(label)
                cur_label += 1
            return x_train, np.array(y_train)

        else:
            file_list = os.listdir(self.db_path)

            for file_name in file_list:
                x_train = np.concatenate([x_train, np.array([file_name])])

            return x_train

    @staticmethod
    def rand_augment(img_data):
        flip_prob = np.random.uniform(0, 1, 1)
        trans_val = np.random.uniform(-0.2, 0.2, 2)
        scale_val = np.random.uniform(0.8, 1.2, 2)
        rotate_val = np.random.uniform(-30, 30, 1)
        blur_prob = np.random.uniform(0, 1, 1)

        if flip_prob > 0.5:
            img_data = img_data.transpose(method=Image.FLIP_LEFT_RIGHT)

        img_data = img_data.transform(img_data.size, Image.AFFINE, (scale_val[0], 0, int(trans_val[0] * img_data.size[0]), 0, scale_val[1], int(trans_val[1] * img_data.size[1])))
        img_data = img_data.rotate(angle=rotate_val)

        if blur_prob > 0.9:
            img_data = img_data.filter(filter=ImageFilter.BLUR)

        return img_data

    def get_img_by_file_name(self, file_name, label=None, augment=False):
        if label is not None:
            file_path = os.path.join(self.db_path[np.argmax(label)], file_name)

        else:
            file_path = os.path.join(self.db_path, file_name)
        f_img = Image.open(file_path)

        if augment:
            f_img = self.rand_augment(f_img)

        img_data = np.array(f_img)
        img_data = img_data / 255.

        img_data -= 0.5
        img_data *= 2.

        return img_data.astype(np.float32)

    def get_data(self, x_batch, y_batch=None, augment=False):
        for batch_idx in range(x_batch.shape[0]):
            file_name = x_batch[batch_idx].numpy().decode('utf-8')
            if y_batch is not None:
                tmp_data = self.get_img_by_file_name(file_name, y_batch[batch_idx], augment=augment)
            else:
                tmp_data = self.get_img_by_file_name(file_name, augment=augment)

            tmp_data = np.expand_dims(tmp_data, axis=0)
            if batch_idx == 0:
                data_batch = tmp_data
            else:
                data_batch = np.concatenate([data_batch, tmp_data], axis=0)

        return data_batch
