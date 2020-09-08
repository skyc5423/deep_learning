from PIL import Image
import numpy as np
import os

DB_PATH = '/Users/sangminlee/Documents/Sangmin/DB/toy_data/circle_bgd_test'
if not os.path.exists(DB_PATH):
    os.mkdir(DB_PATH)

N = 500


def make_circle():
    for n in range(N):
        random_x = np.random.randint(20, 108, 1)
        random_y = np.random.randint(20, 108, 1)
        random_radius = np.random.randint(5, 20, 1)
        random_color = np.random.uniform(0, 1, 3)
        random_bgd = np.random.uniform(0, 1, 3)

        x = np.arange(128)
        xx, yy = np.meshgrid(x - random_x, x - random_y)
        zz = np.expand_dims(np.sqrt(xx ** 2 + yy ** 2), axis=-1)
        zz = np.tile(zz, [1, 1, 3])

        circle = np.where(zz < random_radius, random_color * 255, random_bgd * 255)
        circle = circle.astype(np.uint8)
        img = Image.fromarray(circle)
        save_name = os.path.join(DB_PATH, 'circle_%d.png' % n)
        img.save(save_name)
        img.close()


def make_rectangle():
    for n in range(N):
        random_x = np.random.randint(20, 108, 1)
        random_y = np.random.randint(20, 108, 1)
        random_radius = np.random.randint(5, 20, 1)
        random_color = np.random.uniform(0, 1, 3)
        random_bgd = np.random.uniform(0, 1, 3)

        x = np.arange(128)
        xx, yy = np.meshgrid(x - random_x, x - random_y)
        xx_ = np.expand_dims(xx, 0)
        yy_ = np.expand_dims(yy, 0)
        zz_tmp = np.concatenate([xx_, yy_], axis=0)
        zz = np.expand_dims(np.max(np.abs(zz_tmp), axis=0), axis=-1)
        zz = np.tile(zz, [1, 1, 3])

        circle = np.where(zz < random_radius, random_color * 255, random_bgd * 255)
        circle = circle.astype(np.uint8)
        img = Image.fromarray(circle)
        save_name = os.path.join(DB_PATH, 'rectangle_%d.png' % n)
        img.save(save_name)
        img.close()

make_circle()