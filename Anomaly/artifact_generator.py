from db_helper import DBHelper
import numpy as np
import math
import matplotlib.pyplot as plt


def artifact_generator(batch_size, data_batch, total_len, const_theta, const_phi, const_tau, const_step_size_min, const_step_size_max):
    img_size = data_batch.shape[1]

    theta_cur = np.zeros(batch_size)
    phi_cur = np.zeros(batch_size)
    step_size = np.random.uniform(const_step_size_min, const_step_size_max, batch_size)

    cur_x = np.random.uniform(0, img_size, batch_size)
    cur_y = np.random.uniform(0, img_size, batch_size)

    arr_pix = []

    for step in range(total_len):
        theta_x = np.random.randint(0, 2, batch_size)
        theta_interval = np.random.uniform(0, const_theta)
        theta_delta = np.random.uniform(-theta_interval, theta_interval, batch_size)
        theta_cur = theta_cur + theta_x * theta_delta

        phi_x = np.random.randint(0, 2, batch_size)
        phi_interval = np.random.uniform(0, const_phi)
        phi_delta = np.random.uniform(-phi_interval, phi_interval, batch_size)
        phi_cur = phi_cur + phi_x * phi_delta + theta_cur

        cur_x = cur_x + step_size * np.cos(phi_cur)
        cur_y = cur_y + step_size * np.sin(phi_cur)

        arounded_cur_x = np.around(cur_x)
        arounded_cur_y = np.around(cur_y)

        tmp_x = np.min(np.concatenate([(img_size - 1) * np.ones([batch_size, 1]), np.expand_dims(arounded_cur_x, 1)], axis=1), axis=1)
        bounded_cur_x = np.max(np.concatenate([np.zeros([batch_size, 1]), np.expand_dims(tmp_x, 1)], axis=1), axis=1)

        tmp_y = np.min(np.concatenate([(img_size - 1) * np.ones([batch_size, 1]), np.expand_dims(arounded_cur_y, 1)], axis=1), axis=1)
        bounded_cur_y = np.max(np.concatenate([np.zeros([batch_size, 1]), np.expand_dims(tmp_y, 1)], axis=1), axis=1)

        arr_pix.append((bounded_cur_x, bounded_cur_y))

    pix_val = np.zeros(batch_size)

    tmp_mask = np.zeros([batch_size, img_size, img_size])

    for batch_idx in range(batch_size):
        for pix_artifact in arr_pix:
            if tmp_mask[batch_idx, int(pix_artifact[0][batch_idx]), int(pix_artifact[1][batch_idx])] == 1.:
                continue
            pix_val[batch_idx] += data_batch[batch_idx, int(pix_artifact[0][batch_idx]), int(pix_artifact[1][batch_idx])]
            tmp_mask[batch_idx, int(pix_artifact[0][batch_idx]), int(pix_artifact[1][batch_idx])] = 1.

        average_pix_val = pix_val[batch_idx] / np.sum(tmp_mask[batch_idx])

        for pix_artifact in arr_pix:
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if int(pix_artifact[0][batch_idx]) + i > img_size - 1 or int(pix_artifact[0][batch_idx]) + i < 0:
                        continue
                    if int(pix_artifact[1][batch_idx]) + j > img_size - 1 or int(pix_artifact[1][batch_idx]) + j < 0:
                        continue
                    random_tau = np.random.uniform(-const_tau, const_tau, 1)
                    data_batch[batch_idx, int(pix_artifact[0][batch_idx]) + i, int(pix_artifact[1][batch_idx]) + j] = average_pix_val + random_tau

    return data_batch, tmp_mask
