import numpy as np


def circular_sample(num_sam, x, y, r_out, r_in=0):
    r_sample = np.random.uniform(r_in, r_out, num_sam)
    theta_sample = np.random.uniform(0, 2 * np.pi, num_sam)
    rtn_x = x + r_sample * np.cos(theta_sample)
    rtn_y = y + r_sample * np.sin(theta_sample)
    return np.concatenate([np.expand_dims(rtn_x, 1), np.expand_dims(rtn_y, 1)], axis=1)


def rectangular_sample(num_sam, x, y, dx, dy):
    dx_sample = np.random.uniform(0, dx, num_sam)
    dy_sample = np.random.uniform(0, dy, num_sam)
    rtn_x = x + dx_sample
    rtn_y = y + dy_sample
    return np.concatenate([np.expand_dims(rtn_x, 1), np.expand_dims(rtn_y, 1)], axis=1)
