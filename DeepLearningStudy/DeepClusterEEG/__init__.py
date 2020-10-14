import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans

from DeepClusterEEG.Network import AutoEncoder
from DeepClusterEEG.db_helper import *

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel as C


def scatter_result(u_list, plot_nn_range, plot_min_dist_range, output_directory, color_list, seed=0, alpha=0.2):
    plot_nn_start = plot_nn_range[0]
    plot_nn_num = plot_nn_range[1] - plot_nn_range[0] + 1
    plot_min_dist_starrt = plot_min_dist_range[0]
    plot_min_dist_num = plot_min_dist_range[1] - plot_min_dist_range[0] + 1

    fig, ax = plt.subplots(plot_nn_num, plot_min_dist_num, figsize=(plot_min_dist_num * 4, plot_nn_num * 4))
    for j in range(plot_nn_num):
        for i in range(plot_min_dist_num):
            ax[j, i].scatter(u_list[plot_min_dist_starrt + i][plot_nn_start + j][:, 0], u_list[plot_min_dist_starrt + i][plot_nn_start + j][:, 1], color=color_list, alpha=0.2)
    fig.savefig('./%s/result_test/scatter_%d.png' % (output_directory, seed))
    plt.close(fig)


def k_mean_cluster(u, n_cluster, color_list=None):
    km = KMeans(n_cluster)
    km.fit(u)
    res = km.predict(u)

    if color_list is None:
        class_color_result = res
    else:
        class_color_result = []
        for cls in res:
            class_color_result.append(color_list[cls])

    return km, class_color_result


def interrupt_result_compare(auto_encoder, feature_array, label_list, feature_name, original_feature_array, epoch_idx, random_seed=None,
                             output_directory=None, nn_list=None, min_dist_list=None):
    if not os.path.exists('./%s/result_test' % output_directory):
        os.mkdir('./%s/result_test' % output_directory)
    if random_seed is None:
        random_seed = [1, 2, 3]
    if nn_list is None:
        nn_list = [0.1, 0.2, 0.4, 0.8, 1.]
    if min_dist_list is None:
        min_dist_list = [3, 4, 5, 6, 8]

    en, re = auto_encoder(feature_array[:, :])
    # en = auto_encoder.encode_cluster_net(feature_array[:, :])
    u_list = []
    import umap
    for crt_seed in random_seed:
        tmp_u_list = []
        fig, ax = plt.subplots(len(nn_list), len(min_dist_list), figsize=(len(min_dist_list) * 4, len(nn_list) * 4))
        col_idx = 0
        for min_dist in nn_list:
            row_idx = 0
            tmp_u_list_2 = []
            for nn in min_dist_list:
                u = umap.UMAP(n_neighbors=nn, random_state=crt_seed, n_epochs=1000, min_dist=min_dist, negative_sample_rate=10, target_n_neighbors=10).fit_transform(en)
                ax[col_idx, row_idx].scatter(u[:, 0], u[:, 1], cmap='autumn', alpha=0.4)
                tmp_u_list_2.append(u)
                row_idx += 1
            col_idx += 1
            tmp_u_list.append(tmp_u_list_2)

        fig.savefig('./%s/result_test/umap_visual_compare_%d_%d.png' % (output_directory, epoch_idx, crt_seed))

        plt.close(fig)
        u_list.append(tmp_u_list)

    return u_list


def main_pretrain(select_data_label=None, k=16):
    if not os.path.exists('./pretrain_result'):
        os.mkdir('./pretrain_result')

    feature_adhd_list, feature_name_list, label_list_adhd = get_feature_adhd()
    feature_adhd_array = np.array(feature_adhd_list).T

    feature_array = feature_adhd_array

    gathered_feature, gathered_feature_name = feature_gathering(feature_array, feature_name_list)
    gathered_feature_array = np.array(gathered_feature).T[:, :]
    # feature_array = np.concatenate([feature_array, gathered_feature_array], axis=1)
    feature_array = gathered_feature_array

    feature_array = feature_array[np.where(np.max(feature_array, axis=1) == np.max(feature_array, axis=1))[0]]

    feature_num = feature_array.shape[1]
    original_feature_array = np.array(feature_array)

    feature_name = gathered_feature_name
    # feature_name = feature_name_list + gathered_feature_name

    med_std_array = np.zeros([feature_num, 2])

    for feature_idx in range(feature_num):
        feature_array[:, feature_idx] = (original_feature_array[:, feature_idx] - np.median(original_feature_array[:, feature_idx])) / (
                2 * np.std(original_feature_array[:, feature_idx]))
        med_std_array[feature_idx, 0] = np.median(original_feature_array[:, feature_idx])
        med_std_array[feature_idx, 1] = 2 * np.std(original_feature_array[:, feature_idx])

    feature_array = np.where(feature_array > 1, 1, np.where(feature_array < -1, -1, feature_array))

    auto_encoder = AutoEncoder('auto_encoder', feature_num=feature_num, k=k, beta=0.05, delta=1., gamma=1., noise_std=0.05)
    np.save('med_std_array.npy', med_std_array)

    for epoch_idx in range(1, 50001):
        idx = np.arange(feature_array.shape[0])
        np.random.shuffle(idx)
        for iter_idx in range(1):
            random_idx = idx[iter_idx * int(idx.shape[0] / 1):(iter_idx + 1) * int(idx.shape[0] / 1)]
            loss_tmp, loss_encoder, loss_decoder, loss_cluster = auto_encoder.train_(feature_array[random_idx], epoch_idx, cluster=False)
        print("%02d: total=%f, encoder=%f, decoder=%f, cluster=%f" % (epoch_idx, loss_tmp, loss_encoder, loss_decoder, loss_cluster))

        if epoch_idx % 100 == 0:
            fig, ax = plt.subplots(2, 2)
            ax[0, 0].plot(feature_array[0], color='blue')
            ax[0, 0].plot(auto_encoder(feature_array[0:1])[1][0], color='red')
            ax[0, 1].plot(feature_array[1], color='blue')
            ax[0, 1].plot(auto_encoder(feature_array[10:11])[1][0], color='red')
            ax[1, 0].plot(feature_array[2], color='blue')
            ax[1, 0].plot(auto_encoder(feature_array[21:22])[1][0], color='red')
            ax[1, 1].plot(feature_array[3], color='blue')
            ax[1, 1].plot(auto_encoder(feature_array[33:34])[1][0], color='red')
            fig.savefig('./pretrain_result/tmp_%d.png' % epoch_idx)
            plt.close(fig)

        if epoch_idx % 1000 == 0:
            auto_encoder.save_weights('./pre_trained_network')


def interupt_result(auto_encoder, feature_array, epoch_idx=None, age_list=None):
    en, re = auto_encoder(feature_array[:, :])
    u_list = []
    if age_list is None:
        c = None
    else:
        c = np.array(age_list)
    import umap
    if epoch_idx is None:
        # fig, ax = plt.subplots(2, 3, figsize=(12, 9))
        for i in range(1):
            u = umap.UMAP(n_neighbors=i + 5, random_state=1).fit_transform(en)
            u_list.append(u)
            # ax[int(i / 3), i % 3].scatter(u[0:, 0], u[0:, 1], c=c, cmap='autumn', alpha=0.5)
        # fig.savefig('./result/umap_visual.png')
        # plt.close(fig)
    else:
        fig, ax = plt.subplots(2, 4, figsize=(15, 9))
        for i in range(6):
            u = umap.UMAP(n_neighbors=i + 3, random_state=1).fit_transform(en)
            u_list.append(u)
            ax[int(i / 4), i % 4].scatter(u[0:, 0], u[0:, 1], c=c, cmap='autumn', alpha=0.5)
        fig.savefig('./result/umap_visual_%d.png' % (epoch_idx))
        plt.close(fig)
    return u_list


def main(output_directory, k=16):
    feature_adhd_list, feature_name_list, label_list_adhd = get_feature_adhd()
    feature_adhd_array = np.array(feature_adhd_list).T

    feature_array = feature_adhd_array

    gathered_feature, gathered_feature_name = feature_gathering(feature_array, feature_name_list)
    gathered_feature_array = np.array(gathered_feature).T[:, :]
    # feature_array = np.concatenate([feature_array, gathered_feature_array], axis=1)
    feature_array = gathered_feature_array

    feature_array = feature_array[np.where(np.max(feature_array, axis=1) == np.max(feature_array, axis=1))[0]]
    feature_num = feature_array.shape[1]

    original_feature_array = np.array(feature_array)
    feature_name = gathered_feature_name

    med_std_array = np.load('med_std_array.npy')

    feature_array = (feature_array - med_std_array[:, 0]) / (med_std_array[:, 1])
    feature_array = np.where(feature_array > 1, 1, np.where(feature_array < -1, -1, feature_array))

    auto_encoder = AutoEncoder('auto_encoder', feature_num=feature_num, k=k, beta=0.05, delta=1., gamma=1., noise_std=0.05)
    auto_encoder.load_weights('./pre_trained_network')

    for epoch_idx in range(1, 5001):
        idx = np.arange(feature_array.shape[0])
        np.random.shuffle(idx)
        for iter_idx in range(1):
            random_idx = idx[iter_idx * int(idx.shape[0] / 1):(iter_idx + 1) * int(idx.shape[0] / 1)]
            loss_tmp, loss_encoder, loss_decoder, loss_cluster = auto_encoder.train_(feature_array[random_idx], epoch_idx)
        print("%02d: total=%f, encoder=%f, decoder=%f, cluster=%f" % (epoch_idx, loss_tmp, loss_encoder, loss_decoder, loss_cluster))

        if epoch_idx % 500 == 0:
            auto_encoder.save_weights('./%s/trained_network_%d' % (output_directory, epoch_idx))

    network_info = open('./%s/network_info.txt' % output_directory, 'w')
    for weight in auto_encoder.weights:
        network_info.write(str(weight.name))
        network_info.write(str(weight.shape))


def main_test(output_directory, k=16, model_directory=None):
    feature_adhd_list, feature_name_list, label_list_adhd = get_feature_adhd()
    feature_adhd_array = np.array(feature_adhd_list).T

    feature_array = feature_adhd_array

    gathered_feature, gathered_feature_name = feature_gathering(feature_array, feature_name_list)
    gathered_feature_array = np.array(gathered_feature).T[:, :]
    feature_array = np.concatenate([feature_array, gathered_feature_array], axis=1)
    feature_array = gathered_feature_array

    feature_array = feature_array[np.where(np.max(feature_array, axis=1) == np.max(feature_array, axis=1))[0]]
    feature_num = feature_array.shape[1]

    original_feature_array = np.array(feature_array)
    feature_name = feature_name_list + gathered_feature_name
    feature_name = gathered_feature_name

    med_std_array = np.load('med_std_array.npy')

    feature_array = (feature_array - med_std_array[:, 0]) / (med_std_array[:, 1])
    feature_array = np.where(feature_array > 1, 1, np.where(feature_array < -1, -1, feature_array))

    auto_encoder = AutoEncoder('auto_encoder', feature_num=feature_num, k=k, beta=0.05, delta=1., gamma=1., noise_std=0.05)
    auto_encoder.load_weights('%s' % model_directory)

    random_seed_list = [1]
    # min_dist_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]  # col
    # nn_list = [3, 4, 5, 6, 7, 8, 9]  # row
    min_dist_list = [0.2, 0.35, 0.5, 0.75, 1.]
    nn_list = [4, 6, 8, 10]

    u = interrupt_result_compare(auto_encoder, feature_array, label_list_adhd, feature_name, original_feature_array, epoch_idx=9999, random_seed=random_seed_list,
                                 output_directory=output_directory, nn_list=min_dist_list, min_dist_list=nn_list)

    selected_nn = 0
    selected_min_dist = 1
    n_cluster = 3
    plot_min_dist_range = [0, 4]
    plot_nn_range = [0, 3]
    cl = ['red', 'orange', 'yellow']
    # cl = ['red', 'orange', 'yellow', 'green', 'blue']
    for i in range(len(random_seed_list)):
        km, class_cl = k_mean_cluster(u[i][selected_min_dist][selected_nn], n_cluster, cl)
        scatter_result(u[i], plot_nn_range, plot_min_dist_range, output_directory, color_list=class_cl, seed=random_seed_list[i], alpha=0.2)

    kernel = C(1, (1E-10, 1E10)) * RBF(1, (1E-10, 1E10)) + WhiteKernel(noise_level=0.5)
    # kernel = C(1, (1E-10, 1E10)) * RBF(1, (1E-10, 1E10))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    x = u[0][2][1]
    gpr.fit(x[:, 0:1], x[:, 1])
    tmp = np.linspace(-15, 15, 100)
    tmp = np.expand_dims(tmp, axis=1)
    y = gpr.predict(tmp)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x[:, 0], x[:, 1])
    ax.plot(tmp, y)
    fig.savefig('./%s/result_test/reg.png' % output_directory)

    closest_x_list = []
    for i in range(x.shape[0]):
        closest_dist = 99999.
        for j in range(y.shape[0]):
            crt_dist = np.sqrt(np.square(x[i, 0] - tmp[j, 0]) + np.square(x[i, 1] - y[j]))
            if crt_dist < closest_dist:
                closest_dist = crt_dist
                closest_x = tmp[j, 0]
        closest_x_list.append(closest_x)
    closest_x_array = np.array(closest_x_list)

    print()
    # scatter_result()


def main_gif():
    import imageio
    from PIL import Image

    paths = []
    for i in range(1000):
        img = Image.open('./tmp_%d.png' % i)
        paths.append(img)
    imageio.mimsave('./result.gif', paths, fps=40)


if __name__ == "__main__":
    import os

    # Output directory path
    k = 8
    model_directory = './gathered_feature_with_channel_%d/trained_network_5000' % k
    output_directory = './gathered_feature_with_channel_%d' % k

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # main_pretrain(k=k)
    # main(output_directory, k=k)
    main_test(output_directory, k=k, model_directory=model_directory)
