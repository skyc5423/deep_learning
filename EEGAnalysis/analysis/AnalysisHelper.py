import os
import umap
import matplotlib.pyplot as plt
from deep_neural_net import Network
from data.DBHelper import DBHelper
import matplotlib.colors as mcolors
import skfuzzy as fuzz
import numpy as np


class AnalysisHelper(object):
    def __init__(self, network: Network, db_helper: DBHelper):
        self.network = network
        self.db_helper = db_helper
        self.umap_list = None
        self.umap_feature_list = None
        self.random_seed = 0
        self.distance = None
        self.cntr = None
        self.u = None
        self.cluster_array = None
        self.named_color_list = []
        for key in mcolors.CSS4_COLORS.keys():
            self.named_color_list.append(key)
        self.used_color_list = None

    def analysis_umap(self, min_dist_list: list, nn_list: list, random_seed=0, output_directory=None):
        if self.network is None:
            print("Analysis Helper has no network.")
            return

        if self.umap_list is not None:
            print("UMAP list already exists. old UMAP list will be replaced with new one")
        if self.umap_feature_list is not None:
            print("UMAP feature list already exists. old UMAP feature list will be replaced with new one")

        encoded_feature_array = self.network.compute_cluster_y(self.db_helper.norm_total_data).numpy()
        self.random_seed = random_seed
        umap_feature_list = []
        learned_umap_list = []
        min_dist_idx = 0
        for min_dist in min_dist_list:
            nn_idx = 0
            tmp_umap_feature_list = []
            tmp_learned_umap_list = []
            for nn in nn_list:
                if min_dist > 1.:
                    learned_umap = umap.UMAP(n_neighbors=nn, random_state=random_seed, spread=min_dist, min_dist=min_dist, negative_sample_rate=10, target_n_neighbors=10,
                                             n_epochs=2000)
                    umap_feature = learned_umap.fit_transform(encoded_feature_array)
                else:
                    learned_umap = umap.UMAP(n_neighbors=nn, random_state=random_seed, min_dist=min_dist, negative_sample_rate=10, target_n_neighbors=10,
                                             n_epochs=2000)
                    umap_feature = learned_umap.fit_transform(encoded_feature_array)

                tmp_umap_feature_list.append(umap_feature)
                tmp_learned_umap_list.append(learned_umap)
                nn_idx += 1
            min_dist_idx += 1
            umap_feature_list.append(tmp_umap_feature_list)
            learned_umap_list.append(tmp_learned_umap_list)

        self.umap_list = learned_umap_list
        self.umap_feature_list = umap_feature_list

        if output_directory is not None:
            if not os.path.exists('%s' % output_directory):
                os.mkdir('%s' % output_directory)

            fig, ax = plt.subplots(len(min_dist_list), len(nn_list), figsize=(len(nn_list) * 4, len(min_dist_list) * 4))
            if len(min_dist_list) > 1 and len(nn_list) > 1:
                for i in range(len(min_dist_list)):
                    for j in range(len(nn_list)):
                        ax[i, j].scatter(umap_feature_list[i][j][:self.db_helper.adhd_start_idx, 0],
                                         umap_feature_list[i][j][:self.db_helper.adhd_start_idx, 1], alpha=0.1,
                                         color='black')
                        ax[i, j].scatter(umap_feature_list[i][j][self.db_helper.adhd_start_idx:self.db_helper.mipdb_start_idx, 0],
                                         umap_feature_list[i][j][self.db_helper.adhd_start_idx:self.db_helper.mipdb_start_idx, 1], alpha=0.3, color='red')
                        ax[i, j].scatter(umap_feature_list[i][j][self.db_helper.mipdb_start_idx:, 0],
                                         umap_feature_list[i][j][self.db_helper.mipdb_start_idx:, 1], alpha=0.3,
                                         color='blue')
            elif len(min_dist_list) > 1:
                for i in range(len(min_dist_list)):
                    ax[i].scatter(umap_feature_list[i][0][:, 0], umap_feature_list[i][0][:, 1], alpha=0.2)
            elif len(nn_list) > 1:
                for j in range(len(nn_list)):
                    ax[j].scatter(umap_feature_list[0][j][:, 0], umap_feature_list[0][j][:, 1], alpha=0.2)
            else:
                ax.scatter(umap_feature_list[0][0][:, 0], umap_feature_list[0][0][:, 1], alpha=0.2)
            fig.savefig('%s/umap_scatter_random_seed_%d.png' % (output_directory, random_seed))
            plt.close(fig)

    def random_seed_umap(self, min_dist: float, nn: int, random_seed_list: list, output_directory=None, color_list=None, distance_std=1., cluster_array=None):
        if self.network is None:
            print("Analysis Helper has no network.")
            return

        # if self.umap_list is not None:
        #     print("UMAP list already exists. old UMAP list will be replaced with new one")
        # if self.umap_feature_list is not None:
        #     print("UMAP feature list already exists. old UMAP feature list will be replaced with new one")

        if cluster_array is None:
            cluster_array = self.cluster_array
        if self.used_color_list is None:
            used_color_list = self.named_color_list
        else:
            used_color_list = self.used_color_list
        if color_list is None:
            color_list = used_color_list

        cluster_num = np.max(cluster_array) + 1
        d = 1 / self.distance
        d = (d - np.min(d)) / (np.max(d) - np.min(d))

        ax_num = 3
        fig, ax = plt.subplots(ax_num, ax_num, figsize=(12, 12))
        fig_cluster, ax_cluster = plt.subplots(ax_num, ax_num, figsize=(12, 12))
        encoded_feature_array = self.network.compute_cluster_y(self.db_helper.norm_total_data).numpy()
        for random_seed_idx in range(len(random_seed_list)):
            random_seed = random_seed_list[random_seed_idx]
            learned_umap = umap.UMAP(n_neighbors=nn, random_state=random_seed, spread=min_dist, min_dist=min_dist, negative_sample_rate=10, target_n_neighbors=10, n_epochs=2000)
            umap_feature = learned_umap.fit_transform(encoded_feature_array)
            ax[int(random_seed_idx / ax_num), random_seed_idx % ax_num].scatter(umap_feature[:self.db_helper.adhd_start_idx, 0],
                                                                                umap_feature[:self.db_helper.adhd_start_idx, 1], alpha=0.1,
                                                                                color='black', s=15)
            ax[int(random_seed_idx / ax_num), random_seed_idx % ax_num].scatter(umap_feature[self.db_helper.adhd_start_idx:self.db_helper.mipdb_start_idx, 0],
                                                                                umap_feature[self.db_helper.adhd_start_idx:self.db_helper.mipdb_start_idx, 1], alpha=0.3,
                                                                                color='red', s=15)
            ax[int(random_seed_idx / ax_num), random_seed_idx % ax_num].scatter(umap_feature[self.db_helper.mipdb_start_idx:, 0],
                                                                                umap_feature[self.db_helper.mipdb_start_idx:, 1], alpha=0.3,
                                                                                color='blue', s=15)
            for cluster_idx in range(cluster_num):
                ax_cluster[int(random_seed_idx / ax_num), random_seed_idx % ax_num].scatter(umap_feature[:, 0],
                                                                                            umap_feature[:, 1], alpha=0.2, color=color_list[cluster_idx],
                                                                                            s=np.where(d[cluster_idx] > np.mean(d) + distance_std * np.std(d), d[cluster_idx],
                                                                                                       0) * 80)
                ax_cluster[int(random_seed_idx / ax_num), random_seed_idx % ax_num].scatter(umap_feature[np.argmax(d[cluster_idx]), 0],
                                                                                            umap_feature[np.argmax(d[cluster_idx]), 1], marker='*', color=color_list[cluster_idx],
                                                                                            edgecolors='black',
                                                                                            linewidths=0.5)

        fig.savefig('%s/umap_scatter_random_seed_list.png' % (output_directory))
        plt.close(fig)

        fig_cluster.savefig('%s/umap_scatter_random_seed_list_cluster.png' % (output_directory))
        plt.close(fig_cluster)

    def plot_cluster_umap(self, min_dist_list: list, nn_list: list, output_directory: str, color_list: list, cluster_array=None):
        if not os.path.exists('%s' % output_directory):
            os.mkdir('%s' % output_directory)
        if not os.path.exists('%s/result_max' % output_directory):
            os.mkdir('%s/result_max' % output_directory)

        if cluster_array is None:
            cluster_array = self.cluster_array
        if self.used_color_list is None:
            used_color_list = self.named_color_list
        else:
            used_color_list = self.used_color_list
        if color_list is None:
            color_list = used_color_list

        cluster_num = np.max(cluster_array) + 1

        d = 1 / self.distance
        d = (d - np.min(d)) / (np.max(d) - np.min(d))

        fig, ax = plt.subplots(len(min_dist_list), len(nn_list), figsize=(len(nn_list) * 4, len(min_dist_list) * 4))
        if len(min_dist_list) > 1 and len(nn_list) > 1:
            for i in range(len(min_dist_list)):
                for j in range(len(nn_list)):
                    for cluster_idx in range(cluster_num):
                        ax[i, j].scatter(self.umap_feature_list[i][j][np.argmax(d[cluster_idx]), 0],
                                         self.umap_feature_list[i][j][np.argmax(d[cluster_idx]), 1], marker='*', color=color_list[cluster_idx], edgecolors='black', linewidths=0.5)
                        ax[i, j].scatter(self.umap_feature_list[i][j][np.where(cluster_array == cluster_idx), 0],
                                         self.umap_feature_list[i][j][np.where(cluster_array == cluster_idx), 1], alpha=0.2, color=color_list[cluster_idx])
        elif len(min_dist_list) > 1:
            for i in range(len(min_dist_list)):
                for cluster_idx in range(cluster_num):
                    ax[i].scatter(self.umap_feature_list[i][0][np.where(cluster_array == cluster_idx), 0],
                                  self.umap_feature_list[i][0][np.where(cluster_array == cluster_idx), 1], alpha=0.2, color=color_list[cluster_idx])
        elif len(nn_list) > 1:
            for j in range(len(nn_list)):
                for cluster_idx in range(cluster_num):
                    ax[j].scatter(self.umap_feature_list[0][j][np.where(cluster_array == cluster_idx), 0],
                                  self.umap_feature_list[0][j][np.where(cluster_array == cluster_idx), 1], alpha=0.2, color=color_list[cluster_idx])
        else:
            ax.scatter(self.umap_feature_list[0][0][:, 0], self.umap_feature_list[0][0][:, 1], alpha=0.2)
        fig.savefig('%s/result_max/umap_scatter_with_cluster_%d.png' % (output_directory, self.random_seed))
        plt.close(fig)

    def plot_cluster_umap_distance(self, min_dist_list: list, nn_list: list, output_directory: str, color_list: list, cluster_array=None, distance_std: float = 1.):
        if not os.path.exists('%s/result_distance' % output_directory):
            os.mkdir('%s/result_distance' % output_directory)

        if cluster_array is None:
            cluster_array = self.cluster_array
        if self.used_color_list is None:
            used_color_list = self.named_color_list
        else:
            used_color_list = self.used_color_list
        if color_list is None:
            color_list = used_color_list

        d = 1 / self.distance
        d = (d - np.min(d)) / (np.max(d) - np.min(d))

        cluster_num = np.max(cluster_array) + 1

        fig_total, ax_total = plt.subplots(len(min_dist_list), len(nn_list), figsize=(len(nn_list) * 4, len(min_dist_list) * 4))
        for cluster_idx in range(cluster_num):
            fig, ax = plt.subplots(len(min_dist_list), len(nn_list), figsize=(len(nn_list) * 4, len(min_dist_list) * 4))
            for i in range(len(min_dist_list)):
                for j in range(len(nn_list)):
                    ax[i, j].scatter(self.umap_feature_list[i][j][:, 0],
                                     self.umap_feature_list[i][j][:, 1], alpha=0.3, color=color_list[cluster_idx],
                                     s=np.where(d[cluster_idx] > np.mean(d) + distance_std * np.std(d), d[cluster_idx], 0) * 60)
                    ax_total[i, j].scatter(self.umap_feature_list[i][j][:, 0],
                                           self.umap_feature_list[i][j][:, 1], alpha=0.2, color=color_list[cluster_idx],
                                           s=np.where(d[cluster_idx] > np.mean(d) + distance_std * np.std(d), d[cluster_idx], 0) * 60)
                    ax[i, j].scatter(self.umap_feature_list[i][j][np.argmax(d[cluster_idx]), 0],
                                     self.umap_feature_list[i][j][np.argmax(d[cluster_idx]), 1], marker='*', color=color_list[cluster_idx], edgecolors='black', linewidths=0.5)
                    ax_total[i, j].scatter(self.umap_feature_list[i][j][np.argmax(d[cluster_idx]), 0],
                                           self.umap_feature_list[i][j][np.argmax(d[cluster_idx]), 1], marker='*', color=color_list[cluster_idx], edgecolors='black',
                                           linewidths=0.5)
            fig.savefig('%s/result_distance/umap_scatter_with_cluster_%d.png' % (output_directory, cluster_idx))
            plt.close(fig)
        fig_total.savefig('%s/result_distance/umap_scatter_with_cluster.png' % (output_directory))
        plt.close(fig_total)

    def fuzzy_cluster(self, cluster_num: int, m: float, min_dist=None):
        encoded_feature_array = self.network.compute_cluster_y(self.db_helper.norm_total_data).numpy()
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(encoded_feature_array.T, cluster_num, m=m, error=0.005, maxiter=10000, init=None, seed=0)
        self.distance = d
        self.cntr = cntr
        self.u = u
        if min_dist is None:
            self.cluster_array = np.argmax(u, axis=0)
        # else:
        #     self.cluster_array = np.

    def plot_bar_graph(self, output_directory: str, color_list: list, cluster_array=None):
        if cluster_array is None:
            cluster_array = self.cluster_array
        if self.used_color_list is None:
            used_color_list = self.named_color_list
        else:
            used_color_list = self.used_color_list
        if color_list is None:
            color_list = used_color_list

        cluster_num = np.max(cluster_array) + 1

        x, y, yerr = [], [], []
        y_child, yerr_child = [], []
        y_adhd, yerr_adhd = [], []
        for cluster_idx in range(cluster_num):
            # x.append(cluster_idx)
            # y.append(np.mean(self.db_helper.norm_total_data[np.where(cluster_array == cluster_idx)[0]], axis=0))
            # yerr.append(np.std(self.db_helper.norm_total_data[np.where(cluster_array == cluster_idx)[0]], axis=0))
            # y_child.append(np.mean(self.db_helper.norm_total_data[np.where(d[cluster_idx, :self.db_helper.adhd_start_idx] > np.mean(d) + distance_std * np.std(d))[0]], axis=0))
            # yerr_child.append(np.std(self.db_helper.norm_total_data[np.where(d[cluster_idx, :self.db_helper.adhd_start_idx] > np.mean(d) + distance_std * np.std(d))[0]], axis=0))
            # y_adhd_db.append(
            #     np.mean(self.db_helper.norm_total_data[
            #                 np.where(d[cluster_idx, self.db_helper.adhd_start_idx:] > np.mean(d) + distance_std * np.std(d))[0] + self.db_helper.adhd_start_idx], axis=0))
            # yerr_adhd_db.append(
            #     np.std(self.db_helper.norm_total_data[
            #                np.where(d[cluster_idx, self.db_helper.adhd_start_idx:] > np.mean(d) + distance_std * np.std(d))[0] + self.db_helper.adhd_start_idx], axis=0))
            # y_adhd.append(
            #     np.mean(self.db_helper.norm_total_data[np.array(np.where(np.array(self.db_helper.adhd_label['DZ_G']) == 0)[0] + self.db_helper.adhd_start_idx)[np.where(
            #         d[cluster_idx, np.array(np.where(np.array(self.db_helper.adhd_label['DZ_G']) == 0)[0] + self.db_helper.adhd_start_idx)] > np.mean(d) + distance_std * np.std(
            #             d))[0]]], axis=0))
            # yerr_adhd.append(
            #     np.std(self.db_helper.norm_total_data[np.array(np.where(np.array(self.db_helper.adhd_label['DZ_G']) == 0)[0] + self.db_helper.adhd_start_idx)[np.where(
            #         d[cluster_idx, np.array(np.where(np.array(self.db_helper.adhd_label['DZ_G']) == 0)[0] + self.db_helper.adhd_start_idx)] > np.mean(d) + distance_std * np.std(
            #             d))[0]]], axis=0))

            x.append(cluster_idx)
            y.append(np.mean(self.db_helper.norm_total_data[np.where(cluster_array == cluster_idx)[0]], axis=0))
            yerr.append(np.std(self.db_helper.norm_total_data[np.where(cluster_array == cluster_idx)[0]], axis=0))
            y_child.append(np.mean(self.db_helper.norm_total_data[np.where(cluster_array[:self.db_helper.adhd_start_idx] == cluster_idx)[0]], axis=0))
            yerr_child.append(np.std(self.db_helper.norm_total_data[np.where(cluster_array[:self.db_helper.adhd_start_idx] == cluster_idx)[0]], axis=0))
            y_adhd.append(
                np.mean(self.db_helper.norm_total_data[np.where(cluster_array[self.db_helper.adhd_start_idx:] == cluster_idx)[0] + self.db_helper.adhd_start_idx], axis=0))
            yerr_adhd.append(
                np.std(self.db_helper.norm_total_data[np.where(cluster_array[self.db_helper.adhd_start_idx:] == cluster_idx)[0] + self.db_helper.adhd_start_idx], axis=0))
        for i in range(self.db_helper.norm_total_data.shape[1]):
            fig, ax = plt.subplots(3, 1, figsize=(12, 12))
            ax[0].bar(x, np.array(y)[:, i], yerr=np.array(yerr)[:, i], color=color_list[:cluster_num])
            ax[1].bar(x, np.array(y_child)[:, i], yerr=np.array(yerr_child)[:, i], color=color_list[:cluster_num])
            ax[2].bar(x, np.array(y_adhd)[:, i], yerr=np.array(yerr_adhd)[:, i], color=color_list[:cluster_num])
            for ax_i in range(3):
                ax[ax_i].plot(np.linspace(-1, cluster_num, 100), np.zeros(100), 'k--', linewidth=0.5)
                ax[ax_i].plot(np.linspace(-1, cluster_num, 100), np.ones(100) * 0.5, 'r--', linewidth=0.5)
                ax[ax_i].plot(np.linspace(-1, cluster_num, 100), -np.ones(100) * 0.5, 'b--', linewidth=0.5)
                ax[ax_i].plot(np.linspace(-1, cluster_num, 100), np.ones(100) * 0.25, 'g--', linewidth=0.5)
                ax[ax_i].plot(np.linspace(-1, cluster_num, 100), -np.ones(100) * 0.25, 'g--', linewidth=0.5)
            fig.savefig('%s/result_max/hist_feature_%s.png' % (output_directory, self.db_helper.feature_name_list[i]))
            plt.close(fig)

        for cluster_idx in range(cluster_num):
            print("Max Cluster - %d : # of Total = %d, # of ADHD = %d" % (
                cluster_idx, np.sum(np.where(cluster_array == cluster_idx, 1, 0)), np.sum(np.where(cluster_array[self.db_helper.adhd_start_idx:] == cluster_idx, 1, 0))))

    def plot_bar_graph_distance(self, output_directory: str, color_list: list, cluster_array=None, distance_std: float = 1.):
        if cluster_array is None:
            cluster_array = self.cluster_array
        if self.used_color_list is None:
            used_color_list = self.named_color_list
        else:
            used_color_list = self.used_color_list
        if color_list is None:
            color_list = used_color_list

        d = 1 / self.distance
        d = (d - np.min(d)) / (np.max(d) - np.min(d))

        cluster_num = np.max(cluster_array) + 1

        x, y, yerr = [], [], []
        y_child, yerr_child = [], []
        y_adhd, yerr_adhd = [], []
        y_sch_db, yerr_sch_db = [], []
        y_mipdb_db, yerr_mipdb_db = [], []
        for cluster_idx in range(cluster_num):
            x.append(cluster_idx)
            y.append(np.mean(self.db_helper.norm_total_data[np.where(d[cluster_idx] > np.mean(d) + distance_std * np.std(d))[0]], axis=0))
            yerr.append(np.std(self.db_helper.norm_total_data[np.where(d[cluster_idx] > np.mean(d) + distance_std * np.std(d))[0]], axis=0))
            y_child.append(np.mean(self.db_helper.norm_total_data[np.where(d[cluster_idx, :self.db_helper.adhd_start_idx] > np.mean(d) + distance_std * np.std(d))[0]], axis=0))
            yerr_child.append(np.std(self.db_helper.norm_total_data[np.where(d[cluster_idx, :self.db_helper.adhd_start_idx] > np.mean(d) + distance_std * np.std(d))[0]], axis=0))
            y_sch_db.append(
                np.mean(self.db_helper.norm_total_data[
                            np.where(d[cluster_idx, self.db_helper.adhd_start_idx:self.db_helper.mipdb_start_idx] > np.mean(d) + distance_std * np.std(d))[
                                0] + self.db_helper.adhd_start_idx], axis=0))
            yerr_sch_db.append(
                np.std(self.db_helper.norm_total_data[
                           np.where(d[cluster_idx, self.db_helper.adhd_start_idx:self.db_helper.mipdb_start_idx] > np.mean(d) + distance_std * np.std(d))[
                               0] + self.db_helper.adhd_start_idx], axis=0))
            y_adhd.append(
                np.mean(self.db_helper.norm_total_data[np.array(np.where(np.array(self.db_helper.adhd_label['DZ_G']) == 0)[0] + self.db_helper.adhd_start_idx)[np.where(
                    d[cluster_idx, np.array(np.where(np.array(self.db_helper.adhd_label['DZ_G']) == 0)[0] + self.db_helper.adhd_start_idx)] > np.mean(d) + distance_std * np.std(
                        d))[0]]], axis=0))
            yerr_adhd.append(
                np.std(self.db_helper.norm_total_data[np.array(np.where(np.array(self.db_helper.adhd_label['DZ_G']) == 0)[0] + self.db_helper.adhd_start_idx)[np.where(
                    d[cluster_idx, np.array(np.where(np.array(self.db_helper.adhd_label['DZ_G']) == 0)[0] + self.db_helper.adhd_start_idx)] > np.mean(d) + distance_std * np.std(
                        d))[0]]], axis=0))
            y_mipdb_db.append(
                np.mean(self.db_helper.norm_total_data[
                            np.where(d[cluster_idx, self.db_helper.mipdb_start_idx:] > np.mean(d) + distance_std * np.std(d))[0] + self.db_helper.mipdb_start_idx], axis=0))
            yerr_mipdb_db.append(
                np.std(self.db_helper.norm_total_data[
                           np.where(d[cluster_idx, self.db_helper.mipdb_start_idx:] > np.mean(d) + distance_std * np.std(d))[0] + self.db_helper.mipdb_start_idx], axis=0))

        for i in range(self.db_helper.norm_total_data.shape[1]):
            fig, ax = plt.subplots(5, 1, figsize=(12, 12))
            ax[0].bar(x, np.array(y)[:, i], yerr=np.array(yerr)[:, i], color=color_list[:cluster_num])
            ax[1].bar(x, np.array(y_child)[:, i], yerr=np.array(yerr_child)[:, i], color=color_list[:cluster_num])
            ax[2].bar(x, np.array(y_sch_db)[:, i], yerr=np.array(yerr_sch_db)[:, i], color=color_list[:cluster_num])
            ax[3].bar(x, np.array(y_adhd)[:, i], yerr=np.array(yerr_adhd)[:, i], color=color_list[:cluster_num])
            ax[4].bar(x, np.array(y_mipdb_db)[:, i], yerr=np.array(yerr_mipdb_db)[:, i], color=color_list[:cluster_num])
            for ax_i in range(5):
                ax[ax_i].set_ylim(-1.1, 1.1)
                ax[ax_i].plot(np.linspace(-1, cluster_num, 100), np.zeros(100), 'k--', linewidth=0.5)
                ax[ax_i].plot(np.linspace(-1, cluster_num, 100), np.ones(100) * 0.5, 'r--', linewidth=0.5)
                ax[ax_i].plot(np.linspace(-1, cluster_num, 100), -np.ones(100) * 0.5, 'b--', linewidth=0.5)
                ax[ax_i].plot(np.linspace(-1, cluster_num, 100), np.ones(100) * 0.25, 'g--', linewidth=0.5)
                ax[ax_i].plot(np.linspace(-1, cluster_num, 100), -np.ones(100) * 0.25, 'g--', linewidth=0.5)
            fig.savefig('%s/result_distance/hist_feature_%s.png' % (output_directory, self.db_helper.feature_name_list[i]))
            plt.close(fig)

        dist_sub_txt = open('./%s/result_distance/dist_sub.txt' % output_directory, 'w')
        for cluster_idx in range(cluster_num):
            dist_sub_txt.write("Distance Cluster - %d : # of Total = %d, # of ADHD-db = %d, # of ADHD = %d, # of MIPDB = %d\n" % (
                cluster_idx,
                np.sum(np.where(d[cluster_idx] > np.mean(d) + distance_std * np.std(d), 1, 0)),
                np.sum(np.where(d[cluster_idx, self.db_helper.adhd_start_idx:self.db_helper.mipdb_start_idx] > np.mean(d) + distance_std * np.std(d), 1, 0)),
                np.sum(
                    np.where(d[cluster_idx, np.where(np.array(self.db_helper.adhd_label['DZ_G']) == 0)[0] + self.db_helper.adhd_start_idx] > np.mean(d) + distance_std * np.std(d),
                             1, 0)),
                np.sum(np.where(d[cluster_idx, self.db_helper.mipdb_start_idx:] > np.mean(d) + distance_std * np.std(d), 1, 0))
            ))
            print("Distance Cluster - %d : # of Total = %d, # of ADHD-db = %d, # of ADHD = %d, # of MIPDB = %d" % (
                cluster_idx,
                np.sum(np.where(d[cluster_idx] > np.mean(d) + distance_std * np.std(d), 1, 0)),
                np.sum(np.where(d[cluster_idx, self.db_helper.adhd_start_idx:self.db_helper.mipdb_start_idx] > np.mean(d) + distance_std * np.std(d), 1, 0)),
                np.sum(
                    np.where(d[cluster_idx, np.where(np.array(self.db_helper.adhd_label['DZ_G']) == 0)[0] + self.db_helper.adhd_start_idx] > np.mean(d) + distance_std * np.std(d),
                             1, 0)),
                np.sum(np.where(d[cluster_idx, self.db_helper.mipdb_start_idx:] > np.mean(d) + distance_std * np.std(d), 1, 0))
            ))

    def min_fuzzy_m(self, cluster_num):
        return 1 + \
               (1418 / self.db_helper.norm_total_data.shape[0] + 22.05) / np.power(cluster_num, 2) + \
               (12.33 / self.db_helper.norm_total_data.shape[0] + 0.243) * np.power(cluster_num, -0.0406 * np.log(self.db_helper.norm_total_data.shape[0]) - 0.1134)

    def fuzzy_score(self, alpha=5):
        en_f = self.network.compute_cluster_y(self.db_helper.norm_total_data).numpy()
        m = 1.2
        num_list = []
        fig, ax = plt.subplots(1, 1)
        for i in range(4, 30):
            total_f = 0
            total_s = 0
            cntr, u_, u0, d, jm, p, fpc = fuzz.cluster.cmeans(en_f.T, i, m, error=0.005, maxiter=1000, init=None)
            for j in range(u_.shape[1]):
                c = np.argmax(u_[:, j])
                a = np.mean(np.sqrt(np.sum(np.square(u_[:, np.where(np.argmax(u_, axis=0) == c)[0]] - u_[:, j:j + 1]), axis=1)))
                b = np.sqrt(np.min(np.sum(np.square(u_[:, np.where(np.argmax(u_, axis=0) != c)[0]] - u_[:, j:j + 1]), axis=0)))
                s = (b - a) / np.max([a, b])
                qp_idx = np.argsort(u_[:, j])[:-2]
                total_s += np.power(u_[qp_idx[1], j] - u_[qp_idx[0], j], alpha) * s
                total_f += np.power(u_[qp_idx[1], j] - u_[qp_idx[0], j], alpha)
            print("%d-%f: %f" % (i, m, total_s / total_f))
            num_list.append(total_s / total_f)
        ax.plot(num_list)
        fig.savefig('./f_12.png')
        plt.close(fig)
