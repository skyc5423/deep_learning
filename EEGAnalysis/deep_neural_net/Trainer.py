from data.DBHelper import DBHelper
from deep_neural_net import Network
import numpy as np
import os
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, network: Network, db_helper: DBHelper, output_directory: str):
        self.network = network
        self.db_helper = db_helper
        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.mkdir(self.output_directory)

    def pretrain_network(self, max_epoch=50000, save_result_per_epoch: int = None):
        feature_array = self.db_helper.norm_total_data
        save_flag = True
        for epoch_idx in range(1, max_epoch + 1):
            idx = np.arange(feature_array.shape[0])
            np.random.shuffle(idx)
            for iter_idx in range(1):
                random_idx = idx[iter_idx * int(idx.shape[0] / 1):(iter_idx + 1) * int(idx.shape[0] / 1)]
                loss_tmp, loss_encoder, loss_decoder, loss_cluster = self.network.train_(feature_array[random_idx], epoch_idx, cluster=False)
            print("Pretrain %02d: total=%f, encoder=%f, decoder=%f, cluster=%f" % (epoch_idx, loss_tmp, loss_encoder, loss_decoder, loss_cluster))

            if save_result_per_epoch is not None:
                if epoch_idx % save_result_per_epoch == 0:
                    if not os.path.isdir('./%s/pretrain_result/' % self.output_directory):
                        os.mkdir('./%s/pretrain_result/' % self.output_directory)
                    fig, ax = plt.subplots(4, 4)
                    for i in range(4):
                        for j in range(4):
                            ax[i, j].plot(feature_array[random_idx[4 * i + j]], color='blue', linewidth=0.3)
                            ax[i, j].plot(self.network(feature_array[random_idx[4 * i + j:4 * i + j + 1]])[1][0], color='red', linewidth=0.3)
                    fig.savefig('./%s/pretrain_result/tmp_%d.png' % (self.output_directory, epoch_idx))
                    plt.close(fig)

                    self.network.save_weights('./%s/pre_trained_network_at_%d' % (self.output_directory, epoch_idx))

                    if save_flag:
                        network_info = open('./%s/network_info.txt' % self.output_directory, 'w')
                        for weight in self.network.weights:
                            network_info.write(str(weight.name) + '\n')
                            network_info.write(str(weight.shape) + '\n')

                        network_info.close()
                        save_flag = False

        self.network.save_weights('./%s/pre_trained_network' % self.output_directory)
        if save_flag:

            network_info = open('./%s/network_info.txt' % self.output_directory, 'w')
            for weight in self.network.weights:
                network_info.write(str(weight.name) + '\n')
                network_info.write(str(weight.shape) + '\n')

            network_info.close()

    def train_network(self, pre_trained_network_epoch: int = None, max_epoch=50000, save_result_per_epoch: int = None, save_directory: str = None):

        if save_result_per_epoch is None:
            print('Argument save_result_per_epoch should be determined')
        if save_directory is None:
            print('Argument save_directory should be determined')

        if pre_trained_network_epoch is None:
            self.network.load_weights('./%s/%s' % (self.output_directory, 'pre_trained_network'))
        else:
            self.network.load_weights('./%s/%s' % (self.output_directory, 'pre_trained_network_at_%d' % pre_trained_network_epoch))

        if not os.path.isdir('./%s/%s' % (self.output_directory, save_directory)):
            os.mkdir('./%s/%s' % (self.output_directory, save_directory))

        if not os.path.isdir('./%s/%s/result_figure' % (self.output_directory, save_directory)):
            os.mkdir('./%s/%s/result_figure' % (self.output_directory, save_directory))

        loss_total_list = []
        loss_encoder_list = []
        loss_decoder_list = []
        loss_cluster_list = []

        feature_array = self.db_helper.norm_total_data

        for epoch_idx in range(1, max_epoch + 1):
            idx = np.arange(feature_array.shape[0])
            np.random.shuffle(idx)
            for iter_idx in range(1):
                random_idx = idx[iter_idx * int(idx.shape[0] / 1):(iter_idx + 1) * int(idx.shape[0] / 1)]
                loss_tmp, loss_encoder, loss_decoder, loss_cluster = self.network.train_(feature_array[random_idx], epoch_idx)
            print("Cluster Train %02d: total=%f, encoder=%f, decoder=%f, cluster=%f" % (epoch_idx, loss_tmp, loss_encoder, loss_decoder, loss_cluster))

            if epoch_idx % 10 == 0:
                loss_total_list.append(loss_tmp)
                loss_encoder_list.append(loss_encoder)
                loss_decoder_list.append(loss_decoder)
                loss_cluster_list.append(loss_cluster)

            if epoch_idx % save_result_per_epoch == 0:
                self.network.save_weights('./%s/%s/trained_network_%d' % (self.output_directory, save_directory, epoch_idx))
                fig, ax = plt.subplots(4, 1, figsize=(6, 12))
                ax[0].plot(10 * np.arange(len(loss_total_list)), loss_total_list)
                ax[1].plot(10 * np.arange(len(loss_total_list)), loss_encoder_list)
                ax[2].plot(10 * np.arange(len(loss_total_list)), loss_decoder_list)
                ax[3].plot(10 * np.arange(len(loss_total_list)), loss_cluster_list)
                fig.savefig('./%s/%s/result_figure/loss.png' % (self.output_directory, save_directory))
                plt.close(fig)
                fig, ax = plt.subplots(4, 4)
                for i in range(4):
                    for j in range(4):
                        ax[i, j].plot(feature_array[random_idx[4 * i + j]], color='blue', linewidth=0.3)
                        ax[i, j].plot(self.network(feature_array[random_idx[4 * i + j:4 * i + j + 1]])[1][0], color='red', linewidth=0.3)
                fig.savefig('./%s/%s/result_figure/decoded_%d.png' % (self.output_directory, save_directory, epoch_idx))
                plt.close(fig)
