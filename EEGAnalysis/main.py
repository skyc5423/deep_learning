from deep_neural_net.Network import AutoEncoder
from deep_neural_net.Trainer import Trainer
from data.DBHelper import DBHelper
from analysis.AnalysisHelper import AnalysisHelper
from util import Util
import numpy as np


def pretrain_network(db_helper, output_directory, max_epoch, k, save_result_per_epoch, beta=0.05, delta=5, gamma=1, noise_std=0.1):
    network = Util.make_network(feature_num=db_helper.norm_total_data.shape[1], k=k, m=0, beta=beta, delta=delta, gamma=gamma, noise_std=noise_std, name='network')

    trainer = Trainer(network=network,
                      db_helper=db_helper,
                      output_directory=output_directory)

    trainer.pretrain_network(max_epoch=max_epoch,
                             save_result_per_epoch=save_result_per_epoch)


def cluster_train_network(db_helper, pretrain_directory, pre_trained_network_epoch, max_epoch, save_result_per_epoch, cluster_train_directory, m=2., beta=0.05, delta=5, gamma=1,
                          noise_std=0.1):
    node_size_list = Util.parse_network(pretrain_directory)
    network = AutoEncoder(feature_num=db_helper.norm_total_data.shape[1],
                          m=m,
                          beta=beta,
                          delta=delta,
                          gamma=gamma,
                          noise_std=noise_std,
                          pretrain_dir=pretrain_directory,
                          node_size_list=node_size_list,
                          name='network')

    trainer = Trainer(network=network,
                      db_helper=db_helper,
                      output_directory=pretrain_directory)
    trainer.train_network(pre_trained_network_epoch=pre_trained_network_epoch,
                          max_epoch=max_epoch,
                          save_result_per_epoch=save_result_per_epoch, save_directory=cluster_train_directory)


def test_network(db_helper, pretrain_directory, cluster_train_directory, cluster_train_epoch, cluster_num: int, min_dist_list: list, nn_list: list, m: float, random_seed: int = 0,
                 test_name='', distance_std: float = 1.):
    output_dir = './%s/%s/test_result_epoch%d_%s' % (pretrain_directory, cluster_train_directory, cluster_train_epoch, test_name)

    node_size_list = Util.parse_network(pretrain_directory)
    network = AutoEncoder(feature_num=db_helper.norm_total_data.shape[1], m=m, beta=0, delta=0, gamma=0, noise_std=0, pretrain_dir=pretrain_directory,
                          load_weight_path='./%s/%s/trained_network_%d' % (pretrain_directory, cluster_train_directory, cluster_train_epoch),
                          node_size_list=node_size_list, name='network')
    network.load_network_weight()

    analysis_helper = AnalysisHelper(network, db_helper=db_helper)
    analysis_helper.analysis_umap(min_dist_list=min_dist_list, nn_list=nn_list, random_seed=random_seed,
                                  output_directory=output_dir)

    analysis_helper.fuzzy_cluster(cluster_num=cluster_num,
                                  m=m,
                                  min_dist=None)

    color_list = ['red', 'orange', 'olive', 'yellow', 'greenyellow', 'lightgreen', 'springgreen', 'lightseagreen', 'dodgerblue', 'navy', 'indigo', 'violet',
                  'magenta', 'hotpink', 'maroon', 'plum', 'rosybrown', 'gainsboro', 'sienna', 'powderblue']

    # analysis_helper.plot_cluster_umap(min_dist_list=min_dist_list, nn_list=nn_list, output_directory=output_dir, color_list=color_list, cluster_array=None)
    analysis_helper.plot_cluster_umap_distance(min_dist_list=min_dist_list, nn_list=nn_list, output_directory=output_dir, color_list=color_list, cluster_array=None,
                                               distance_std=distance_std)

    analysis_helper.random_seed_umap(min_dist=0.2, nn=10, random_seed_list=list(np.arange(9) * 100), output_directory=output_dir, color_list=color_list, distance_std=distance_std)
    # analysis_helper.plot_bar_graph(output_dir, color_list=color_list, cluster_array=None)
    analysis_helper.plot_bar_graph_distance(output_dir, color_list=color_list, cluster_array=None, distance_std=distance_std)


def main():
    db_helper = DBHelper(
        # hbn_data_path='/home/ybrain-analysis/문서/feature/child_data',
        # adhd_data_path="/home/ybrain-analysis/문서/feature/ADHD_sooncheonhyang",
        # mipdb_data_path="/home/ybrain-analysis/문서/feature/MIPDB",

        hbn_data_path='/Users/sangminlee/Documents/YBRAIN/DB/child_data',
        adhd_data_path="/Users/sangminlee/Documents/YBRAIN/DB/ADHD_sooncheonhyang",
        mipdb_data_path="/Users/sangminlee/Documents/YBRAIN/DB/MIPDB"
    )
    db_helper.load_data(hbn_outlier_remove=True, use_abs_pow=False, region_combine_power=True)

    pretrain_dir = 'pretrain_v2_k8_total_power_'
    cluster_train_dir = 'train_v2_m20'

    # pretrain_network(db_helper=db_helper, output_directory=pretrain_dir, max_epoch=10000, save_result_per_epoch=1000, k=8)
    # cluster_train_network(db_helper=db_helper, pretrain_directory=pretrain_dir, pre_trained_network_epoch=10000, max_epoch=100000, save_result_per_epoch=2000,
    #                       cluster_train_directory=cluster_train_dir, m=1.2)
    # 21000 40000
    test_network(db_helper=db_helper, pretrain_directory=pretrain_dir, cluster_train_directory=cluster_train_dir, cluster_train_epoch=100000, cluster_num=12,
                 min_dist_list=[0.2, 0.5, 1.], nn_list=[5, 10, 20, 50], m=1.2, test_name='test_m_12_c_12_d_15', distance_std=1.5)


if __name__ == '__main__':
    main()
