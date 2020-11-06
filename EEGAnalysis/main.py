from deep_neural_net.Network import AutoEncoder
from deep_neural_net.Trainer import Trainer
from data.DBHelper import DBHelper
from analysis.AnalysisHelper import AnalysisHelper
from util import Util


def pretrain_network(db_helper, output_directory, max_epoch, k, beta=0.05, delta=5, gamma=1, noise_std=0.1):
    network = Util.make_network(feature_num=db_helper.norm_total_data.shape[1], k=k, m=0, beta=beta, delta=delta, gamma=gamma, noise_std=noise_std, name='network')

    trainer = Trainer(network=network,
                      db_helper=db_helper,
                      output_directory=output_directory)

    trainer.pretrain_network(max_epoch=max_epoch,
                             save_result_per_epoch=max_epoch)


def cluster_train_network(db_helper, pretrain_directory, pre_trained_network_epoch, max_epoch, cluster_train_directory, m=2, beta=0.05, delta=5, gamma=1, noise_std=0.1):
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
                          save_result_per_epoch=max_epoch, save_directory=cluster_train_directory)


def test_network(db_helper, pretrain_directory, cluster_train_directory, cluster_train_epoch, min_dist_list: list, nn_list: list, m: float, random_seed: int = 0, test_name=''):
    output_dir = './%s/%s/test_result_epoch%d_%s' % (pretrain_directory, cluster_train_directory, cluster_train_epoch, test_name)

    node_size_list = Util.parse_network(pretrain_directory)
    network = AutoEncoder(feature_num=db_helper.norm_total_data.shape[1], m=m, beta=0, delta=0, gamma=0, noise_std=0, pretrain_dir=pretrain_directory,
                          node_size_list=node_size_list, name='network')
    analysis_helper = AnalysisHelper(network, db_helper=db_helper)
    analysis_helper.analysis_umap(min_dist_list=min_dist_list, nn_list=nn_list, random_seed=random_seed,
                                  output_directory=output_dir)

    analysis_helper.fuzzy_cluster(cluster_num=12,
                                  m=m,
                                  min_dist=None)

    color_list = ['red', 'orange', 'olive', 'yellow', 'greenyellow', 'lightgreen', 'springgreen', 'lightseagreen', 'dodgerblue', 'navy', 'indigo', 'violet', 'magenta', 'hotpink']

    analysis_helper.plot_cluster_umap(min_dist_list=min_dist_list, nn_list=nn_list, output_directory=output_dir, color_list=color_list, cluster_array=None)

    analysis_helper.plot_bar_graph(output_dir, color_list=color_list, cluster_array=None)


def main():
    db_helper = DBHelper(
        hbn_data_path='/Users/sangminlee/Documents/YBRAIN/DB/child_data/',
        adhd_data_path="/Users/sangminlee/Documents/YBRAIN/DB/ADHD_sooncheonhyang/")
    db_helper.load_data(hbn_outlier_remove=True, use_abs_pow=False)

    pretrain_dir = 'pretrain_k8'
    cluster_train_dir = 'train_m2'

    # pretrain_network(db_helper=db_helper, output_directory=pretrain_dir, max_epoch=5, k=12)
    # cluster_train_network(db_helper=db_helper, pretrain_directory=pretrain_dir, pre_trained_network_epoch=5, max_epoch=5, cluster_train_directory=cluster_train_dir, m=2)
    test_network(db_helper=db_helper, pretrain_directory=pretrain_dir, cluster_train_directory=cluster_train_dir, cluster_train_epoch=5000,
                 min_dist_list=[0.15, 0.3, 0.5, 1.], nn_list=[3, 5, 10, 20, 50], m=1.2, test_name='test_m_1_2')


if __name__ == '__main__':
    main()
