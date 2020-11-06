from deep_neural_net.Network import AutoEncoder


class Util(object):
    @staticmethod
    def parse_network(pretrain_dir: str):
        f = open(('./%s/network_info.txt' % pretrain_dir), 'r')
        info_lines = f.readlines()
        kernel_size_list = []
        for i in range(int(len(info_lines) / 2)):
            if 'dense/bias:0' in info_lines[2 * i]:
                kernel_size_list.append(int(info_lines[2 * i + 1].split(',')[0][1:]))
            elif 'dense_1/bias:0' in info_lines[2 * i]:
                kernel_size_list.append(int(info_lines[2 * i + 1].split(',')[0][1:]))
            elif 'dense_2/bias:0' in info_lines[2 * i]:
                kernel_size_list.append(int(info_lines[2 * i + 1].split(',')[0][1:]))
            elif 'dense_3/bias:0' in info_lines[2 * i]:
                kernel_size_list.append(int(info_lines[2 * i + 1].split(',')[0][1:]))
            elif 'dense_4/bias:0' in info_lines[2 * i]:
                kernel_size_list.append(int(info_lines[2 * i + 1].split(',')[0][1:]))
            elif 'discriminator_1/bias:0' in info_lines[2 * i]:
                kernel_size_list.append(int(info_lines[2 * i + 1].split(',')[0][1:]))
            elif 'discriminator_2/bias:0' in info_lines[2 * i]:
                kernel_size_list.append(int(info_lines[2 * i + 1].split(',')[0][1:]))
            elif 'discriminator_3/bias:0' in info_lines[2 * i]:
                kernel_size_list.append(int(info_lines[2 * i + 1].split(',')[0][1:]))

        return kernel_size_list

    @staticmethod
    def make_network(feature_num, k=8, m=2, beta=0.05, delta=5, gamma=1, noise_std=0.1, pretrain_dir=None, name='network'):
        net = AutoEncoder(name=name, feature_num=feature_num, k=k, m=m, beta=beta, delta=delta, gamma=gamma, noise_std=noise_std, pretrain_dir=pretrain_dir)
        return net
