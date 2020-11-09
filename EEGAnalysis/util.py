from deep_neural_net.Network import AutoEncoder
import mne
import os
import pyedflib
import datetime


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

    @staticmethod
    def write_edf(mne_raw, fname, picks=None, tmin=0, tmax=None, overwrite=False):
        if not issubclass(type(mne_raw), mne.io.BaseRaw):
            raise TypeError('Must be mne.io.Raw type')
        if not overwrite and os.path.exists(fname):
            return
            raise OSError('File already exists. No overwrite.')
        # static settings
        file_type = pyedflib.FILETYPE_EDFPLUS
        sfreq = mne_raw.info['sfreq']
        first_sample = int(sfreq * tmin)
        last_sample = int(sfreq * tmax) if tmax is not None else None

        # convert data
        channels = mne_raw.get_data(picks,
                                    start=first_sample,
                                    stop=last_sample)

        # set conversion parameters
        dmin, dmax = [-32768, 32767]
        pmin, pmax = [channels.min(), channels.max()]
        n_channels = len(channels)

        # create channel from this
        try:
            f = pyedflib.EdfWriter(fname,
                                   n_channels=n_channels,
                                   file_type=file_type)

            channel_info = []
            data_list = []

            for i in range(n_channels):
                ch_dict = {'label': mne_raw.ch_names[i],
                           'dimension': 'uV',
                           'sample_rate': sfreq,
                           'physical_min': pmin,
                           'physical_max': pmax,
                           'digital_min': dmin,
                           'digital_max': dmax,
                           'transducer': '',
                           'prefilter': ''}

                channel_info.append(ch_dict)
                data_list.append(channels[i])

            f.setTechnician('mne-gist-save-edf-skjerns')
            f.setSignalHeaders(channel_info)
            f.writeSamples(data_list)
        except Exception as e:
            print(e)
            return False
        finally:
            f.close()
        return True
