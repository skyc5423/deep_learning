import numpy as np
from scipy.signal import welch
from scipy import fftpack
import math
import os
import json
from scipy.signal import hilbert
from data.preprocess.DataLoader import DataLoader


class FeatureExtractor(object):
    def __init__(self):
        self.raw = None
        self.reset_feature_value()

    def reset_feature_value(self):
        self.abs_power = None
        self.rel_power = None
        self.rat_power = None
        self.psd = None
        self.sef95 = None
        self.coherence = None
        self.samp_en = None
        self.cp = None

    def set_raw(self, raw):
        self.raw = raw
        self.reset_feature_value()

    @staticmethod
    def get_csd(x, fs, y=None, nperseg=None, nfft=1024):
        t = x.shape[-1]
        is_psd = False
        if nperseg is None:
            nperseg = int(t / 8)
        if y is None:
            y = np.array(x)
            is_psd = True
        num = int(t / nperseg)
        win = np.hamming(nperseg)
        pxy = np.zeros([num, nfft], dtype=np.complex64)
        U = np.dot(win, win) * fs
        for idx in range(num):
            x_idx = x[nperseg * idx:nperseg * (idx + 1)]
            y_idx = y[nperseg * idx:nperseg * (idx + 1)]
            wx_idx = x_idx * win
            wy_idx = y_idx * win

            x_reshape = np.zeros(nfft * (int(nperseg / nfft) + 1))
            x_reshape[:nperseg] = wx_idx
            x_reshape = np.reshape(x_reshape, [int(nperseg / nfft) + 1, nfft])
            x_sum = np.sum(x_reshape, 0)
            x_f = np.fft.fft(x_sum)

            y_reshape = np.zeros(nfft * (int(nperseg / nfft) + 1))
            y_reshape[:nperseg] = wy_idx
            y_reshape = np.reshape(y_reshape, [int(nperseg / nfft) + 1, nfft])
            y_sum = np.sum(y_reshape, 0)
            y_f = np.fft.fft(y_sum)

            pxy[idx] = x_f * np.conjugate(y_f) / U

        if is_psd:
            res = np.abs(np.mean(pxy, 0))
        else:
            res = np.mean(pxy, 0)
        return fftpack.fftfreq(nfft, 1 / fs), res

    def analysis_power(self, bands=None):

        from scipy.signal import welch
        if bands is None:
            bands = {'Delta': [1, 4],
                     'Theta': [4, 8],
                     'Alpha': [8, 12],
                     'Beta': [12, 25],
                     'High Beta': [25, 30],
                     'Gamma': [30, 40]}

        fs = self.raw.info['sfreq']
        input_data = self.raw.get_data()
        data_ = np.array(input_data)
        data = data_
        # fft_for_psd = np.zeros([data.shape[0], data.shape[1]])
        for ch in range(data.shape[0]):
            # fft_for_psd[ch] = np.abs(np.fft.fft(data[ch]))
            f, psd_tmp = welch(data[ch], fs=fs, nfft=fs * 32, nperseg=fs * 2, noverlap=fs)
            if ch == 0:
                psd = np.expand_dims(psd_tmp, 0)
            else:
                psd = np.concatenate([psd, np.expand_dims(psd_tmp, 0)], axis=0)

        abs_power = {}
        rel_power = {}

        for band in bands.keys():
            band_freq = bands.get(band)
            band_idx = np.where(np.where(f >= band_freq[0], f, 999) <= band_freq[1])[0]

            power = np.mean(psd[:, band_idx], axis=1) * (band_freq[1] - band_freq[0])

            abs_power[band] = power

        total_power = np.zeros(psd.shape[0])
        for ch in range(psd.shape[0]):
            for band in abs_power.keys():
                total_power[ch] += abs_power.get(band)[ch]

        for band in abs_power.keys():
            rel_power[band] = abs_power.get(band) / total_power

        rat_power = {}

        rat_power['DAR'] = abs_power['Delta'] / abs_power['Alpha']
        rat_power['TAR'] = abs_power['Theta'] / abs_power['Alpha']
        rat_power['TBR'] = abs_power['Theta'] / abs_power['Beta']

        self.abs_power = abs_power
        self.rel_power = rel_power
        self.rat_power = rat_power
        self.psd = psd

    def analysis_sef95(self):
        ori_data = self.raw.get_data()
        mean_data = np.mean(ori_data, axis=0)
        f, y = welch(mean_data, fs=self.raw.info['sfreq'], nperseg=int(self.raw.info['sfreq'] * 4), nfft=int(self.raw.info['sfreq'] * 10))
        se = 0
        min = np.where(f == 1.)[0][0]
        max = np.where(f == 50.)[0][0]
        for i in range(min, y.shape[0]):
            se += y[i] / np.sum(y[min:max + 1])
            if se > 0.95:
                break
        self.sef95 = f[i]

    def analysis_coherence(self, bands=None, for_phase=False):
        if bands is None:
            bands = {'Delta': [1, 4],
                     'Theta': [4, 8],
                     'Alpha': [8, 12],
                     'Beta': [12, 25],
                     'High Beta': [25, 30],
                     'Gamma': [30, 40]}

        input_data = self.raw.get_data()
        fs = self.raw.info['sfreq']

        window_size = 1
        while True:
            if fs / window_size >= (0.25 + 0.125):
                window_size *= 2
            else:
                break

        overlap_size = int(window_size * 0.25)
        n_f = int(window_size * 0.5)
        total_batch = int((input_data.shape[1] - window_size) / overlap_size)

        coherence = {}
        for keys in bands.keys():
            coherence[keys] = np.zeros([total_batch, 19, 19])

        for t in range(total_batch):
            for n in range(19):
                for m in range(19):
                    f, csd = self.get_csd(input_data[n, t * overlap_size:t * overlap_size + window_size], fs, input_data[m, t * overlap_size:t * overlap_size + window_size],
                                          nfft=n_f,
                                          nperseg=1024)
                    _, py1 = self.get_csd(input_data[n, t * overlap_size:t * overlap_size + window_size], fs=fs, nfft=n_f, nperseg=n_f)
                    _, py2 = self.get_csd(input_data[m, t * overlap_size:t * overlap_size + window_size], fs=fs, nfft=n_f, nperseg=n_f)
                    tmp = csd / (np.sqrt(py1) + 1E-7) / (np.sqrt(py2) + 1E-7)
                    Cyy = (tmp.imag)

                    for key in bands.keys():
                        low, high = bands[key][0], bands[key][1]
                        freq_bands_indicator = ((low <= f) & (f < high))
                        if for_phase:
                            coherence[key][t, n, m] = np.mean(Cyy[freq_bands_indicator])
                        else:
                            coherence[key][t, n, m] = np.abs(np.mean(Cyy[freq_bands_indicator]))

        for key in bands.keys():
            coherence[key] = np.mean(coherence[key], 0)

        self.coherence = coherence

    # def analysis_asymmetry(self):
    #
    #     if isinstance(power, dict):
    #         asym = {}
    #         for key in power.keys():
    #             asym_tmp = np.zeros([19, 19])
    #             for n in range(19):
    #                 for m in range(19):
    #                     asym_tmp[n, m] = np.log(power[key][n]) - np.log(power[key][m])
    #             asym[key] = asym_tmp
    #     else:
    #         asym = np.zeros([19, 19])
    #         for n in range(19):
    #             for m in range(19):
    #                 asym[n, m] = np.log(power[n]) - np.log(power[m])
    #
    #     return asym
    @staticmethod
    def down_sampling(data, fs, target_fs):
        ch_num = data.shape[0]
        scale = fs / target_fs
        if not scale.is_integer():
            scale = round(scale)
        else:
            scale = int(scale)

        len_used = int(data.shape[1] / scale) * scale
        data_tmp = data[:, :len_used]
        new_data = data_tmp.reshape([ch_num, -1, scale])

        return np.mean(new_data, axis=2)

    def analysis_samp_entropy(self, target_fs=250, time_window=5, m_0=1, r=0.25):
        ori_data = self.raw.get_data()
        fs = int(self.raw.info['sfreq'])

        data = self.down_sampling(ori_data, fs, target_fs)
        N = data.shape[1]
        window = int(target_fs * time_window)
        samp_en = np.zeros(ori_data.shape[0])
        m_1 = m_0 + 1
        for epoch_idx in range(int(N / window)):
            for ch in range(ori_data.shape[0]):
                cur_data = np.array(data[ch, epoch_idx * window:epoch_idx * window + window])
                cur_std = np.std(cur_data)
                cur_r = cur_std * r
                mi = np.zeros([window + 1, window + 1, 2, m_0])
                for i in range(m_0):
                    tmp = np.meshgrid(cur_data[i:], cur_data[i:])
                    mi[:window - i, :window - i, 0, i] = tmp[0]
                    mi[:window - i, :window - i, 1, i] = tmp[1]

                mi = np.array(mi[:-m_0, :-m_0])
                B = (np.sum(np.where(np.max(np.abs(mi[:, :, 0, :] - mi[:, :, 1, :]), axis=2) <= cur_r, 1, 0)) - (window + 1 - m_0)) / ((window - m_0 - 1) * float(window - m_0))

                mi = np.zeros([window + 1, window + 1, 2, m_1])
                for i in range(m_1):
                    tmp = np.meshgrid(cur_data[i:], cur_data[i:])
                    mi[:window - i, :window - i, 0, i] = tmp[0]
                    mi[:window - i, :window - i, 1, i] = tmp[1]

                mi = np.array(mi[:-m_1, :-m_1])
                A = (np.sum(np.where(np.max(np.abs(mi[:, :, 0, :] - mi[:, :, 1, :]), axis=2) <= cur_r, 1, 0)) - (window + 1 - m_1)) / ((window - m_1 - 1) * float(window - m_1))

                samp_en[ch] += -np.log(A / B)
        self.samp_en = samp_en / float((N / window) * 1)

    def analysis_coupling_analysis(self, window_sec=2, overlap_size=200, phase_freq=[4, 8], amp_freq=[30, 60], mi_bin_size=18):
        cp = np.zeros([self.raw.get_data().shape[0], self.raw.get_data().shape[0]])
        window_size = int(self.raw.info['sfreq'] * window_sec)
        for ch1 in range(self.raw.get_data().shape[0]):
            for ch2 in range(self.raw.get_data().shape[0]):
                total_phase = []
                total_gamma_amp = []
                data_p = np.mean(self.raw.get_data()[[ch1]], axis=0)
                data_f = np.mean(self.raw.get_data()[[ch2]], axis=0)
                total_step = int((data_f.shape[0] - window_size) / overlap_size + 1)
                for t in range(total_step):
                    cur_frontal = data_f[t:t + window_size]
                    cur_hil_frontal = hilbert(cur_frontal)[
                                      int(window_size * (phase_freq[0] / self.raw.info['sfreq'])):int(window_size * (phase_freq[1] / self.raw.info['sfreq']))]
                    cur_phase = np.angle(cur_hil_frontal)
                    total_phase.append(cur_phase)
                    cur_parietal = data_p[t:t + window_size]
                    cur_hil_parietal = hilbert(cur_parietal)[
                                       int(window_size * (amp_freq[0] / self.raw.info['sfreq'])):int(window_size * (amp_freq[1] / self.raw.info['sfreq']))]
                    cur_gamma_amp = np.abs(cur_hil_parietal)
                    total_gamma_amp.append(cur_gamma_amp)
                total_phase = np.array(total_phase)
                total_gamma_amp = np.array(total_gamma_amp)
                # total_gamma_amp = np.log(total_gamma_amp)
                coupling = np.zeros([total_phase.shape[1], total_gamma_amp.shape[1]])
                for n in range(total_phase.shape[1]):
                    for m in range(total_gamma_amp.shape[1]):
                        cur_phase = total_phase[:, n]
                        cur_gamma_amp = total_gamma_amp[:, m]
                        V1 = cur_phase.copy()
                        V2 = cur_gamma_amp.copy()
                        q1 = np.linspace(-180, 180, mi_bin_size + 1)  # min ~ max 까지 50개의 영역 생성
                        N = total_step
                        M1 = np.zeros(mi_bin_size)
                        avg_amp = np.zeros(mi_bin_size)
                        for step in range(N):
                            x = V1[step] / math.pi * 180
                            y = V2[step]
                            for i in range(mi_bin_size):
                                if (x >= q1[i]) and (x < q1[i + 1]):
                                    break
                            # for j in range(mi_bin_size):
                            #     if (y >= q2[j]) and (y < q2[j + 1]):
                            #         break
                            M1[i] += 1.
                            # M2[j] += 1.
                            avg_amp[i] += y
                        # print(M1)
                        MI = 0
                        for i in range(M1.shape[0]):
                            if M1[i] == 0.:
                                continue
                            avg_amp[i] /= M1[i]
                        avg_amp = avg_amp / sum(avg_amp)
                        for i in range(M1.shape[0]):
                            MI -= avg_amp[i] * np.log(avg_amp[i] + 1E-9)
                        coupling[n, m] = (np.log(mi_bin_size) - MI) / np.log(mi_bin_size)
                cp[ch1, ch2] = np.mean(coupling)
        self.cp = cp

    def make_jsonable_data(self, data):
        if isinstance(data, dict):
            for key in data.keys():
                tmp_data = data[key]
                data[key] = self.make_jsonable_data(tmp_data)
            return data
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

    def make_json_from_data(self):
        json_data = {}
        if self.abs_power is not None:
            json_data['abs_power'] = self.make_jsonable_data(self.abs_power)
        if self.rel_power is not None:
            json_data['rel_power'] = self.make_jsonable_data(self.rel_power)
        if self.rat_power is not None:
            json_data['rat_power'] = self.make_jsonable_data(self.rat_power)
        if self.sef95 is not None:
            json_data['sef95'] = self.make_jsonable_data(self.sef95)
        if self.coherence is not None:
            json_data['coherence'] = self.make_jsonable_data(self.coherence)
        if self.samp_en is not None:
            json_data['samp_en'] = self.make_jsonable_data(self.samp_en)
        if self.cp is not None:
            json_data['cp'] = self.make_jsonable_data(self.cp)

        json_data = json.dumps(json_data)
        return json_data

    def save_feature_at(self, output_path, file_name):

        json_data = self.make_json_from_data()
        print(os.path.join(output_path, '%s.json' % file_name))
        f = open(os.path.join(output_path, '%s.json' % file_name), 'w')
        f.write(json_data)
        f.close()

    def analysis_data(self, data_loader: DataLoader, analysis_power, analysis_coherence, analysis_sef95, analysis_samp_entropy, analysis_coupling_analysis, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        for crt_data in data_loader.total_data_list:
            try:
                crt_raw = data_loader.read_edf_file(crt_data)
                crt_file_name = crt_data.split('/')[-1].split('.')[0]
                self.set_raw(crt_raw)
                if analysis_power:
                    self.analysis_power()
                if analysis_coherence:
                    self.analysis_coherence()
                if analysis_sef95:
                    self.analysis_sef95()
                if analysis_samp_entropy:
                    self.analysis_samp_entropy()
                if analysis_coupling_analysis:
                    self.analysis_coupling_analysis()

                self.save_feature_at(output_path, crt_file_name)
            except:
                continue