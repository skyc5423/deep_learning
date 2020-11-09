import mne

mne.set_config('MNE_LOGGING_LEVEL', 'warning')
import data.preprocess.DataLoader
import numpy as np
import scipy
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches
import os
import tensorflow as tf


class Preprocessor(object):
    def __init__(self):
        self.tf_model = None
        self.init_tf_model()

    def init_tf_model(self):
        print(tf.config.list_physical_devices('GPU'))
        model_path = './data/preprocess/50'
        imported = tf.saved_model.load(model_path)
        self.tf_model = imported

    @staticmethod
    def preprocess_filtering(raw):
        raw.load_data()
        raw.filter(l_freq=1.0, h_freq=60, fir_design='firwin', phase='zero')
        raw.notch_filter(60, trans_bandwidth=3, notch_widths=3, filter_length='auto', phase='zero')
        return raw

    @staticmethod
    def artifact_remove_asr(custom_raw, overlap_ratio=0.05):
        filtered_data = custom_raw.get_data()
        sample_rate = int(custom_raw.info['sfreq'])
        overlap_size = int(sample_rate * overlap_ratio)
        total_step = int((filtered_data.shape[1] - sample_rate + overlap_size) / overlap_size)
        removed_asr = np.zeros([overlap_size * total_step + sample_rate - overlap_size, filtered_data.shape[0]])
        ma_data = filtered_data

        for step in range(total_step):
            u, s, vh = np.linalg.svd(ma_data[:, step * overlap_size: (step + int(sample_rate / overlap_size)) * overlap_size].T, full_matrices=False)
            for ch in range(filtered_data.shape[0]):
                if np.std(u[:, ch]) * s[ch] > 80 or np.max(u[:, ch]) - np.mean(u[:, ch]) > 0.25 or np.min(u[:, ch]) - np.mean(u[:, ch]) < -0.25 or np.abs(
                        scipy.stats.kurtosis(u[:, ch])) > 1:
                    u[:, ch] *= 0.
            smat = np.diag(s)
            removed_asr[step * overlap_size: (step + int(sample_rate / overlap_size)) * overlap_size, :] += np.dot(u, np.dot(smat, vh))

        removed_ma = removed_asr.T * overlap_size / float(sample_rate)

        raw = mne.io.RawArray(removed_ma, custom_raw.info)

        return raw

    def predict_artifact(self, model, topomap):
        N = 100
        res_bin = np.zeros([N, 4])
        for i in range(N):
            res_bin[i] = model(np.expand_dims(topomap, 0), False)

        res = np.expand_dims(np.mean(res_bin, 0), 0)
        res_std = np.expand_dims(np.std(res_bin, 0), 0)
        res_var = res_std ** 2

        u = res[0, 0]
        var = res_var[0, 0]
        a = u * (u * (1 - u) / var - 1)
        b = (1 - u) * (u * (1 - u) / var - 1)

        p_val = beta.cdf(0.5, a, b)

        if p_val < 0.05:
            is_artifact = False
        else:
            is_artifact = True
        return is_artifact, np.argmax(res), res_bin, p_val

    @staticmethod
    def plot_ica_topomap(power, pos, outlines):
        aspect_ratio = [(255. - 64.) / (897. - 128.) * (10. / 3.), (271. - 16.) / (889. - 120.) * (10. / 3.)]

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        # interpolate data
        pos_ = np.zeros(pos.shape)
        pos_[:, 0] = pos[:, 0] * aspect_ratio[0]
        pos_[:, 1] = pos[:, 1] * aspect_ratio[1]

        xlim = np.inf, -np.inf,
        ylim = np.inf, -np.inf,
        mask_ = np.c_[outlines['mask_pos']]
        mask_[:, 0] *= aspect_ratio[0]
        mask_[:, 1] *= aspect_ratio[1]
        xmin, xmax = (np.min(np.r_[xlim[0], mask_[:, 0]]),
                      np.max(np.r_[xlim[1], mask_[:, 0]]))
        ymin, ymax = (np.min(np.r_[ylim[0], mask_[:, 1]]),
                      np.max(np.r_[ylim[1], mask_[:, 1]]))

        from scipy.interpolate import griddata

        tmpx, tmpy = np.mgrid[np.min(pos_[:, 0]):np.max(pos_[:, 0]):128j, np.min(pos_[:, 1]):np.max(pos_[:, 1]):128j]
        gt = griddata(pos_ * 1.07, power, (tmpx, tmpy), 'cubic')
        gt = np.flip(np.transpose(gt), 0)

        im = ax.imshow(gt, cmap='jet', vmin=-np.max([np.abs(np.min(power)), np.max(power)]), vmax=np.max([np.abs(np.min(power)), np.max(power)]),
                       extent=(xmin, xmax, ymin, ymax))
        patch_ = patches.Ellipse((0, 0),
                                 1 * aspect_ratio[0],
                                 1 * aspect_ratio[1],
                                 clip_on=True,
                                 transform=ax.transData)
        ax.axis('off')

        if patch_ is not None:
            im.set_clip_path(patch_)

        ax.figure.canvas.draw()
        rtn = np.array(ax.figure.canvas.renderer._renderer)

        for ch in range(19):
            ax.plot(pos_[ch, 0] * 1.15, pos_[ch, 1] * 1.15, 'r', marker='o', markersize=2)
        #
        # if not os.path.isdir('figure_out/%s' % request_id):
        #     os.mkdir('figure_out/%s' % request_id)
        # ax.figure.savefig("figure_out/" + request_id + "/ica_sources_%d_%d.png" % (idx + 1, sign))

        rtn_tmp = np.array(ax.figure.canvas.renderer._renderer)

        plt.close(fig)
        return rtn[:, :, 0:3] / 255., rtn_tmp[:, :, 0:3] / 255.

    def artifact_remove_ica(self, custom_raw):
        n_components = 19
        ica = mne.preprocessing.ICA(n_components=n_components, method="infomax", random_state=0, max_iter=50000, fit_params=dict(extended=True))
        ica.fit(custom_raw, reject_by_annotation=False)
        pos = mne.viz.topomap._prepare_topo_plot(custom_raw, 'eeg', None)[1]
        pos, outlines = mne.viz.topomap._check_outlines(pos, 'head', None)
        ica_exclude = []

        for i in range(n_components):
            is_artifact = True
            for sign_idx in range(2):
                if sign_idx == 0:
                    sign = -1
                else:
                    sign = 1

                # make ica component topomaps for find artifact
                topomap, topomap_tmp = self.plot_ica_topomap(sign * ica.get_components()[:, i], pos, outlines)
                is_artifact_tmp, result, prob, p_val = self.predict_artifact(self.tf_model, topomap)
                if not is_artifact_tmp:
                    is_artifact = False
            if is_artifact:
                ica_exclude.append(i)

        for source in ica_exclude:
            ica.exclude.append(source)

        print("Zeroed out sources idx: " + str(ica.exclude))

        raw_d = np.array(custom_raw.get_data())
        for ch in range(19):
            raw_d[ch] -= ica.pca_mean_[ch]
        pca_d = np.dot(ica.pca_components_, raw_d)
        s = np.dot(ica.unmixing_matrix_, pca_d)
        for ch in ica_exclude:
            s[ch] *= 0
        pca_d_clean = np.dot(np.linalg.inv(ica.unmixing_matrix_), s)
        raw_d_clean = np.dot(np.linalg.inv(ica.pca_components_), pca_d_clean)
        for ch in range(19):
            raw_d_clean += ica.pca_mean_[ch]

        new_custom_raw = mne.io.RawArray(raw_d_clean, custom_raw.info)

        return new_custom_raw
