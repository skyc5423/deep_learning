import numpy as np
import os
import pandas as pd
import json
import math
from scipy.stats import gamma
from util import Util


class DBHelper(object):
    def __init__(self, hbn_data_path=None, adhd_data_path=None, mipdb_data_path=None):
        self.hbn_data_path = hbn_data_path
        self.adhd_data_path = adhd_data_path
        self.mipdb_data_path = mipdb_data_path

        self.total_data = None
        self.norm_total_data = None
        self.hbn_data = None
        self.adhd_data = None
        self.mipdb_data = None

        self.hbn_label = None
        self.adhd_label = None
        self.mipdb_label = None

        self.hbn_age = None
        self.adhd_age = None
        self.mipdb_age = None
        self.total_age = None

        self.adhd_start_idx = -1
        self.mipdb_start_idx = -1

        self.feature_name_list = None

    def age_normalize_feature(self):
        k_theta_arr = np.load('./data/ref_file/k_theta_arr_v2.npy')
        z_score = np.zeros_like(self.total_data)
        for age_idx in range(13):
            crt_idx = np.where(self.total_age == age_idx + 5)[0]
            for feature_idx in range(self.total_data.shape[1]):
                crt_k, crt_theta = k_theta_arr[age_idx, feature_idx]
                z_score[crt_idx, feature_idx] = gamma(a=crt_k, scale=crt_theta).cdf(self.total_data[crt_idx, feature_idx])

        self.norm_total_data = 2 * (z_score - 0.5)

    def remove_outlier(self, feature_array, age_arr, feature_idx):
        flag_outlier, new_feature_array, new_age_arr = self.detect_outlier(feature_array, age_arr, feature_idx)
        if flag_outlier:
            return self.remove_outlier(new_feature_array, new_age_arr, feature_idx)
        else:
            return new_feature_array, new_age_arr

    def detect_outlier(self, feature_array, age_arr, feature_idx):
        idx_feature_array = feature_array[:, feature_idx]
        gamma_dist, _, _ = Util.get_gamma_dist(idx_feature_array)
        normal_idx = np.where(np.where(gamma_dist.cdf(idx_feature_array) > 0.0005, gamma_dist.cdf(idx_feature_array), 999) < 0.9995)[0]
        if normal_idx.shape[0] == feature_array.shape[0]:
            return False, feature_array, age_arr
        else:
            return True, feature_array[normal_idx], age_arr[normal_idx]

    def outlier_removal_hbn(self):
        new_tbr_arr, new_label_arr = np.array(self.hbn_data), np.array(self.hbn_label)
        for i in range(30):
            for feature_idx in range(self.hbn_data.shape[1]):
                new_tbr_arr, new_label_arr = self.remove_outlier(new_tbr_arr, new_label_arr, feature_idx)

        self.hbn_data, self.hbn_label = new_tbr_arr, new_label_arr

    def flatten(self, dict_data, y_ch_list):
        feature_list = []
        feature_name_list = []
        for key in dict_data.keys():
            if key in ['abs_power', 'rel_power', 'rat_power']:
                for band_key in dict_data[key].keys():
                    for ch in range(19):
                        feature_list.append(np.array(dict_data[key][band_key])[:, ch])
                        feature_name_list.append('%s_%s_%s' % (key, band_key, y_ch_list[ch]))
            elif key in ['alpha_peak', 'alpha_peak_power']:
                for ch in range(19):
                    feature_list.append(np.array(dict_data[key])[:, ch])
                    feature_name_list.append('%s_%s' % (key, y_ch_list[ch]))
            else:
                feature_list.append(np.array(dict_data[key]))
                feature_name_list.append('%s' % (key))
        return feature_list, feature_name_list

    def load_adhd_data(self):
        our_ch_list = ['Fp1', 'F7', 'T3', 'T5', 'T6', 'T4', 'F8', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'Pz', 'Fz', 'Cz', 'O1', 'P3', 'C3', 'F3']
        fl = os.listdir(self.adhd_data_path)
        for fn in fl:
            if fn.endswith('xlsm'):
                file_name = os.path.join(self.adhd_data_path, fn)
                sheet_1 = pd.read_excel(file_name, 0)
                label_list = {}
                for key in sheet_1.keys():
                    label_list[key] = []
                break

        normal_data = {'abs_power': {'Delta': [],
                                     'Theta': [],
                                     'Alpha': [],
                                     'Beta': [],
                                     'High Beta': [],
                                     'Gamma': []},
                       'rel_power': {'Delta': [],
                                     'Theta': [],
                                     'Alpha': [],
                                     'Beta': [],
                                     'High Beta': [],
                                     'Gamma': []},
                       'rat_power': {'DAR': [],
                                     'TAR': [],
                                     'TBR': []},
                       }

        feature_data_path = '%s/ADHD_ica_total' % self.adhd_data_path

        for json_file in os.listdir(feature_data_path):
            if not json_file.endswith('json'):
                continue
            file_exists = False
            for crt_idx in range(sheet_1['Hospital Number'].shape[0]):
                if sheet_1['Hospital Number'][crt_idx] == int(json_file.split('_')[0]):
                    file_exists = True
                    break

            if not file_exists:
                print('file does not exist: %s' % json_file)
                continue

            jf = open(os.path.join(feature_data_path, json_file))
            j = json.load(jf)

            if j['abs_power']['Delta'].__len__() != 19:
                print('channel num does not match: %s' % json_file)
                continue

            for key_feature in j.keys():
                if key_feature not in ['abs_power', 'rel_power', 'rat_power']:
                    continue
                for key_band in j[key_feature]:
                    normal_data[key_feature][key_band].append(np.array(j[key_feature][key_band]))

            for key in sheet_1.keys():
                label_list[key].append(sheet_1[key][crt_idx])

        feature_list, feature_name_list = self.flatten(normal_data, our_ch_list)
        self.adhd_data = np.array(feature_list).T
        self.adhd_label = label_list

    def load_hbn_data(self):
        feature_file_list = os.listdir('%s/power_feature' % self.hbn_data_path)
        c = pd.read_csv('./data/ref_file/pheno.csv')

        feature_list = []
        label_list = []

        file_name_list = np.load('./data/ref_file/obj_arr.npy')
        for file_name in file_name_list:
            subject_name = file_name.split('/')[4]

            if c.index[c['EID'] == subject_name].__len__() < 1:
                print(subject_name)
                continue

            age = c['Age'][c.index[c['EID'] == subject_name][-1]]
            sex = c['Sex'][c.index[c['EID'] == subject_name][-1]]
            ehq = c['EHQ_Total'][c.index[c['EID'] == subject_name][-1]]
            if age != age or sex != sex or ehq != ehq:
                continue
            mean_feature_val = np.zeros([15 * 19])
            feature_cnt = 0
            for feature_file_name in feature_file_list:
                if subject_name not in feature_file_name:
                    continue
                feature_file_path = os.path.join(self.hbn_data_path, 'power_feature', feature_file_name)
                feature_val_arr, feature_name_list = Util.get_feature_from_json(feature_file_path)
                if feature_val_arr is None:
                    continue
                mean_feature_val += np.reshape(np.array(feature_val_arr), [-1])
                feature_cnt += 1
            if feature_cnt < 1:
                continue
            label_list.append([age, sex, ehq, subject_name])
            feature_list.append(mean_feature_val / feature_cnt)

        self.hbn_data = np.array(feature_list)

        age_child_array = np.array(label_list)[:, 0].astype(np.float32).astype(np.int8)
        self.hbn_label = np.where(age_child_array < 18, age_child_array, 17)
        self.feature_name_list = feature_name_list

    def load_mipdb_data(self):
        feature_file_list = os.listdir('%s/mipdb_feature_power' % self.mipdb_data_path)
        c = pd.read_csv('%s/MIPDB_PublicFile2.csv' % self.mipdb_data_path)

        subject_name_list = []
        for feature_file_name in feature_file_list:
            subject_name_list.append(feature_file_name.split('_')[1][:-3])
        subject_name_list = list(dict.fromkeys(subject_name_list))

        feature_list = []
        label_list = []

        for subject_name in subject_name_list:
            # if c.index[c['EID'] == subject_name].__len__() < 1:
            #     print(subject_name)
            #     continue
            if len(c.index[c['Subject'] == subject_name]) < 1:
                continue
            age = c['Age'][c.index[c['Subject'] == subject_name][-1]]
            # sex = c['Sex'][c.index[c['EID'] == subject_name][-1]]
            # ehq = c['EHQ_Total'][c.index[c['EID'] == subject_name][-1]]
            if age != age:
                continue
            mean_feature_val = np.zeros([15 * 19])
            feature_cnt = 0
            for feature_file_name in feature_file_list:
                if subject_name not in feature_file_name:
                    continue
                feature_file_path = os.path.join(self.mipdb_data_path, 'mipdb_feature_power', feature_file_name)
                feature_val_arr, feature_name_list = Util.get_feature_from_json(feature_file_path)
                if feature_val_arr is None:
                    continue
                mean_feature_val += np.reshape(np.array(feature_val_arr), [-1])
                feature_cnt += 1
            if feature_cnt < 1:
                continue
            label_list.append([age, subject_name])
            feature_list.append(mean_feature_val / feature_cnt)

        self.mipdb_data = np.array(feature_list)
        self.mipdb_data[:, :19 * 6] *= 1E12

        age_mipdb_array = np.array(label_list)[:, 0].astype(np.float32).astype(np.int8)
        self.mipdb_age = np.where(age_mipdb_array < 18, age_mipdb_array, 17)
        self.feature_name_list = feature_name_list

    def region_combine_power(self):
        yb_ch_list = ['Fp1', 'F7', 'T3', 'T5', 'T6', 'T4', 'F8', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'Pz', 'Fz', 'Cz', 'O1', 'P3', 'C3', 'F3']
        frontal_ch_list = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4']
        left_temporal_ch_list = ['F7', 'T3', 'T5']
        right_temporal_ch_list = ['F8', 'T4', 'T6']
        parietal_ch_list = ['C3', 'C4', 'P3', 'P4', 'Pz']
        occipital_ch_list = ['O1', 'O2']
        central_ch_list = ['Cz']
        frontal_idx = []
        left_temporal_idx = []
        right_temporal_idx = []
        parietal_idx = []
        occipital_idx = []
        central_idx = []
        for ch_idx in range(len(yb_ch_list)):
            if yb_ch_list[ch_idx] in frontal_ch_list:
                frontal_idx.append(ch_idx)
            if yb_ch_list[ch_idx] in left_temporal_ch_list:
                left_temporal_idx.append(ch_idx)
            if yb_ch_list[ch_idx] in right_temporal_ch_list:
                right_temporal_idx.append(ch_idx)
            if yb_ch_list[ch_idx] in parietal_ch_list:
                parietal_idx.append(ch_idx)
            if yb_ch_list[ch_idx] in occipital_ch_list:
                occipital_idx.append(ch_idx)
            if yb_ch_list[ch_idx] in central_ch_list:
                central_idx.append(ch_idx)

        tmp_total_data = []
        tmp_feature_name_list = []

        for feature_set_idx in range(int(self.norm_total_data.shape[1] / 19)):
            idx_bias = feature_set_idx * 19

            frontal_data = np.mean(self.norm_total_data[:, np.array(frontal_idx) + idx_bias], axis=1)
            left_temporal_data = np.mean(self.norm_total_data[:, np.array(left_temporal_idx) + idx_bias], axis=1)
            right_temporal_data = np.mean(self.norm_total_data[:, np.array(right_temporal_idx) + idx_bias], axis=1)
            parietal_data = np.mean(self.norm_total_data[:, np.array(parietal_idx) + idx_bias], axis=1)
            occipital_data = np.mean(self.norm_total_data[:, np.array(occipital_idx) + idx_bias], axis=1)
            central_data = np.mean(self.norm_total_data[:, np.array(central_idx) + idx_bias], axis=1)

            tmp_total_data.append(frontal_data)
            tmp_total_data.append(left_temporal_data)
            tmp_total_data.append(right_temporal_data)
            tmp_total_data.append(parietal_data)
            tmp_total_data.append(occipital_data)
            tmp_total_data.append(central_data)
            tmp_feature_name_list.append("%s_%s" % (''.join(self.feature_name_list[idx_bias].split('_')[:-1]), "frontal"))
            tmp_feature_name_list.append("%s_%s" % (''.join(self.feature_name_list[idx_bias].split('_')[:-1]), "left_temporal"))
            tmp_feature_name_list.append("%s_%s" % (''.join(self.feature_name_list[idx_bias].split('_')[:-1]), "right_temporal"))
            tmp_feature_name_list.append("%s_%s" % (''.join(self.feature_name_list[idx_bias].split('_')[:-1]), "parietal"))
            tmp_feature_name_list.append("%s_%s" % (''.join(self.feature_name_list[idx_bias].split('_')[:-1]), "occipital"))
            tmp_feature_name_list.append("%s_%s" % (''.join(self.feature_name_list[idx_bias].split('_')[:-1]), "central"))

        self.norm_total_data = np.array(tmp_total_data).T
        self.feature_name_list = tmp_feature_name_list

    def load_data(self, hbn_outlier_remove=True, use_abs_pow=False, region_combine_power=False):

        if self.mipdb_data_path is not None:
            self.load_mipdb_data()
            # self.mipdb_age = np.where(np.array(self.mipdb_label['Age']) < 18, np.array(self.adhd_label['Age']), 17)

        if self.hbn_data_path is not None:
            self.load_hbn_data()
            if hbn_outlier_remove:
                self.outlier_removal_hbn()
            self.hbn_age = self.hbn_label

        if self.adhd_data_path is not None:
            self.load_adhd_data()
            self.adhd_age = np.where(np.array(self.adhd_label['Age']) < 18, np.array(self.adhd_label['Age']), 17)

        if self.hbn_data is not None and self.adhd_data is not None:
            self.total_data = np.concatenate([self.hbn_data, self.adhd_data, self.mipdb_data], axis=0)
            self.total_age = np.concatenate([self.hbn_age, self.adhd_age, self.mipdb_age])
            self.adhd_start_idx = self.hbn_data.shape[0]
            self.mipdb_start_idx = self.hbn_data.shape[0] + self.adhd_data.shape[0]
        elif self.hbn_data is not None:
            self.total_data = self.hbn_data
            self.total_age = self.hbn_age
        elif self.adhd_data is not None:
            self.total_data = self.adhd_data
            self.total_age = self.adhd_age
            self.adhd_start_idx = 0

        self.age_normalize_feature()

        if not use_abs_pow:
            for i in range(19):
                self.norm_total_data = np.concatenate([self.norm_total_data, np.expand_dims(np.mean(self.norm_total_data[:, np.arange(6) * 19 + i], axis=1), axis=1)], axis=1)
                self.feature_name_list.append("total_power_%s" % self.feature_name_list[i].split('_')[-1])
            # self.total_data = np.concatenate([self.total_data, ])

            self.total_data = self.total_data[:, 19 * 6:]
            self.adhd_data = self.adhd_data[:, 19 * 6:]
            self.hbn_data = self.hbn_data[:, 19 * 6:]
            self.norm_total_data = self.norm_total_data[:, 19 * 6:]
            self.feature_name_list = self.feature_name_list[19 * 6:]

        if region_combine_power:
            self.region_combine_power()
