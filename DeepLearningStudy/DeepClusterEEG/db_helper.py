import json
import pandas as pd
import numpy as np


def feature_gathering(feature_array, feature_name_list):
    total_ch_list = []
    frontal_ch_list = []
    temporal_ch_list = []
    parietal_ch_list = []
    central_ch_list = []
    left_ch_list = []
    right_ch_list = []
    occipital_ch_list = []
    gathered_feature = []
    gathered_feature_name = []

    for feature_idx in range(19):
        feature_name = feature_name_list[feature_idx]
        total_ch_list.append(feature_idx)
        if feature_name.split('_')[-1].startswith('F'):
            frontal_ch_list.append(feature_idx)
        elif feature_name.split('_')[-1].startswith('O'):
            occipital_ch_list.append(feature_idx)
        elif feature_name.split('_')[-1].startswith('T'):
            temporal_ch_list.append(feature_idx)
        elif feature_name.split('_')[-1].startswith('P'):
            parietal_ch_list.append(feature_idx)
        elif feature_name.split('_')[-1].startswith('C'):
            central_ch_list.append(feature_idx)
        if feature_name.split('_')[-1][-1].isdigit():
            if int(feature_name.split('_')[-1][-1]) % 2 == 0:
                right_ch_list.append(feature_idx)
            else:
                left_ch_list.append(feature_idx)

    bias = 0
    for feature in ['abs_power', 'rel_power']:
        for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'High Beta', 'Gamma']:
            # gathered_feature.append(np.mean(feature_array[:, list(np.array(total_ch_list) + bias)], axis=1))
            # gathered_feature_name.append('%s_%s_%s' % (feature, band, 'total'))
            gathered_feature.append(np.mean(feature_array[:, list(np.array(frontal_ch_list) + bias)], axis=1))
            gathered_feature_name.append('%s_%s_%s' % (feature, band, 'frontal'))
            gathered_feature.append(np.mean(feature_array[:, list(np.array(left_ch_list) + bias)], axis=1))
            gathered_feature_name.append('%s_%s_%s' % (feature, band, 'left'))
            gathered_feature.append(np.mean(feature_array[:, list(np.array(right_ch_list) + bias)], axis=1))
            gathered_feature_name.append('%s_%s_%s' % (feature, band, 'right'))
            gathered_feature.append(np.mean(feature_array[:, list(np.array(temporal_ch_list) + bias)], axis=1))
            gathered_feature_name.append('%s_%s_%s' % (feature, band, 'temporal'))
            gathered_feature.append(np.mean(feature_array[:, list(np.array(parietal_ch_list) + bias)], axis=1))
            gathered_feature_name.append('%s_%s_%s' % (feature, band, 'parietal'))
            gathered_feature.append(np.mean(feature_array[:, list(np.array(central_ch_list) + bias)], axis=1))
            gathered_feature_name.append('%s_%s_%s' % (feature, band, 'central'))
            gathered_feature.append(np.mean(feature_array[:, list(np.array(occipital_ch_list) + bias)], axis=1))
            gathered_feature_name.append('%s_%s_%s' % (feature, band, 'occipital'))
            bias += 19

    for band in ['DAR', 'TAR', 'TBR']:
        # gathered_feature.append(np.mean(feature_array[:, list(np.array(total_ch_list) + bias)], axis=1))
        # gathered_feature_name.append('%s_%s_%s' % ('rat_power', band, 'total'))
        gathered_feature.append(np.mean(feature_array[:, list(np.array(frontal_ch_list) + bias)], axis=1))
        gathered_feature_name.append('%s_%s_%s' % ('rat_power', band, 'frontal'))
        gathered_feature.append(np.mean(feature_array[:, list(np.array(left_ch_list) + bias)], axis=1))
        gathered_feature_name.append('%s_%s_%s' % ('rat_power', band, 'left'))
        gathered_feature.append(np.mean(feature_array[:, list(np.array(right_ch_list) + bias)], axis=1))
        gathered_feature_name.append('%s_%s_%s' % ('rat_power', band, 'right'))
        gathered_feature.append(np.mean(feature_array[:, list(np.array(temporal_ch_list) + bias)], axis=1))
        gathered_feature_name.append('%s_%s_%s' % ('rat_power', band, 'temporal'))
        gathered_feature.append(np.mean(feature_array[:, list(np.array(parietal_ch_list) + bias)], axis=1))
        gathered_feature_name.append('%s_%s_%s' % ('rat_power', band, 'parietal'))
        gathered_feature.append(np.mean(feature_array[:, list(np.array(central_ch_list) + bias)], axis=1))
        gathered_feature_name.append('%s_%s_%s' % ('rat_power', band, 'central'))
        gathered_feature.append(np.mean(feature_array[:, list(np.array(occipital_ch_list) + bias)], axis=1))
        gathered_feature_name.append('%s_%s_%s' % ('rat_power', band, 'occipital'))
        bias += 19

    # for feature in ['alpha_peak', 'alpha_peak_power']:
    #     # gathered_feature.append(np.mean(feature_array[:, list(np.array(total_ch_list) + bias)], axis=1))
    #     # gathered_feature_name.append('%s_%s' % (feature, 'total'))
    #     gathered_feature.append(np.mean(feature_array[:, list(np.array(frontal_ch_list) + bias)], axis=1))
    #     gathered_feature_name.append('%s_%s' % (feature, 'frontal'))
    #     gathered_feature.append(np.mean(feature_array[:, list(np.array(left_ch_list) + bias)], axis=1))
    #     gathered_feature_name.append('%s_%s' % (feature, 'left'))
    #     gathered_feature.append(np.mean(feature_array[:, list(np.array(right_ch_list) + bias)], axis=1))
    #     gathered_feature_name.append('%s_%s' % (feature, 'right'))
    #     gathered_feature.append(np.mean(feature_array[:, list(np.array(temporal_ch_list) + bias)], axis=1))
    #     gathered_feature_name.append('%s_%s' % (feature, 'temporal'))
    #     gathered_feature.append(np.mean(feature_array[:, list(np.array(parietal_ch_list) + bias)], axis=1))
    #     gathered_feature_name.append('%s_%s' % (feature, 'parietal'))
    #     gathered_feature.append(np.mean(feature_array[:, list(np.array(central_ch_list) + bias)], axis=1))
    #     gathered_feature_name.append('%s_%s' % (feature, 'central'))
    #     gathered_feature.append(np.mean(feature_array[:, list(np.array(occipital_ch_list) + bias)], axis=1))
    #     gathered_feature_name.append('%s_%s' % (feature, 'occipital'))
    #     bias += 19
    #
    # for i in range(9, 0, -1):
    #     gathered_feature.append(feature_array[:, -i])
    #     gathered_feature_name.append(feature_name_list[-i])

    return gathered_feature, gathered_feature_name


def flatten(dict_data, y_ch_list):
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


def get_feature_adhd():
    our_ch_list = ['Fp1', 'F7', 'T3', 'T5', 'T6', 'T4', 'F8', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'Pz', 'Fz', 'Cz', 'O1', 'P3', 'C3', 'F3']
    import os
    fl = os.listdir("/Users/sangminlee/Documents/YBRAIN/DB/ADHD_sooncheonhyang/")
    for fn in fl:
        if fn.endswith('xlsm'):
            file_name = os.path.join("/Users/sangminlee/Documents/YBRAIN/DB/ADHD_sooncheonhyang/", fn)
            sheet_1 = pd.read_excel(file_name, 0)
            label_list = {}
            for key in sheet_1.keys():
                label_list[key] = []
            break

    normal_data = {'abs_power':
                       {'Delta': [],
                        'Theta': [],
                        'Alpha': [],
                        'Beta': [],
                        'High Beta': [],
                        'Gamma': []},
                   'rel_power':
                       {'Delta': [],
                        'Theta': [],
                        'Alpha': [],
                        'Beta': [],
                        'High Beta': [],
                        'Gamma': []},
                   'rat_power':
                       {'DAR': [],
                        'TAR': [],
                        'TBR': []},
                   }

    feature_data_path = '/Users/sangminlee/Documents/YBRAIN/DB/ADHD_sooncheonhyang/ADHD_asr_total'

    for json_file in os.listdir(feature_data_path):
        file_exists = False
        for crt_idx in range(sheet_1['Hospital Number'].shape[0]):
            if sheet_1['Hospital Number'][crt_idx] == int(json_file.split('_')[0]):
                file_exists = True
                break

        if not file_exists:
            print('file does not exist: %s'%json_file)
            continue

        jf = open(os.path.join(feature_data_path, json_file))
        j = json.load(jf)

        if j['abs_power']['Delta'].__len__() != 19:
            print('channel num does not match: %s'%json_file)
            continue

        for key_feature in j.keys():
            if key_feature not in ['abs_power', 'rel_power', 'rat_power']:
                continue
            for key_band in j[key_feature]:
                normal_data[key_feature][key_band].append(np.array(j[key_feature][key_band]))

        for key in sheet_1.keys():
            label_list[key].append(sheet_1[key][crt_idx])

    feature_list, feature_name_list = flatten(normal_data, our_ch_list)
    return feature_list, feature_name_list, label_list
