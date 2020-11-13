import os
import pyedflib
import mne
import numpy as np
from util import Util
from data.preprocess.Preprocessor import Preprocessor


class DataLoader(object):
    def __init__(self, for_preprocess=False):
        self.total_data_list = []
        self.yb_ch_list = ['Fp1', 'F7', 'T3', 'T5', 'T6', 'T4', 'F8', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'Pz', 'Fz', 'Cz', 'O1', 'P3', 'C3', 'F3']
        if for_preprocess:
            self.preprocessor = Preprocessor()

    def add_data_list_at(self, list_path, suffix='.edf'):
        for file_name in os.listdir(list_path):
            if file_name.endswith(suffix):
                self.total_data_list.append(os.path.join(list_path, file_name))

    def match_channel_cnt(self, raw_cnt):
        raw_data = raw_cnt.get_data()
        ch_matched_data = []

        ch_types = []
        montage = 'standard_1020'

        for ch_idx in range(len(self.yb_ch_list)):
            ch = self.yb_ch_list[ch_idx]
            if ch == 'T3':
                raw_ch_idx = raw_cnt.ch_names.index('T7')
            elif ch == 'T4':
                raw_ch_idx = raw_cnt.ch_names.index('T8')
            elif ch == 'T5':
                raw_ch_idx = raw_cnt.ch_names.index('P7')
            elif ch == 'T6':
                raw_ch_idx = raw_cnt.ch_names.index('P8')
            else:
                raw_ch_idx = raw_cnt.ch_names.index(ch.upper())
            ch_matched_data.append(raw_data[raw_ch_idx])
            ch_types.append('eeg')
        ch_matched_data = np.array(ch_matched_data)

        info = mne.create_info(self.yb_ch_list, raw_cnt.info['sfreq'], ch_types, montage)
        raw = mne.io.RawArray(np.array(ch_matched_data) * 1E6, info)
        return raw

    def read_edf_file(self, edf_file_path):
        raw = mne.io.read_raw_edf(edf_file_path)
        # if raw.get_data().shape[0] != 19:
        #     try:
        #         raw = self.match_channel_cnt(raw)
        #     except:
        #         return None
        return raw

    def read_cnt_file(self, cnt_file_path):
        raw = mne.io.read_raw_cnt(cnt_file_path)
        try:
            raw = self.match_channel_cnt(raw)
        except:
            return None
        return raw

    def preprocess_all(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        preprocessor = Preprocessor()
        for crt_data in self.total_data_list:
            if crt_data.endswith('.cnt'):
                raw = self.read_cnt_file(crt_data)
            else:
                raw = self.read_edf_file(crt_data)

            if raw is None:
                continue
            raw = preprocessor.preprocess_filtering(raw)
            raw = preprocessor.artifact_remove_asr(raw)
            raw = preprocessor.artifact_remove_ica(raw)
            file_name = crt_data.split('/')[-1].split('.')[0]
            Util.write_edf(raw, os.path.join(output_path, '%s.edf' % file_name))
