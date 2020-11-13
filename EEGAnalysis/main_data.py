import os
import numpy as np
import pandas as pds
from util import Util
from data.preprocess.DataLoader import DataLoader
from data.feature_extract.FeatureExtractor import FeatureExtractor
import h5py
import mne


def parse_mipdb():
    if not os.path.exists('./parsed_data_mipdb'):
        os.mkdir('./parsed_data_mipdb')

    our_ch_list = ['Fp1', 'F7', 'T3', 'T5', 'T6', 'T4', 'F8', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'Pz', 'Fz', 'Cz', 'O1', 'P3', 'C3', 'F3']
    child_db_ch_list = ['E22', 'E33', 'E45', 'E58', 'E96', 'E108', 'E122', 'E9', 'E124', 'E104', 'E92', 'E83', 'E62', 'E11', 'Cz', 'E70', 'E52', 'E36', 'E24']

    ch_types = []
    for i in range(19):
        ch_types.append('eeg')

    montage = 'standard_1020'

    MPIDB_PATH = '/home/ybrain-analysis/문서/dataset/MPIDB/preprocessed_raw'
    for mpidb_file_name in os.listdir(MPIDB_PATH):
        try:
            print(mpidb_file_name)
            break_flag = False
            hf = h5py.File(os.path.join(MPIDB_PATH, mpidb_file_name))
            fs = int(np.array(hf['EEG']['srate'])[0][0])

            info = mne.create_info(our_ch_list, fs, ch_types, montage)

            mpidb_ch_index_list = []
            for ch in range(len(our_ch_list)):
                for ch2 in range(len(hf['EEG']['chanlocs']['labels'])):
                    chanlocs_obj = hf[hf['EEG']['chanlocs']['labels'][ch2][0]]
                    chanlocs = ''.join(chr(i) for i in chanlocs_obj[:])
                    if child_db_ch_list[ch] == chanlocs:
                        mpidb_ch_index_list.append(ch2)

            closed_list = []
            for event_idx in range(hf['EEG']['event']['type'].len()):
                event_obj = hf[hf['EEG']['event']['type'][event_idx][0]]
                event = int(''.join(chr(i) for i in event_obj[:]))
                if event == 30:
                    latency_obj = hf[hf['EEG']['event']['latency'][event_idx][0]]
                    if event_idx + 1 >= hf['EEG']['event']['type'].len():
                        latency_end = hf['EEG']['data'].shape[0] - 1
                    else:
                        latency_end_obj = hf[hf['EEG']['event']['latency'][event_idx + 1][0]]
                        latency_end = int(np.array(latency_end_obj)[0][0])
                    latency = int(np.array(latency_obj)[0][0])
                    closed_list.append((latency, latency_end))

            for i in range(closed_list.__len__()):
                closed_data = np.array(hf['EEG']['data']).T[mpidb_ch_index_list, closed_list[i][0]:closed_list[i][1]]
                bad_ch_list = Util.detect_bad_channel(closed_data)
                if len(bad_ch_list) != 0:
                    neighbor_ch_list = Util.get_neighbor_ch_list(hf, bad_ch_list)
                    for ch in range(len(bad_ch_list)):
                        neighbor_ch = neighbor_ch_list[ch]
                        bad_ch_len = len(Util.detect_bad_channel(np.array(hf['EEG']['data']).T[neighbor_ch, closed_list[i][0]:closed_list[i][1]]))
                        if len(neighbor_ch) - bad_ch_len == 0:
                            break_flag = True
                            break
                        closed_data[bad_ch_list[ch]] = np.sum(np.array(hf['EEG']['data']).T[neighbor_ch, closed_list[i][0]:closed_list[i][1]], axis=0) / (len(neighbor_ch) - bad_ch_len)
                raw = mne.io.RawArray(closed_data[:, 750:-750], info)
                if raw.get_data().shape[1] < 1000 or break_flag:
                    break_flag = False
                    continue
                Util.write_edf(raw, './parsed_data_mipdb/%s_%d.edf' % (mpidb_file_name.split('.')[0], i))
        except:
            continue



def preprocess_hbn_data():
    print("Preprocess HBN starts")


def preprocess_mipdb_data(mipdb_data_list_path):
    print("Preprocess MIPDB starts")
    data_loader = DataLoader()
    data_loader.add_data_list_at(list_path=mipdb_data_list_path, suffix='.edf')
    data_loader.preprocess_all('./mipdb_preprocess')


def preprocess_sch_data(sch_data_list_path):
    print("Preprocess SoonCheonHyang starts")
    data_loader = DataLoader()
    data_loader.add_data_list_at(list_path=sch_data_list_path, suffix='.cnt')
    data_loader.preprocess_all('./sooncheonhyang_preprocess')


def feature_extract(path):
    data_loader = DataLoader()
    data_loader.add_data_list_at(list_path=path, suffix='.edf')
    feature_extractor = FeatureExtractor()
    feature_extractor.analysis_data(data_loader=data_loader,
                                    analysis_power=True,
                                    analysis_coherence=False,
                                    analysis_sef95=False,
                                    analysis_samp_entropy=False,
                                    analysis_coupling_analysis=False,
                                    output_path='./mipdb_feature_power')


def main():
    print("Data Main starts")
    # parse_mipdb()

    PATH_HBN_RAW_DATA = ''
    PATH_MIPDB_RAW_DATA = '/home/ybrain-analysis/문서/dataset/MIPDB/parsed_data_mipdb'
    PATH_SCH_V1_RAW_DATA = '/home/ybrain-analysis/문서/dataset/sooncheonhyang/v1'

    PATH_HBN_PREPROCESS_DATA = '/home/ybrain-analysis/문서/dataset/hbn_data/preprocess'
    PATH_MIPDB_PREPROCESS_DATA = ''
    PATH_SCH_PREPROCESS_DATA = ''

    # preprocess_mipdb_data(PATH_MIPDB_RAW_DATA)
    feature_extract(PATH_MIPDB_RAW_DATA)
    # preprocess_sch_data(PATH_SCH_V1_RAW_DATA)
    # feature_extract(PATH_HBN_PREPROCESS_DATA)


if __name__ == '__main__':
    main()
