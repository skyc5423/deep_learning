import os
from data.preprocess.DataLoader import DataLoader


def preprocess_hbn_data():
    print("Preprocess HBN starts")


def preprocess_mipdb_data():
    print("Preprocess HBN starts")


def preprocess_sch_data(sch_data_list_path):
    print("Preprocess SoonCheonHyang starts")
    data_loader = DataLoader()
    data_loader.add_data_list_at(list_path=sch_data_list_path, suffix='.cnt')
    data_loader.preprocess_all('./sooncheonhyang_preprocess')


def main():
    print("Data Main starts")

    PATH_HBN_RAW_DATA = ''
    PATH_MIPDB_RAW_DATA = ''
    PATH_SCH_V1_RAW_DATA = '/home/ybrain-analysis/문서/dataset/sooncheonhyang/v1'

    PATH_HBN_PREPROCESS_DATA = '/home/ybrain-analysis/문서/dataset/hbn_data/preprocess'
    PATH_MIPDB_PREPROCESS_DATA = ''
    PATH_SCH_PREPROCESS_DATA = ''

    preprocess_hbn_data()
    preprocess_sch_data(PATH_SCH_V1_RAW_DATA)


if __name__ == '__main__':
    main()
