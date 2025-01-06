"""
Code to prepare ISRUC dataset from raw data, based on ziyujia and XiaopengJi-USQ.
Modified and refactored by Stephan Goerttler.
"""

import argparse
import os
import numpy as np
import re
import scipy.io as sio
from scipy import signal


class ISRUCReader(object):

    def __init__(self, dataset_config):
        super().__init__()
        self.dataset_config = dataset_config
        self.data_file_list = GetFileList(self.dataset_config['path_data'], '.mat',
                                                self.dataset_config['exclude_subjects_data'])
        # leave out human scorer 2 as two labels cannot be easily combined (see ziyujia, e.g.)
        self.label_file_list = GetFileList(self.dataset_config['path_label'], '_1.txt',
                                                 self.dataset_config['exclude_subjects_label'])

        # extract patient id from file name and sort data and labels accordingly
        pat_idcs_data = np.array([int(re.findall(r'\d+', f)[-1]) for f in self.data_file_list])
        pat_idcs_label = np.array([int(re.findall(r'\d+', f)[-2]) for f in self.label_file_list])
        idcs_sort_data = np.argsort(pat_idcs_data)
        idcs_sort_label = np.argsort(pat_idcs_label)

        # ensure that the patient ids match
        assert np.all(pat_idcs_data[idcs_sort_data] == pat_idcs_label[idcs_sort_label])

        self.data_file_list = np.array(self.data_file_list)[idcs_sort_data]
        self.label_file_list = np.array(self.label_file_list)[idcs_sort_label]

    def Read1DataFile(self, file_name):
        mat_data = sio.loadmat(file_name)
        resample = 3000
        psg_use = list()
        for each_channel in self.dataset_config['channels_to_use']:
            psg_use.append(
                np.expand_dims(signal.resample(mat_data[each_channel], resample, axis=-1), 1))
        psg_use = np.concatenate(psg_use, axis=1)
        return psg_use

    def Read1LabelFile(self, file_name):
        original_label = list()
        ignore = 30
        with open(file_name, "r") as f:
            for line in f.readlines():
                if (line != '' and line != '\n'):
                    label = int(line.strip('\n'))
                    original_label.append(label)
        return np.array(original_label[:-ignore])


def GetFileList(path, filter_words=None, exclude_files=list()):
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root, file))  # Full path to file
    rs = list()

    if len(exclude_files) > 0 and filter_words:
        exclude_files = [i + filter_words for i in exclude_files]

    for each_file in all_files:
        if filter_words:
            if (filter_words in each_file) and (each_file not in exclude_files):
                rs.append(each_file)
        else:
            if len(exclude_files) > 0:
                exclude = False
                for j in exclude_files:
                    if j in each_file:
                        exclude = True
                        break
                if exclude == False:
                    rs.append(each_file)
            else:
                rs.append(each_file)
    rs.sort()
    return rs


def clean_ISRUC(dataset_config):
    isruc_process = ISRUCReader(dataset_config)

    fold_label = []
    fold_data = []
    fold_len = []

    for idx_file in range(0, len(isruc_process.data_file_list)):
        print('Read data file:', isruc_process.data_file_list[idx_file],
              ' label file:', isruc_process.label_file_list[idx_file])
        data = isruc_process.Read1DataFile(isruc_process.data_file_list[idx_file])
        label = isruc_process.Read1LabelFile(isruc_process.label_file_list[idx_file])
        print('data shape:', data.shape, ', label shape', label.shape)
        assert len(label) == len(data)
        # in ISRUC, 0-Wake, 1-N1, 2-N2, 3-N3, 5-REM
        label[label == 5] = 4  # make 4 correspond to REM
        fold_label.append(np.eye(5)[label])
        fold_data.append(data)
        fold_len.append(len(label))
    print('Preprocess over.')
    np.savez(os.path.join(isruc_process.dataset_config['path_prepared_data'], 'ISRUC.npz'),
             Fold_data=np.array(fold_data, dtype=object),
             Fold_label=np.array(fold_label, dtype=object),
             Fold_len=np.array(fold_len, dtype=object)
             )
    print('Saved to', os.path.join(isruc_process.dataset_config['path_prepared_data'], 'ISRUC.npz'))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Dataset configuration parser")

    parser.add_argument("--path_data", type=str, default="../data/ISRUC/ExtractedChannels/", help="Path to the data directory")
    parser.add_argument("--label_path", type=str, default="../data/ISRUC/RawData/", help="Path to the label directory")
    parser.add_argument("--channels_to_use", type=str, nargs='+',
                        default=["F3_A2", "C3_A2", "F4_A1", "C4_A1", "O1_A2", "O2_A1", "ROC_A1", "LOC_A2", "X1", "X2"],
                        help="List of channels to use")
    parser.add_argument("--path_prepared_data", type=str, default="../data/ISRUC/", help="Path to save preprocessed data")
    parser.add_argument("--exclude_subjects_data", type=int, nargs='*', default=[], help="Subjects to exclude from data")
    parser.add_argument("--exclude_subjects_label", type=int, nargs='*', default=[], help="Subjects to exclude from labels")

    return vars(parser.parse_args())


def main():
    dataset_config = parse_arguments()
    # move up one folder if script is called from cleaning folder
    if os.getcwd().split(os.sep)[-1] == 'cleaning':
        dataset_config['path_data'] = os.path.join('..', dataset_config['path_data'])
        dataset_config['path_label'] = os.path.join('..', dataset_config['path_label'])
        dataset_config['path_prepared_data'] = os.path.join('..', dataset_config['path_prepared_data'])

    clean_ISRUC(dataset_config)


if __name__ == '__main__':
    main()

"""
output:
    save to $path_output/ISRUC.npz:
        Fold_data:  [k-fold] list, each element is [N,V,T]
        N:subject, V:node, T:data points
        Fold_label: [k-fold] list, each element is [N,C]
        N:subject, C:label
        Fold_len:   [k-fold] list
"""







