import os

from .download.download_dataset import download_dataset
from .cleaning.prepare_isruc import clean_ISRUC
from .cleaning.prepare_sleep_edf import clean_sleep_edf
from .preprocessing.preprocessing import preprocessing_reshape
from .utils import count_files_in_directory, count_folders_in_directory


def download_prepare_dataset(dataset, ds_conf=None, save_mode='list'):
    ds_conf_full = {}
    if 'ISRUC' in dataset:
        ds_conf_full['path_data'] = '../data/ISRUC/ExtractedChannels/'
        ds_conf_full['path_label'] = '../data/ISRUC/RawData/'
        ds_conf_full['channels_to_use'] = ['F3_A2', 'C3_A2', 'F4_A1', 'C4_A1', 'O1_A2', 'O2_A1', 'ROC_A1', 'LOC_A2', 'X1', 'X2']
        ds_conf_full['path_prepared_data'] = '../data/ISRUC/'
        ds_conf_full['exclude_subjects_data'] = []
        ds_conf_full['exclude_subjects_label'] = []
    elif 'sleep_edf' in dataset:
        ds_conf_full['path_data'] = f'../data/Sleep_EDF_{dataset[-2:]}/edf_files/'
        ds_conf_full['path_prepared_data'] = f'../data/Sleep_EDF_{dataset[-2:]}/'
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    if ds_conf is None:
        ds_conf = ds_conf_full
    else:
        # overwrite default values with input values
        ds_conf = {**ds_conf_full, **ds_conf}

    if save_mode == 'np_array':
        files_exist = [os.path.isfile(os.path.join(ds_conf['path_prepared_data'], f'{dataset}_X.npy')),
                       os.path.isfile(os.path.join(ds_conf['path_prepared_data'], f'{dataset}_y.npy')),
                       os.path.isfile(os.path.join(ds_conf['path_prepared_data'], f'{dataset}_idx_fold.npy')),
                       os.path.isfile(os.path.join(ds_conf['path_prepared_data'], f'{dataset}_idx_part.npy')),
                       os.path.isfile(os.path.join(ds_conf['path_prepared_data'], f'{dataset}_idx_epoch.npy'))]
    elif save_mode == 'list':
        files_exist = [os.path.isfile(os.path.join(ds_conf['path_prepared_data'], f'{dataset}.npz'))]

    if dataset == 'ISRUC':
        # download dataset if not all files in specified folder
        num_recordings = count_files_in_directory(ds_conf['path_data'])
        num_recording_labels = count_folders_in_directory(ds_conf['path_label'])

        if num_recordings < 10 or num_recording_labels < 10:
            print('Downloading ISRUC dataset...')
            download_dataset('ISRUC')
            print('Finished downloading ISRUC dataset.\n')

        # prepare dataset if not available yet
        if not os.path.isfile(os.path.join(ds_conf['path_prepared_data'], f'ISRUC.npz')):
            print('Cleaning ISRUC dataset...')
            clean_ISRUC(ds_conf)
            print('Finished cleaning ISRUC dataset.\n')

        # build dataset as numpy arrays, already built as list in clean_ISRUC
        if save_mode == 'np_array':
            if not all(files_exist):
                print('Building ISRUC dataset...')
                preprocessing_reshape(dataset, ds_conf['path_prepared_data'], ds_conf['path_prepared_data'], save_mode=save_mode)
                print('Finished building ISRUC dataset.\n')

    elif dataset in ['sleep_edf_20', 'sleep_edf_78']:
        dataset_type = dataset[-2:]

        # download dataset if not all files in specified folder
        num_recordings = count_files_in_directory(ds_conf['path_data']) // 2
        if num_recordings < int(dataset_type):
            print(f'Downloading Sleep-EDF-{dataset_type} dataset...')
            download_dataset(dataset)
            print(f'Finished downloading Sleep-EDF-{dataset_type} dataset.\n')

        # prepare dataset if not available yet
        num_cleaned = count_files_in_directory(os.path.join(ds_conf['path_prepared_data'], 'cleaned'))
        if num_cleaned < int(dataset_type):
            print(f'Cleaning Sleep-EDF-{dataset_type} dataset...')
            clean_sleep_edf(dataset_type=dataset_type)
            print(f'Finished cleaning Sleep-EDF-{dataset_type} dataset.\n')

        # build dataset
        if not all(files_exist):
            print(f'Building Sleep-EDF-{dataset_type} dataset...')
            preprocessing_reshape(dataset, ds_conf['path_prepared_data'], ds_conf['path_prepared_data'], save_mode=save_mode)
            print(f'Finished building Sleep-EDF-{dataset_type} dataset.\n')
