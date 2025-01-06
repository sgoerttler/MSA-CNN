import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_handler.build_data import download_prepare_dataset


def get_dataloaders_torch(config, X, y, train_idcs, test_idcs, device='cpu', multiprocessing_context=None,
                          sampler=None):
    """Create PyTorch DataLoader objects for training and testing."""
    train_dataset = []
    test_dataset = []

    for idx_train in train_idcs:
        train_dataset.append([[torch.tensor(X[idx_train], dtype=torch.float32).to(device)],
                              torch.tensor(y[idx_train], dtype=torch.float32).to(device)])
    for idx_test in test_idcs:
        test_dataset.append([[torch.tensor(X[idx_test], dtype=torch.float32).to(device)],
                             torch.tensor(y[idx_test], dtype=torch.float32).to(device)])

    # Create a DataLoader for shuffling and batching
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=sampler is None,
                              drop_last=config['drop_last_batch'], multiprocessing_context=multiprocessing_context, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=256, drop_last=False,
                             multiprocessing_context=multiprocessing_context)

    return train_loader, test_loader


def read_data(config, target_pos=None):
    """Read in cleaned data files."""
    if target_pos is None:
        target_pos, _, _ = get_target_pos(config)

    download_prepare_dataset(config['data'], save_mode='np_array')
    dataset = config['data'].replace('_univariate', '')
    data_folder = f'../data/{dataset}'

    X = np.load(os.path.join(data_folder, f'{dataset}_X.npy'), allow_pickle=True)
    y = np.load(os.path.join(data_folder, f'{dataset}_y.npy'), allow_pickle=True)
    df_sample_info = pd.DataFrame({
        'idx_fold': np.load(os.path.join(data_folder, f'{dataset}_idx_fold.npy'), allow_pickle=True),
        'idx_part': np.load(os.path.join(data_folder, f'{dataset}_idx_part.npy'), allow_pickle=True),
        'idx_epoch': np.load(os.path.join(data_folder, f'{dataset}_idx_epoch.npy'), allow_pickle=True)})

    if (target_pos['channels'] == 2 and target_pos['time_series'] == 1) and target_pos['new_axis'] is None:
        X = np.transpose(X, axes=np.argsort([0, target_pos['channels'], target_pos['time_series']]))
    elif target_pos['new_axis'] is not None:
        X = np.transpose(X[:, :, :, np.newaxis], axes=np.argsort(
            [0, target_pos['channels'], target_pos['time_series'], target_pos['new_axis']]))

    return X, y, df_sample_info


def get_target_pos(config):
    """Each model requires a specific input shape. This function determines the target positions of all dimensions."""
    if 'MSA_CNN' in config['model']:
        target_pos = {'channels': 2, 'time_series': 3, 'new_axis': 1}
    elif '1D_CNN' in config['model']:
        target_pos = {'channels': 3, 'time_series': 2, 'new_axis': 1}
    elif 'LSTM_net' in config['model']:
        target_pos = {'channels': 2, 'time_series': 1, 'new_axis': None}
    elif 'FeatureNet' in config['model'] or 'spectral_features' in config['model']:
        target_pos = {'channels': 1, 'time_series': 2, 'new_axis': 3}
    elif 'DeepSleepNet' in config['model']:
        target_pos = {'channels': 1, 'time_series': 2, 'new_axis': None}
    elif 'FC-STGNN' in config['model']:
        target_pos = {'channels': 2, 'time_series': 3, 'new_axis': 1}
    elif 'EEGNet' in config['model']:
        target_pos = {'channels': 2, 'time_series': 3, 'new_axis': 1}
    elif 'AttnSleep' in config['model']:
        target_pos = {'channels': 1, 'time_series': 2, 'new_axis': None}

    if target_pos['new_axis'] is not None:
        input_permutation = np.argsort([target_pos['channels'], target_pos['time_series'], target_pos['new_axis']])
        input_shape = np.array([config['num_channels'], config['length_time_series'], 1])[input_permutation]
    else:
        input_permutation = np.argsort([target_pos['channels'], target_pos['time_series']])
        input_shape = np.array([config['num_channels'], config['length_time_series']])[input_permutation]

    return target_pos, input_shape, input_permutation
