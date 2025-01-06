import os
import numpy as np

from .utils import unique_to_onehot, get_fold_idx


class kFoldGenerator():
    """
    Data Generator,
    rewritten by SMG to be more efficient for large datasets
    """

    def __init__(self, dataset, x=None, y=None, len_parts=None, num_folds=10, select_channels=False, keep_data=False):
        if x is not None and y is not None:
            if len(x) != len(y):
                assert False, 'Data generator: Length of x or y is not equal to k.'
            self.num_parts = len(x)
            self.x_list = x
            self.y_list = y
            self.len_parts = len_parts
            self.num_folds = num_folds
            if select_channels:
                if dataset == 'ISRUC':
                    self.channel_selection = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                elif 'sleep_edf' in dataset:
                    self.channel_selection = [0, 1, 2, 4]
            else:
                self.channel_selection = np.arange(self.x_list[0].shape[1])
            self.cv_idcs = get_fold_idx(np.arange(self.num_parts), cv_k=num_folds, num_parts=self.num_parts, mode='sequential')

        self.keep_data = keep_data

    # Get i-th fold
    def getFold(self, idx_fold, shuffle=False):
        len_train = sum(self.len_parts[self.cv_idcs != idx_fold])
        len_val = sum(self.len_parts[self.cv_idcs == idx_fold])

        train_data = np.zeros((len_train, len(self.channel_selection), self.x_list[0].shape[2]))
        train_targets = np.zeros((len_train, self.y_list[0].shape[1]))
        val_data = np.zeros((len_val, len(self.channel_selection), self.x_list[0].shape[2]))
        val_targets = np.zeros((len_val, self.y_list[0].shape[1]))

        train_start = 0
        val_start = 0

        for part in range(self.num_parts):
            len_p = self.x_list[part].shape[0]
            if self.cv_idcs[part] != idx_fold:
                train_data[train_start:train_start + len_p, :, :] = self.x_list[part][:, self.channel_selection, :]
                train_targets[train_start:train_start + len_p, :] = self.y_list[part]
                train_start += len_p
            else:
                val_data[val_start:val_start + len_p, :, :] = self.x_list[part][:, self.channel_selection, :]
                val_targets[val_start:val_start + len_p, :] = self.y_list[part]
                val_start += len_p

        if shuffle:
            idcs = np.arange(len(train_data))
            np.random.shuffle(idcs)
            train_data = train_data[idcs]
            train_targets = train_targets[idcs]

        if not self.keep_data:
            del self.x_list
            del self.y_list

        return train_data, train_targets, val_data, val_targets


class kFoldGeneratorcVAN(kFoldGenerator):
    def __init__(self, dataset, data_file, num_folds=10, select_channels=False, keep_data=False):
        super().__init__(dataset, num_folds=num_folds, select_channels=select_channels, keep_data=keep_data)
        self.data_interp = np.load(data_file.replace('preprocessing_method', 'interpolation').replace('label_type', 'X'))
        self.data_stft = np.load(data_file.replace('preprocessing_method', 'stft').replace('label_type', 'X'))
        self.data_y = np.load(data_file.replace('preprocessing_method', 'interpolation').replace('label_type', 'y'))
        self.idx_fold = np.load(data_file.replace('preprocessing_method_label_type', 'idx_fold'))

    def getFold(self, idx_fold, shuffle=False, interpolation=None):
        train_idcs = self.idx_fold != idx_fold
        test_idcs = self.idx_fold == idx_fold

        x_interp_train = np.abs(np.array(self.data_interp[train_idcs]))
        x_stft_train = self.data_stft[train_idcs]
        y_train = self.data_y[train_idcs]
        x_interp_test = np.abs(np.array(self.data_interp[test_idcs]))
        x_stft_test = self.data_stft[test_idcs]
        y_test = self.data_y[test_idcs]
        return x_interp_train, x_stft_train, y_train, x_interp_test, x_stft_test, y_test


class DomainGenerator():
    '''
    Domain Generator
    '''
    # Initializate
    def __init__(self, data_folder, dataset, num_folds, len_parts):
        num_parts = len(len_parts)
        cv_idcs = get_fold_idx(np.arange(num_parts), cv_k=num_folds, num_parts=num_parts, mode='sequential')

        # save domain information required by MSTGCN
        if (os.path.exists(os.path.join(data_folder, f'{dataset}_sample_fold_idcs.npy')) and \
                os.path.exists(os.path.join(data_folder, f'{dataset}_sample_part_idcs.npy'))):
            self.sample_fold_idcs = np.load(os.path.join(data_folder, f'{dataset}_sample_fold_idcs.npy'),
                                            allow_pickle=True)
            self.sample_part_idcs = np.load(os.path.join(data_folder, f'{dataset}_sample_part_idcs.npy'),
                                            allow_pickle=True)
        else:
            fold_idcs = []
            part_idcs = []
            for idx_part, len_part, idx_cv in zip(np.arange(len(len_parts)), len_parts, cv_idcs):
                fold_idcs.extend([idx_cv] * len_part)
                part_idcs.extend([idx_part] * len_part)
            self.sample_fold_idcs = np.array(fold_idcs)
            self.sample_part_idcs = np.array(part_idcs)
            np.save(os.path.join(data_folder, f'{dataset}_sample_fold_idcs.npy'), self.sample_fold_idcs)
            np.save(os.path.join(data_folder, f'{dataset}_sample_part_idcs.npy'), self.sample_part_idcs)

    # Get i-th fold, updated by SMG
    def getFold(self, i):
        domain_all = unique_to_onehot(self.sample_part_idcs, num_classes=max(self.sample_part_idcs)+1)
        train_domain = domain_all[self.sample_fold_idcs != i, :]
        val_domain = domain_all[self.sample_fold_idcs == i, :]

        return train_domain, val_domain
