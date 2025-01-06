import os
import numpy as np


def unique_to_onehot(y_unique, num_classes=10):
    y_onehot = np.zeros((len(y_unique), num_classes))
    for i in range(len(y_unique)):
        y_onehot[i, y_unique[i]] = 1
    return y_onehot


def onehot_to_unique(y_onehot, axis=1):
    return np.argmax(y_onehot, axis=axis)


def get_fold_idx(idx_part, cv_k=10, num_parts=None, mode='sequential'):
    idx_part = np.asarray(idx_part)
    if mode == 'sequential':
        return idx_part % cv_k
    elif mode == 'clustered':
        if num_parts is None:
            raise ValueError('num_parts must be specified for "clustered" mode')
        return (idx_part // (num_parts / cv_k)).astype(int)
    else:
        raise ValueError(f'Unknown mode: {mode}')


def count_files_in_directory(directory):
    """Counts the number of files in the given directory."""
    if not os.path.isdir(directory):
        return 0
    file_list = os.listdir(directory)
    # filter hidden files in macOS or Linux
    file_list_not_hidden = [file for file in file_list if not file.startswith('.')]
    return len(file_list_not_hidden)


def count_folders_in_directory(directory):
    """Counts the number of folders in the given directory."""
    if not os.path.isdir(directory):
        return 0
    return len([entry for entry in os.listdir(directory) if os.path.isdir(os.path.join(directory, entry))])
