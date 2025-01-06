import os
import numpy as np
from scipy.stats import pearsonr

from ..cross_validation import kFoldGenerator


def compute_distance_matrix(dataset, data_folder, path_preprocessed_data, path_distance_matrix, fold=-1, num_folds=10):
    # either compute distance matrix for all folds or for one specific fold
    if fold < 0:
        fold_start = 0
        fold_stop = num_folds
    else:
        fold_start = fold
        fold_stop = fold + 1

    ReadList = np.load(path_preprocessed_data, allow_pickle=True)
    Fold_Num = ReadList['Fold_len']  # Num of samples of each fold
    Fold_Data = ReadList['Fold_data']  # Data of each fold
    Fold_Label = ReadList['Fold_label']  # Labels of each fold

    # build data generator, delete data due to memory limitation
    DataGenerator = kFoldGenerator(dataset, x=Fold_Data, y=Fold_Label, len_parts=Fold_Num, num_folds=num_folds,
                                   select_channels=False, keep_data=True)
    del Fold_Data, Fold_Label

    for idx_fold in range(fold_start, fold_stop):
        train_data, train_targets, val_data, val_targets = DataGenerator.getFold(idx_fold)
        A = np.zeros((train_data.shape[1], train_data.shape[1]))
        for idx_1 in range(train_data.shape[1]):
            for idx_2 in np.arange(idx_1 + 1, train_data.shape[1]):
                pearson_corr = pearsonr(train_data[:, idx_1, :].flatten(), train_data[:, idx_2, :].flatten())[0]
                A[idx_1, idx_2] = pearson_corr
        # compute correlation matrix
        A += A.T  # symmetrize
        A = np.abs(A)
        A[np.eye(train_data.shape[1]) == 1] = 1
        # retrieve connectivity matrix from correlation matrix, roughly modeled after MSTGCN implementation
        A = np.log(A) + 5
        A = np.maximum(A, 0)
        A /= np.mean(A) / 0.1

        print('Final connectivity matrix:')
        print(A)
        os.makedirs(os.path.dirname(path_distance_matrix), exist_ok=True)
        np.save(path_distance_matrix.replace('idx_fold', str(idx_fold)), A)


if __name__ == "__main__":
    # distance matrix for ISRUC dataset is already given by original MSTGCN implementation
    for dataset in ['sleep_edf_20']:
        data_folder = f"../data/Sleep_EDF_{dataset[-2:]}"
        path_preprocessed_data = f"../../data/Sleep_EDF_{dataset[-2:]}/sleep_edf_{dataset[-2:]}.npz"
        path_distance_matrix = f"../../data/Sleep_EDF_{dataset[-2:]}/distance_matrices/DistanceMatrix_fold_idx_fold.npy"
        compute_distance_matrix(dataset, data_folder, path_preprocessed_data, path_distance_matrix, fold=10)
