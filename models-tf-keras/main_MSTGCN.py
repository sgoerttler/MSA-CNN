import os
import argparse

from data_handler.build_data import download_prepare_dataset
from data_handler.preprocessing.compute_distance_matrix import compute_distance_matrix
from utils.utils import get_num_runs
from utils.utils_io import ModelWriter, ReadConfig
from training.train_FeatureNet import train_FeatureNet
from training.train_MSTGCN import train_MSTGCN
from training.evaluate_MSTGCN import evaluate_MSTGCN


def main():
    # allow using custom configuration file when passing it as argument
    parser = argparse.ArgumentParser()
    parser.add_argument("config_training", type=str, default="cVAN_training",
                        help="Training configuration file.")
    args = parser.parse_args()

    model_writer = ModelWriter()
    model_writer.dict_row['model'] = 'MSTGCN'
    for dataset in ['ISRUC', 'sleep_edf_20', 'sleep_edf_78']:
        model_writer.dict_row['dataset'] = dataset
        train_conf = ReadConfig(args.config_training)
        ds_conf = ReadConfig(f'MSTGCN_{dataset}')

        download_prepare_dataset(dataset, ds_conf)

        # ensure that distance matrix is available
        if dataset == 'ISRUC':
            # distance matrix is required for ISRUC dataset, not based on raw data
            if not os.path.isfile(ds_conf['path_distance_matrix']):
                raise FileNotFoundError(f"Distance matrix not found for dataset {dataset}.")
        elif 'sleep_edf' in dataset:
            # compute distance matrices if not already done
            file_exists = []
            for idx_fold in range(train_conf['fold']):
                file_exists.append(os.path.isfile(ds_conf['path_distance_matrix'].replace('idx_fold', str(idx_fold))))
            if sum(file_exists) < train_conf['fold']:
                compute_distance_matrix(dataset, ds_conf['path_data'], ds_conf['path_preprocessed_data'],
                                        ds_conf['path_distance_matrix'], num_folds=train_conf['fold'])

        for idx_run in range(get_num_runs(train_conf, ds_conf)):
            train_FeatureNet(idx_run, 'MSTGCN', args.config_training, dataset)
            train_MSTGCN(idx_run, args.config_training, dataset)
            evaluate_MSTGCN(model_writer, idx_run, args.config_training, dataset)


if __name__ == "__main__":
    main()
