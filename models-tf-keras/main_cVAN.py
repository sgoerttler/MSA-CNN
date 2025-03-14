from pathlib import Path
import argparse

from data_handler.build_data import download_prepare_dataset
from data_handler.preprocessing.preprocessing import preprocessing_cvan
from utils.utils import get_num_runs, set_working_directory
from utils.utils_io import ModelWriter, ReadConfig
from training.train_cVAN import train_cVAN
from training.evaluate_cVAN import evaluate_cVAN


def main():
    set_working_directory()

    # allow using custom configuration file when passing it as argument
    parser = argparse.ArgumentParser()
    parser.add_argument("config_training", type=str, default="cVAN_training",
                        help="Training configuration file.")
    args = parser.parse_args()

    model_writer = ModelWriter()
    model_writer.dict_row['model'] = 'cVAN'
    for dataset in ['ISRUC', 'sleep_edf_20', 'sleep_edf_78']:
        model_writer.dict_row['dataset'] = dataset
        train_conf = ReadConfig(args.config_training)
        ds_conf = ReadConfig(f'cVAN_{dataset}')

        download_prepare_dataset(dataset, ds_conf)

        ds_prep_exists = []
        for preprocessing_method, label_type in zip(['interpolation', 'stft', 'interpolation'], ['x', 'x', 'y']):
            ds_prep_exists.append(Path(ds_conf['path_preprocessed_data']
                                       .replace('preprocessing_method', preprocessing_method)
                                       .replace('label_type', label_type)).is_file())
        if not all(ds_prep_exists):
            preprocessing_cvan(dataset, ds_conf['path_prepared_data'], ds_conf['path_prepared_data'])

        for idx_run in range(get_num_runs(train_conf, ds_conf)):
            train_cVAN(idx_run, args.config_training, dataset)
            evaluate_cVAN(model_writer, idx_run, args.config_training, dataset)


if __name__ == "__main__":
    main()
