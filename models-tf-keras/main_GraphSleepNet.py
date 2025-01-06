from pathlib import Path
import argparse

from data_handler.build_data import download_prepare_dataset
from data_handler.preprocessing.preprocessing import preprocessing_graphsleepnet
from utils.utils import get_num_runs
from utils.utils_io import ModelWriter, ReadConfig
from training.train_GraphSleepNet import train_GraphSleepNet
from training.evaluate_GraphSleepNet import evaluate_GraphSleepNet


def main():
    # allow using custom configuration file when passing it as argument
    parser = argparse.ArgumentParser()
    parser.add_argument("config_training", type=str, default="cVAN_training",
                        help="Training configuration file.")
    args = parser.parse_args()

    model_writer = ModelWriter()
    model_writer.dict_row['model'] = 'GraphSleepNet'
    for dataset in ['ISRUC', 'sleep_edf_20', 'sleep_edf_78']:
        model_writer.dict_row['dataset'] = dataset
        train_conf = ReadConfig(args.config_training)
        ds_conf = ReadConfig(f'GraphSleepNet_{dataset}')

        download_prepare_dataset(dataset, ds_conf)

        ds_prep_path = Path(ds_conf['path_preprocessed_data'])
        if not ds_prep_path.is_file():
            preprocessing_graphsleepnet(dataset, ds_conf['path_prepared_data'], ds_conf['path_prepared_data'])

        for idx_run in range(get_num_runs(train_conf, ds_conf)):
            train_GraphSleepNet(idx_run, args.config_training, dataset)
            evaluate_GraphSleepNet(model_writer, idx_run, args.config_training, dataset)


if __name__ == "__main__":
    main()
