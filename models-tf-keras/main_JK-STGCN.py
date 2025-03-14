import argparse

from data_handler.build_data import download_prepare_dataset
from utils.utils import get_num_runs, set_working_directory
from utils.utils_io import ModelWriter, ReadConfig
from training.train_FeatureNet import train_FeatureNet
from training.train_JKSTGCN import train_JK_STGCN
from training.evaluate_JKSTGCN import evaluate_JK_STGCN


def main():
    set_working_directory()

    # allow using custom configuration file when passing it as argument
    parser = argparse.ArgumentParser()
    parser.add_argument("config_training", type=str, default="cVAN_training",
                        help="Training configuration file.")
    args = parser.parse_args()

    model_writer = ModelWriter()
    model_writer.dict_row['model'] = 'JK-STGCN'
    for dataset in ['ISRUC', 'sleep_edf_20', 'sleep_edf_78']:
        model_writer.dict_row['dataset'] = dataset
        train_conf = ReadConfig(args.config_training)
        ds_conf = ReadConfig(f'JK-STGCN_{dataset}')

        download_prepare_dataset(dataset, ds_conf)

        for idx_run in range(get_num_runs(train_conf, ds_conf)):
            train_FeatureNet(idx_run, 'JK-STGCN', args.config_training, dataset)
            train_JK_STGCN(idx_run, args.config_training, dataset)
            evaluate_JK_STGCN(model_writer, idx_run, args.config_training, dataset)


if __name__ == "__main__":
    main()
