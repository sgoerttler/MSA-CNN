import os
import pandas as pd
import socket
import torch
from ptflops import get_model_complexity_info
import gc

from config_manager.config_utils import get_experiment_args
from config_manager.config_loader import config_generator, skip_config_check
from data_io.loader import read_data, get_dataloaders_torch
from data_io.preprocessor import do_filtering, channel_selection
from data_io.writer import OverviewWriter, EpochWriter

from model_setup.build_model import build_model_pytorch
from model_setup.optimizer import get_optimizer
from model_setup.criterion import get_criterion
from evaluation.cross_validation import custom_cv
from evaluation.model_evaluation import Evaluator, save_model

from utils.utils import print_df
from utils.utils_torch import get_device, get_num_model_parameters


def run_model(config, overview_writer, epoch_writer, device, verbose):

    X, y, df_sample_info = read_data(config)

    if any(x in config.get('preprocessing', 'no_filter') for x in ['lowpass', 'highpass', 'bandpass']):
        X = do_filtering(X, config['preprocessing'])

    if 'channel_selection' in config.keys():
        X, config = channel_selection(X, config)

    overview_writer.run_details['total_samples'] = len(X)
    overview_writer.run_details['total_datapoints'] = (len(X) - 1) * len(X[0].flatten()) + len(X[-1].flatten())
    overview_writer.reset_folds()

    for idx_k, train_idcs, test_idcs in custom_cv(df_sample_info['idx_fold'], config['num_folds_k']):
        if config.get('fold') is not None:
            if idx_k != config['fold']:
                continue

        # load data as torch dataloaders, keep data on cpu for large multivariate Sleep-EDF-78 dataset
        is_large_multivariate = (config['data'] == 'sleep_edf_78') and (config['channel_selection'] == 'EEG/EOG/EMG')
        device_data = 'cpu' if is_large_multivariate else device
        train_loader, test_loader = get_dataloaders_torch(config, X, y, train_idcs, test_idcs, device_data)

        model = build_model_pytorch(config)

        optimizer = get_optimizer(config, model)
        criterion = get_criterion(config, y, train_idcs)

        # determine and save number of model parameters
        if idx_k == 0:
            overview_writer.run_details['trainable_params'], \
                overview_writer.run_details['non_trainable_params'], _ = get_num_model_parameters(model, verbose)
            print(f'# trainable parameters: {overview_writer.run_details["trainable_params"]}\n')
            overview_writer.run_details['MACs'], _ = get_model_complexity_info(model, tuple(X[0].shape),
                                                                               as_strings=False, backend='pytorch',
                                                                               print_per_layer_stat=False,
                                                                               verbose=verbose == 1)
            print(f'# MACs: {overview_writer.run_details["MACs"]}\n')

        # train and test model
        print(f'\n*** Training and testing fold {idx_k + 1} / {config["num_folds_k"]} ***\n')
        evaluation = Evaluator(config=config, optimizer=optimizer, criterion=criterion,
                               train_loader=train_loader, test_loader=test_loader,
                               epoch_writer=epoch_writer, idx_k=idx_k, device_data=device_data, verbose=verbose)
        model, history = evaluation.evaluate_model(model)
        overview_writer.write_results(history, idx_k, len(train_idcs), len(test_idcs))

        # optional: save model trained on the whole dataset for analysis
        if config.get('save_model', False):
            save_model(model, config, evaluation, overview_writer)

        # free memory and reset gradients
        del model, train_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()
        optimizer.zero_grad()

    overview_writer.run_details['run_complete'] = True
    overview_writer.update_results_overview(verbose=0)


def main():
    args = get_experiment_args()
    if args['gpu'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = get_device()
    hostname = socket.gethostname().replace('.local', '')
    configs = list(config_generator(args['config_file']))

    results_folder = 'results'
    model_name = args['config_file'].split('/')[-1].split('.')[0]
    results_overview_filename = f'overview_results_{model_name}_{hostname}'
    results_overview_filepath = os.path.join(results_folder, 'overview', f'{results_overview_filename}.csv')
    try:
        df_overview = pd.read_csv(results_overview_filepath, index_col=0)
    except FileNotFoundError:
        df_overview = None

    if args['verbose'] == 1:
        if len(configs) > 20:
            print('List of last 20 configurations in the experiment:')
        elif len(configs) > 1:
            print(f'List of all {len(configs)} configurations in the experiment:')
        elif len(configs) == 1:
            print('Configuration:')
        print_df(pd.DataFrame(configs), num_tail=20, repeat_header=len(configs) > 20)

    # Loop across all experiment configurations
    for idx_config, config in enumerate(configs):
        if skip_config_check(args, config, df_overview):
            continue

        print(f'\n\n********** Starting configuration {idx_config + 1} / {len(configs)} **********\n')

        overview_writer = OverviewWriter(args['config_file'], config=config, results_folder=results_folder,
                                         results_overview_filename=results_overview_filename,
                                         save_results=not args['dry_run'], first_config=(idx_config == 0))
        epoch_writer = EpochWriter(overview_writer.results_folder,
                                   overview_writer.run_details['main_results_filename'],
                                   overview_writer.run_details['sample_results_filename'],
                                   save_results=not args['dry_run'])

        # continue with next configuration if in actual execution in case of an exception
        if args['dry_run']:
            run_model(config, overview_writer, epoch_writer, device, verbose=args['verbose'])
        else:
            try:
                run_model(config, overview_writer, epoch_writer, device, verbose=args['verbose'])
            except Exception as e:
                overview_writer.update_results_overview()
                continue


if __name__ == "__main__":
    main()
