import json
from collections import OrderedDict
import argparse


def get_experiment_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='configuration file')
    parser.add_argument('--verbose', help='verbose level, 0=silent, 1=progress bar, 2=one line per epoch', default='1')
    parser.add_argument('--gpu', help='which gpu to use', default='-1')
    parser.add_argument('--rerun_configs', help='rerun configurations even if present in overview', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--dry_run', help='do not save results', action=argparse.BooleanOptionalAction, default=False)

    args = vars(parser.parse_args())

    args['gpu'] = int(args['gpu'])
    args['verbose'] = int(args['verbose'])

    return args


def get_all_configs_keys(config_file='config.json'):
    with open(config_file, 'r') as f:
        configs = json.load(f, object_pairs_hook=OrderedDict)

    all_config_keys = []
    for configs_key in configs.keys():
        if configs_key == 'data':
            all_config_keys.append('data')
            all_config_keys.append('classes')
            all_config_keys.append('channels')
        elif 'data_config_' in configs_key or 'model_config_' in configs_key:
            for config_model_key in configs[configs_key].keys():
                if config_model_key not in all_config_keys:
                    all_config_keys.append(config_model_key)
        else:
            all_config_keys.append(configs_key)
    return all_config_keys


def get_length_time_series(config):
    if 'ISRUC' in config['data']:
        target_freq = 100
        freq_sampling = target_freq
        return int(freq_sampling * float(config.get('length_time_series', '30s').replace('s', '')))
    elif 'sleep_edf_20' in config['data']:
        target_freq = 100
        freq_sampling = target_freq
        return int(freq_sampling * float(config.get('length_time_series', '30s').replace('s', '')))
    elif 'sleep_edf_78' in config['data']:
        target_freq = 100
        freq_sampling = target_freq
        return int(freq_sampling * float(config.get('length_time_series', '30s').replace('s', '')))


def get_num_channels(config):

    if 'ISRUC' in config['data']:
        if config['channel_selection'] == 'all':
            num_channels = 10
        elif config['channel_selection'] == 'EEG/EOG/EMG':
            num_channels = 9
        elif config['channel_selection'] in ['C3-A2', 'C4-A1', 'F3-A2', 'F4-A1', 'O1-A2', 'O2-A1',
                                             'LOC-A2', 'ROC-A1', 'chin EMG']:
            num_channels = 1
        else:
            raise ValueError(f'Unknown channel selection: {config["channel_selection"]}')
    elif 'sleep_edf' in config['data']:
        if config['channel_selection'] == 'all':
            num_channels = 6
        elif config['channel_selection'] == 'EEG/EOG/EMG':
            num_channels = 4
        elif config['channel_selection'] == 'EEG/EOG':
            num_channels = 3
        elif config['channel_selection'] in ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG', 'EMG']:
            num_channels = 1
        else:
            raise ValueError(f'Unknown channel selection: {config["channel_selection"]}')

    return num_channels
