from functools import reduce
import json
from itertools import product
from collections import OrderedDict
from utils.utils import unique
from config_manager.config_utils import get_length_time_series, get_num_channels


def config_generator(config_file='config.json'):
    """Generate all valid configurations from a configuration file."""
    with open(config_file, 'r') as f:
        configs = json.load(f)

    # collect all (unnested) config variables that will be looped over
    configs_loop_variables = [key for key in configs.keys() if isinstance(configs[key], list)]

    # determine which parameters will be looped over in the configuration file,
    # move optional parameter configurations to the front as they will be changed last
    try:
        configs_loop_variables.insert(0, configs_loop_variables.pop(configs_loop_variables.index('idx_run')))
        configs_loop_variables.insert(1, configs_loop_variables.pop(configs_loop_variables.index('data')))
    except ValueError:
        pass
    try:
        configs_loop_variables.insert(len(configs_loop_variables), configs_loop_variables.pop(configs_loop_variables.index('conv1_mode')))
    except ValueError:
        pass
    try:
        configs_loop_variables.insert(0, configs_loop_variables.pop(configs_loop_variables.index('batch_size')))
    except ValueError:
        pass

    configs_variables_product = list(product(*[enumerate(configs[key]) for key in configs_loop_variables]))
    for config_variables_product_items in configs_variables_product:
        # skip off-diagonal combinations if model and data are paired element-wise
        if configs['model_data_pairing'] == 'element_wise':
            if (config_variables_product_items[configs_loop_variables.index('model')][0] !=
                    config_variables_product_items[configs_loop_variables.index('data')][0]):
                continue

        # all config variables excluding nested variables, such as data config and specific model config
        configs_keys_all = [key for key in configs.keys() if not ('data_config' in key or 'model_config' in key)]

        # add nested data variables to config keys
        try:
            idx_data = configs_loop_variables.index('data')
            data_config_key = '_'.join(['data_config', config_variables_product_items[idx_data][1]])
            configs_nested_data_all_variables = [key for key in configs[data_config_key].keys()]
            configs_nested_data_loop_variables = \
                [key for key in configs[data_config_key].keys() if isinstance(configs[data_config_key][key], list)]
            configs_nested_data_variables_product = list(product(*[
                enumerate(configs[data_config_key][key]) for key in configs_nested_data_loop_variables]))
            configs_keys_all += list(configs[data_config_key].keys())

        except KeyError:
            configs_nested_data_loop_variables = []
            configs_nested_data_all_variables = []
            configs_nested_data_variables_product = [()]

        # add nested model variables to config keys
        try:
            idx_model = configs_loop_variables.index('model')
            model_config_key = '_'.join(['model_config', config_variables_product_items[idx_model][1]])
            configs_nested_model_all_variables = [key for key in configs[model_config_key].keys()]
            configs_nested_model_loop_variables = \
                [key for key in configs[model_config_key].keys() if isinstance(configs[model_config_key][key], list)]
            configs_nested_model_variables_product = list(product(*[
                enumerate(configs[model_config_key][key]) for key in configs_nested_model_loop_variables]))
            configs_keys_all += list(configs[model_config_key].keys())
        except KeyError:
            configs_nested_model_all_variables = []
            configs_nested_model_loop_variables = []
            configs_nested_model_variables_product = [()]

        # remove duplicate entries while preserving the order
        configs_keys_all = unique(configs_keys_all)

        # loop over all possible combinations of the nested variables
        for config_nested_data_variables_product_items in configs_nested_data_variables_product:
            for config_nested_model_variables_product_items in configs_nested_model_variables_product:

                config = OrderedDict()
                valid_config = True

                # add all unnested and nested variables to config
                for configs_key in configs_keys_all:
                    if configs_key in configs_nested_data_loop_variables:
                        idx = configs_nested_data_loop_variables.index(configs_key)
                        config[configs_key] = config_nested_data_variables_product_items[idx][1]
                    elif configs_key in configs_nested_data_all_variables:
                        data_config_key = '_'.join(['data_config', config_variables_product_items[idx_data][1]])
                        config[configs_key] = configs[data_config_key][configs_key]
                    elif configs_key in configs_nested_model_loop_variables:
                        idx = configs_nested_model_loop_variables.index(configs_key)
                        config[configs_key] = config_nested_model_variables_product_items[idx][1]
                    elif configs_key in configs_nested_model_all_variables:
                        model_config_key = '_'.join(['model_config', config_variables_product_items[idx_model][1]])
                        config[configs_key] = configs[model_config_key][configs_key]
                    elif configs_key in configs_loop_variables:
                        idx = configs_loop_variables.index(configs_key)
                        config[configs_key] = config_variables_product_items[idx][1]
                    else:
                        config[configs_key] = configs[configs_key]

                # add length of time series in samples to config
                config['length_time_series'] = get_length_time_series(config)

                # add number of channels to config
                config['num_channels'] = get_num_channels(config)

                # in multi-scale ablation study, vary filter scale start and filter scale end
                if 'filter_scales_start' in config.keys() and 'filter_scales_end' in config.keys():
                    if config['filter_scales_start'] > config['filter_scales_end']:
                        continue  # skip config if filter range is not valid
                    if config['filter_scales_start'] == 1 and config['filter_scales_end'] == 4:
                        continue  # skip default configuration run in main experiment

                if valid_config:
                    yield config


def skip_config_check(args, config, df_overview):
    """Check if configuration has been run before and if it should be run."""

    if args['rerun_configs'] or df_overview is None:
        return False  # don't skip

    # determine all previous successful runs with masking
    masks = []
    for key in config.keys():
        print(f'\t{key}: {config[key]}')
        if config[key] is None:
            if key in ['graph_num_true_modes', 'graph_k_polynomial']:
                masks.append(df_overview[key] == -1)
            else:
                masks.append(df_overview[key].isna())
        else:
            try:
                masks.append(df_overview[key] == config[key])
            except KeyError:
                masks.append(config[key] != config[key])  # new configuration that has not been run before
    masks.append(df_overview['run_complete'])
    masks.append(df_overview['train_loss'] == df_overview['train_loss'])  # remove nan
    masks.append(df_overview['test_loss'] == df_overview['test_loss'])  # remove nan
    mask_config_selection = reduce(lambda mask_1, mask_2: mask_1 & mask_2, masks)

    if df_overview[mask_config_selection].empty:
        return False  # don't skip
    else:
        return True  # skip
