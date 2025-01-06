import torch
import torch.nn as nn

from models.MSA_CNN import MSA_CNN
from models.DeepSleepNet import DeepFeatureNet
from models.EEGNet import EEGNet
from models.AttnSleep import AttnSleep
from utils.utils_torch import get_device


def weights_init_normal(m):
    """Initialize weights of model layers with normal distribution. Function from emadeldeen24."""
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def build_model_pytorch(config, device=None):
    """Build model based on configuration, move to device and initialize, if applicable."""
    if 'MSA_CNN' in config['model']:
        model = MSA_CNN(config)
    elif config['model'] == 'AttnSleep':
        model = AttnSleep()
    elif config['model'] == 'DeepSleepNet':
        # wrap hyperparameters for compatibility with DeepFeatureNet
        class Configuration(object):
            pass
        config_wrap = Configuration()
        config_wrap.input_channels = 1
        config_wrap.num_classes = 5
        if config['data'] == 'ISRUC_univariate':
            hparams = {"features_len": 41 * 128, "clf": 1024, "pt_clf": 5248}
        elif config['data'] == 'sleep_edf_20_univariate':
            hparams = {"features_len": 41 * 128, "clf": 1024, "pt_clf": 5248}
        elif config['data'] == 'sleep_edf_78_univariate':
            hparams = {"features_len": 41 * 128, "clf": 1024, "pt_clf": 5248}
        model = DeepFeatureNet(config_wrap, hparams=hparams, config=config)
    elif config['model'] == 'EEGNet':
        model = EEGNet(config)
    else:
        raise ValueError(f"Model {config['model']} not supported.")

    if device is None:
        device = get_device()
    model.to(device)

    # Initialize weights of model layers, used by AttnSleep
    if 'model_init' in config.keys():
        if config['model_init'] == 'normal_0.02':
            model.apply(weights_init_normal)

    return model
