import numpy as np
import torch
import torch.nn as nn

from utils.utils_torch import get_device


def compute_class_imbalance_weights(config, y, train_idcs):
    """Compute class imbalance weights for weighted cross-entropy loss, partly based on emadeldeen24."""
    if config['model'] == 'AttnSleep':
        labels_count = np.sum(y, axis=0)
        total = np.sum(labels_count)
        num_classes = len(labels_count)
        class_weight = dict()
        factor = 1 / num_classes
        if 'sleep_edf_20' in config['data']:
            mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5]  # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY
        else:
            mu = [factor * 1.5, factor * 1.5, factor * 1.5, factor * 1.5, factor * 1.5]

        for key in range(num_classes):
            score = np.log(mu[key] * total / float(labels_count[key]))
            class_weight[key] = score if score > 1.0 else 1.0
            class_weight[key] = round(class_weight[key] * mu[key], 2)

        class_weight = [class_weight[i] for i in range(num_classes)]
        return class_weight
    else:
        labels_count = np.sum(y[train_idcs], axis=0)
        return np.sum(labels_count) / labels_count


def get_criterion(config, y, train_idcs=None):
    """Get criterion based on model and model configuration."""
    if 'MSA_CNN' in config['model']:
        criterion = nn.CrossEntropyLoss()
    elif config['model'] == 'AttnSleep':
        if config['criterion'] == 'weighted_CrossEntropyLoss':
            device = get_device()
            class_imbalance_weights = torch.tensor(compute_class_imbalance_weights(config, y, train_idcs),
                                                   dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_imbalance_weights)
        else:
            criterion = nn.CrossEntropyLoss()
    elif config['model'] == 'DeepSleepNet':
        criterion = nn.CrossEntropyLoss()
    elif config['model'] == 'EEGNet':
        criterion = nn.CrossEntropyLoss()

    return criterion
