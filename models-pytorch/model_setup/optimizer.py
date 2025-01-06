import torch


def get_optimizer(config, model, stage=1):
    """Get optimizer based on model and model configuration."""
    if config['model'] == 'DeepSleepNet':
        if stage == 1:
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=1e-3)
        elif stage == 2:
            optimizer = torch.optim.Adam([{'params': model.fe.parameters(), 'lr': 1e-6, 'weight_decay': 1e-3},
                              {'params': model.temporal.parameters(), 'lr': 1e-4, 'weight_decay': 0},
                              {'params': model.ft_clf.parameters(), 'lr': 1e-4, 'weight_decay': 0}])
        else:
            raise NotImplementedError("Stage not implemented!")

        return optimizer

    if 'learning_rate_high_level' in config.keys():
        # define different learning rates for different parts of the model
        param_groups = []
        for name, param in model.named_parameters():
            module_name = name.split('.')[0]
            if module_name in ['tcm', 'fc']:
                param_groups.append({'params': param, 'lr': config['learning_rate_high_level']})
            else:
                param_groups.append({'params': param, 'lr': config['learning_rate']})
        lr = 0  # learning rate set in param_groups
    else:
        param_groups = model.parameters()
        lr = config['learning_rate']

    optimizer = torch.optim.Adam(param_groups, lr=lr,
                                 weight_decay=config.get('weight_decay', 0.0),
                                 amsgrad=config.get('amsgrad', False))

    return optimizer
