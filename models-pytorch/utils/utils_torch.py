import torch


def get_num_model_parameters(model, verbose=0):
    """Get number of trainable and non-trainable parameters in torch model."""
    if verbose == 1:
        print('Trainable parameters:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'{name}: {param.numel()}')
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return trainable_params, non_trainable_params, total_params


def get_device():
    """Get device for torch operations if available."""
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_built():
        device = 'mps'
    else:
        device = 'cpu'
    return device
