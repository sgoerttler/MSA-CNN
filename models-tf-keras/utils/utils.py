import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import json
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.profiler import model_analyzer, option_builder


def Instantiation_optim(name, lr, weight_decay=None, lr_decay=None):
    if name == "adam":
        if weight_decay is None:
            if lr_decay is None:
                opt = keras.optimizers.Adam(learning_rate=lr)
            else:
                lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=lr,
                    decay_steps=10_000,
                    decay_rate=lr_decay)
                opt = keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            if lr_decay is None:
                opt = keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay)
            else:
                raise NotImplementedError('Weight decay and learning rate decay are not supported together.')
    elif name == "RMSprop":
        opt = keras.optimizers.RMSprop(learning_rate=lr)
    elif name == "SGD":
        opt = keras.optimizers.SGD(learning_rate=lr)
    else:
        assert False, 'Config: check optimizer, may be not implemented.'
    return opt


def Instantiation_regularizer(l1, l2):
    if   l1!=0 and l2!=0:
        regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)
    elif l1!=0 and l2==0:
        regularizer = keras.regularizers.l1(l1)
    elif l1==0 and l2!=0:
        regularizer = keras.regularizers.l2(l2)
    else:
        regularizer = None
    return regularizer


def AddContext_MultiSub(x, y, Fold_Num, context):
    '''
    input:
        x       : [N,V,F];
        y       : [N,C]; (C:num_of_classes)
        Fold_Num: [kfold];
        context : int;
        i       : int (i-th fold)
    return:
        x with contexts. [N',V,F]
    '''
    if context > 1:
        cut = context // 2
        fold = Fold_Num.copy()
        fold = np.delete(fold, -1)
        id_del = np.concatenate([np.cumsum(fold) - i for i in range(1, context)])
        id_del = np.sort(id_del)
        # id_del = np.array([], dtype=int)

        x_c = np.zeros([x.shape[0] - 2 * cut, context, x.shape[1], x.shape[2]], dtype=float)
        for j in range(cut, x.shape[0] - cut):
            x_c[j - cut] = x[j - cut:j + cut + 1]

        x_c = np.delete(x_c, id_del, axis=0)
        y_c = np.delete(y[cut: -cut], id_del, axis=0)
    elif context == 1:
        x_c = x[:, np.newaxis, :, :]
        y_c = y
    return x_c, y_c


def AddContext_SingleSub(x, y, context):
    if context > 1:
        cut = int(context / 2)
        x_c = np.zeros([x.shape[0] - 2 * cut, context, x.shape[1], x.shape[2]], dtype=float)
        for i in range(cut, x.shape[0] - cut):
            x_c[i - cut] = x[i - cut:i + cut + 1]
        y_c = y[cut:-cut]
    elif context == 1:
        x_c = x[:, np.newaxis, :, :]
        y_c = y
    return x_c, y_c


############################################################################################################
# New functions added by SMG
############################################################################################################
import os
import sys
from contextlib import contextmanager

@contextmanager
def stdout_redirected(to=os.devnull):
    """
    Context manager to suppress stdout and stderr.
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different


def get_num_flops(model, num_inputs=1):
    if num_inputs == 1:
        input_signature = [
            tf.TensorSpec(
                shape=(1, *params.shape[1:]),
                dtype=params.dtype,
                name=params.name
            ) for params in model.inputs
        ]
    elif num_inputs == 2:
        input_signature = [[
            tf.TensorSpec(
                shape=(1, *params.shape[1:]),
                dtype=params.dtype,
                name=params.name
            ) for params in model.inputs]
        ]
    else:
        raise NotImplementedError('Only 1 or 2 inputs are supported.')
    forward_graph = tf.function(model, input_signature).get_concrete_function().graph
    options = option_builder.ProfileOptionBuilder.float_operation()
    # retrieve graph info with stdout suppressed
    with stdout_redirected():
        graph_info = model_analyzer.profile(forward_graph, options=options)
    flops = graph_info.total_float_ops
    return flops


def data_generator(features, targets):
    for feature, target in zip(features, targets):
        yield feature, target


def get_num_runs(training_config, dataset_config):
    # overwrite num_runs if specified in training configuration file
    if 'num_runs' in training_config.keys():
        return training_config['num_runs']
    elif 'num_runs' in dataset_config.keys():
        return dataset_config['num_runs']
    else:
        raise ValueError('Number of runs not found in training or dataset configuration file.')

