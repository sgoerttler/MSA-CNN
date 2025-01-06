import gc
import time

from data_handler.cross_validation import kFoldGenerator
from utils.utils import *
from utils.utils_graph import get_chebyshev_polynomials
from utils.utils_io import ReadConfig
from models.GraphSleepNet import build_GraphSleepNet


def train_GraphSleepNet(idx_run=0, config_training='GraphSleepNet_training', dataset='ISRUC'):
    print('Start to train GraphSleepNet.')

    # training, model and data parameters
    train_conf = ReadConfig(config_training)
    graphsleepnet_conf = ReadConfig('GraphSleepNet')
    ds_conf = ReadConfig(f'GraphSleepNet_{dataset}')

    # Read data
    ReadList = np.load(ds_conf['path_preprocessed_data'], allow_pickle=True)
    Fold_Num = ReadList['Fold_len']
    Fold_Data = ReadList['Fold_data']  # Data of each fold
    Fold_Label = ReadList['Fold_label']  # Labels of each fold
    print('Read data successfully')

    Fold_Num_c = Fold_Num + 1 - train_conf['context']
    print('Number of samples: ', np.sum(Fold_Num), '(with context:', np.sum(Fold_Num_c), ')')
    print('Dataset: ', dataset)
    print(128 * '-')

    DataGenerator = kFoldGenerator(dataset, x=Fold_Data, y=Fold_Label, len_parts=Fold_Num, num_folds=train_conf['fold'],
                                   keep_data=True)

    # train GraphSleepNet
    for idx_fold in range(train_conf['fold']):
        filepath_best_graphsleepnet = f'{ds_conf["path_output"]}GraphSleepNet_Best_idx_run_{idx_run}_idx_fold_{idx_fold}.h5'
        filepath_final_graphsleepnet = f'{ds_conf["path_output"]}GraphSleepNet_Final_idx_run_{idx_run}_idx_fold_{idx_fold}.h5'

        # skip the fold if the model file already exists
        if os.path.isfile(filepath_final_graphsleepnet):
            continue

        # optimizer（opt）
        opt = Instantiation_optim(train_conf['optimizer'], train_conf['learn_rate'])

        # Instantiation l1, l2 regularizer
        regularizer = Instantiation_regularizer(graphsleepnet_conf['l1'], graphsleepnet_conf['l1'])

        # get i th-fold data and label
        train_data, train_targets, val_data, val_targets = DataGenerator.getFold(idx_fold, shuffle=(dataset == 'sleep_edf_78'))

        # use the feature to train GraphSleepNet
        print('Data', train_data.shape, val_data.shape)
        train_data, train_targets = AddContext_MultiSub(train_data, train_targets,
                                                           np.delete(Fold_Num.copy(), idx_fold), train_conf['context'])
        val_data, val_targets = AddContext_SingleSub(val_data, val_targets, train_conf['context'])
        print('Data with context:', train_data.shape, val_data.shape)

        model = build_GraphSleepNet(
            graphsleepnet_conf['cheb_k'],
            graphsleepnet_conf['cheb_filters'],
            graphsleepnet_conf['time_filters'],
            graphsleepnet_conf['time_conv_strides'],
            get_chebyshev_polynomials(ds_conf, graphsleepnet_conf, model='GraphSleepNet', num_folds=train_conf['fold']),
            graphsleepnet_conf['time_conv_kernel'],
            (val_data.shape[1:]),
            graphsleepnet_conf['num_block'],
            graphsleepnet_conf['Globaldense'],
            opt,
            graphsleepnet_conf['adj_matrix'] == 'GL',
            graphsleepnet_conf['GLalpha'],
            regularizer,
            graphsleepnet_conf['dropout'])

        print(128 * '-')
        print('Fold #', idx_fold)

        # train GraphSleepNet, monitor training accuracy to avoid data leakage (SMG)
        start_time = time.time()
        model.fit(
            x=train_data,
            y=train_targets,
            epochs=train_conf['epoch'],
            batch_size=train_conf['batch_size'],
            shuffle=True,
            validation_data=(val_data, val_targets),
            verbose=2,
            callbacks=[keras.callbacks.ModelCheckpoint(filepath_best_graphsleepnet,
                                                       monitor='acc',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto',
                                                       save_freq=1),])
        training_time = time.time() - start_time

        # save the final model
        model.save(filepath_final_graphsleepnet)
        with open(filepath_best_graphsleepnet.replace('.h5', '_metadata.json'), "w") as f:
            json.dump({'training_time': training_time}, f)
        print(128 * '-')

        del model, train_data, train_targets, val_data, val_targets
        keras.backend.clear_session()
        gc.collect()

    print('End of training GraphSleepNet.')
    print(128 * '#')
