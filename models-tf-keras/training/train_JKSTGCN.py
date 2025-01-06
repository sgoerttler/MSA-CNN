import gc
import time

from utils.utils import *
from utils.utils_io import ReadConfig
from models.JKSTGCN import build_JKSTGCN


def train_JK_STGCN(idx_run=0, config_training='K-STGCN_training', dataset='ISRUC'):
    print('Start to train JK-STGCN.')

    # training, model and data parameters
    train_conf = ReadConfig(config_training)
    jkstgcn_conf = ReadConfig('JK-STGCN')
    ds_conf = ReadConfig(f'JK-STGCN_{dataset}')

    # Read data
    ReadList = np.load(ds_conf['path_preprocessed_data'], allow_pickle=True)
    Fold_Num = ReadList['Fold_len']
    print('Read data successfully')

    Fold_Num_c = Fold_Num + 1 - train_conf['context']
    print('Number of samples: ', np.sum(Fold_Num), '(with context:', np.sum(Fold_Num_c), ')')
    print('Dataset: ', dataset)
    print(128 * '-')

    # train JK-STGCN
    for idx_fold in range(train_conf['fold']):
        filepath_features = f'{ds_conf["path_feature"]}Feature_idx_run_{idx_run}_idx_fold_{idx_fold}.npz'
        filepath_best_jkstgcn = f'{ds_conf["path_output"]}JK_STGCN_Best_idx_run_{idx_run}_idx_fold_{idx_fold}.h5'
        filepath_final_jkstgcn = f'{ds_conf["path_output"]}JK_STGCN_Final_idx_run_{idx_run}_idx_fold_{idx_fold}.h5'

        # skip the fold if the model file already exists
        if os.path.isfile(filepath_final_jkstgcn):
            continue

        # optimizer（opt)
        opt = Instantiation_optim('adam', train_conf['learn_rate'], lr_decay=train_conf['lr_decay'])

        # set l1, l2（regularizer）
        if jkstgcn_conf['l1'] != 0 and jkstgcn_conf['l2'] != 0:
            regularizer = keras.regularizers.l1_l2(l1=jkstgcn_conf['l1'], l2=jkstgcn_conf['l2'])
        elif jkstgcn_conf['l1'] != 0 and jkstgcn_conf['l2'] == 0:
            regularizer = keras.regularizers.l1(jkstgcn_conf['l1'])
        elif jkstgcn_conf['l1'] == 0 and jkstgcn_conf['l2'] != 0:
            regularizer = keras.regularizers.l2(jkstgcn_conf['l2'])
        else:
            regularizer = None

        # get i th-fold feature and label
        Features = np.load(filepath_features, allow_pickle=True)
        train_feature = Features['train_feature']
        val_feature = Features['val_feature']
        train_targets = Features['train_targets']
        val_targets = Features['val_targets']

        # use the feature to train JKSTGCN
        print('Feature', train_feature.shape, val_feature.shape)
        train_feature, train_targets = AddContext_MultiSub(train_feature, train_targets,
                                                           np.delete(Fold_Num.copy(), idx_fold), train_conf['context'])
        val_feature, val_targets = AddContext_SingleSub(val_feature, val_targets, train_conf['context'])

        print('Feature with context:', train_feature.shape, val_feature.shape)
        model = build_JKSTGCN(
            jkstgcn_conf['cheb_k'], 
            jkstgcn_conf['cheb_filters'], 
            jkstgcn_conf['time_filters'], 
            jkstgcn_conf['time_conv_strides'], 
            jkstgcn_conf['time_conv_kernel'],
            (val_feature.shape[1:]),
            jkstgcn_conf['Globaldense'],
            opt,
            regularizer,
            jkstgcn_conf['dropout'],
            dataset,
            idx_run,
            idx_fold)

        print(128 * '-')
        print('Fold #', idx_fold)

        # train JK-STGCN, monitor training accuracy to avoid data leakage (SMG)
        start_time = time.time()
        model.fit(
            x=train_feature,
            y=train_targets,
            epochs=train_conf['epoch'],
            batch_size=train_conf['batch_size'],
            shuffle=True,
            validation_data=(val_feature, val_targets),
            verbose=2,
            callbacks=[keras.callbacks.ModelCheckpoint(filepath_best_jkstgcn,
                                                       monitor='acc',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto',
                                                       save_freq=1),])
        training_time = time.time() - start_time

        # save the final model
        model.save(filepath_final_jkstgcn)
        with open(filepath_best_jkstgcn.replace('.h5', '_metadata.json'), "w") as f:
            json.dump({'training_time': training_time}, f)
        print(128 * '-')

        del model, train_feature, train_targets, val_feature, val_targets
        keras.backend.clear_session()
        gc.collect()

    print('End of training JKSTGCN.')
    print(128 * '#')
