import gc
import time

from data_handler.cross_validation import DomainGenerator
from utils.utils import *
from utils.utils_graph import get_chebyshev_polynomials
from utils.utils_io import ReadConfig
from models.MSTGCN import build_MSTGCN


def train_MSTGCN(idx_run=0, config_training='MSTGCN_training', dataset='ISRUC'):
    print('Start to train MSTGCN.')

    # training, model and data parameters
    train_conf = ReadConfig(config_training)
    mstgcn_conf = ReadConfig('MSTGCN')
    ds_conf = ReadConfig(f'MSTGCN_{dataset}')

    # Read data
    ReadList = np.load(ds_conf['path_preprocessed_data'], allow_pickle=True)
    Fold_Num = ReadList['Fold_len']
    print('Read data successfully')

    Fold_Num_c = Fold_Num + 1 - train_conf['context']
    print('Number of samples: ', np.sum(Fold_Num), '(with context:', np.sum(Fold_Num_c), ')')
    print('Dataset: ', dataset)
    print(128 * '-')

    Dom_Generator = DomainGenerator(ds_conf['path_data'], dataset, train_conf['fold'], Fold_Num_c)

    # train MSTGCN
    for idx_fold in range(train_conf['fold']):
        filepath_features = f'{ds_conf["path_feature"]}Feature_idx_run_{idx_run}_idx_fold_{idx_fold}.npz'
        filepath_best_mstgcn = f'{ds_conf["path_output"]}MSTGCN_Best_idx_run_{idx_run}_idx_fold_{idx_fold}.h5'
        filepath_final_mstgcn = f'{ds_conf["path_output"]}MSTGCN_Final_idx_run_{idx_run}_idx_fold_{idx_fold}.h5'

        # skip the fold if the model file already exists
        if os.path.isfile(filepath_final_mstgcn):
            continue

        # optimizer（opt）
        opt = Instantiation_optim(train_conf['optimizer'], train_conf['learn_rate'])

        # Instantiation l1, l2 regularizer
        regularizer = Instantiation_regularizer(mstgcn_conf['l1'], mstgcn_conf['l1'])

        # get i th-fold feature, label and distance matrix
        Features = np.load(filepath_features, allow_pickle=True)
        train_feature = Features['train_feature']
        val_feature = Features['val_feature']
        train_targets = Features['train_targets']
        val_targets = Features['val_targets']

        # use the feature to train MSTGCN
        print('Feature', train_feature.shape, val_feature.shape)
        train_feature, train_targets = AddContext_MultiSub(train_feature, train_targets,
                                                           np.delete(Fold_Num.copy(), idx_fold), train_conf['context'])
        val_feature, val_targets = AddContext_SingleSub(val_feature, val_targets, train_conf['context'])
        train_domain, val_domain = Dom_Generator.getFold(idx_fold)

        print('Feature with context:', train_feature.shape, val_feature.shape)
        model, model_p = build_MSTGCN(
            mstgcn_conf['cheb_k'],
            mstgcn_conf['cheb_filters'],
            mstgcn_conf['time_filters'],
            mstgcn_conf['time_conv_strides'],
            get_chebyshev_polynomials(ds_conf, mstgcn_conf, dataset, idx_fold, num_folds=train_conf['fold']),
            mstgcn_conf['time_conv_kernel'],
            (val_feature.shape[1:]),
            mstgcn_conf['num_block'],
            mstgcn_conf['Globaldense'],
            opt,
            mstgcn_conf['GLalpha'],
            regularizer,
            mstgcn_conf['dropout'],
            train_conf['lambda_GRL'],
            num_classes=5,
            num_domain=train_domain.shape[1])

        print(128 * '-')
        print('Fold #', idx_fold)

        # train MSTGCN, monitor training accuracy to avoid data leakage (SMG)
        start_time = time.time()
        model.fit(
            x=train_feature,
            y=[train_targets, train_domain],
            epochs=train_conf['epoch'],
            batch_size=train_conf['batch_size'],
            shuffle=True,
            validation_data=(val_feature, [val_targets, val_domain]),
            verbose=2,
            callbacks=[keras.callbacks.ModelCheckpoint(filepath_best_mstgcn,
                                                       monitor='Label_acc',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto',
                                                       save_freq=1),])
        training_time = time.time() - start_time

        # save the final model
        model.save(filepath_final_mstgcn)
        with open(filepath_best_mstgcn.replace('.h5', '_metadata.json'), "w") as f:
            json.dump({'training_time': training_time}, f)
        print(128 * '-')

        del model, train_feature, train_targets, val_feature, val_targets
        keras.backend.clear_session()
        gc.collect()

    print('End of training MSTGCN.')
    print(128 * '#')
