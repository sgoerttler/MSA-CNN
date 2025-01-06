import keras.backend as KTF  # changed to align with newer tensorflow version
import gc
import time

from data_handler.cross_validation import kFoldGenerator
from models.FeatureNet import build_FeatureNet
from utils.utils import *
from utils.utils_io import ReadConfig


def train_FeatureNet(idx_run, model, config_training, dataset='ISRUC'):
    print('Start to train FeatureNet.')

    # Get feature net training and dataset configuration
    train_conf = ReadConfig(config_training)
    ds_conf = ReadConfig(f'{model}_{dataset}')

    # set GPU number or use CPU only
    os.environ['CUDA_VISIBLE_DEVICES'] = train_conf['GPU']
    if train_conf['GPU'] != '-1':
        config = tf.compat.v1.ConfigProto()  # changed to align with newer tensorflow version
        config.gpu_options.allow_growth=True
        sess = tf.compat.v1.Session(config=config)  # changed to align with newer tensorflow version
        KTF.set_session(sess)
        print('Use GPU #'+train_conf['GPU'])
    else:
        print('Use CPU only')

    if not os.path.exists(ds_conf['path_feature']):
        os.makedirs(ds_conf['path_feature'])

    # model training (k-fold cross validation)
    for idx_fold in range(train_conf['fold']):
        filepath_features = f'{ds_conf["path_feature"]}Feature_idx_run_{idx_run}_idx_fold_{idx_fold}.npz'
        filepath_best_featurenet = f'{ds_conf["path_feature"]}FeatureNet_Best_idx_run_{idx_run}_idx_fold_{idx_fold}.h5'

        # Skip training if the trained feature map already exists
        if os.path.isfile(filepath_features):
            continue

        # Read data in fold loop to avoid memory overflow
        ReadList = np.load(ds_conf['path_preprocessed_data'], allow_pickle=True)
        Fold_Num = ReadList['Fold_len']  # Num of samples of each fold
        Fold_Data = ReadList['Fold_data']  # Data of each fold
        Fold_Label = ReadList['Fold_label']  # Labels of each fold
        print('Read data successfully')
        print('Number of samples: ', np.sum(Fold_Num))
        print('Dataset: ', dataset)
        print(128 * '-')

        # build data generator, delete data due to memory limitation
        DataGenerator = kFoldGenerator(dataset, x=Fold_Data, y=Fold_Label, len_parts=Fold_Num, num_folds=train_conf['fold'])
        del Fold_Data, Fold_Label

        print('Fold #', idx_fold)

        # Instantiation optimizer
        opt_f = Instantiation_optim(train_conf['optimizer_f'], train_conf['learn_rate_f']) # optimizer of FeatureNet

        # get i th-fold data, shuffle to enable training on large dataset without reading all data into memory (SMG)
        train_data, train_targets, val_data, val_targets = DataGenerator.getFold(idx_fold, shuffle=(dataset == 'sleep_edf_78'))

        featureNet, featureNet_p = build_FeatureNet(opt_f, ds_conf['channels']) # '_p' model is without the softmax layer

        # train FeatureNet, monitor training accuracy to avoid data leakage (SMG)
        start_time = time.time()
        featureNet.fit(
            x=train_data,
            y=train_targets,
            epochs=train_conf['epoch_f'],
            batch_size=train_conf['batch_size_f'],
            shuffle=True,
            validation_data=(val_data, val_targets),
            verbose=2,
            callbacks=[keras.callbacks.ModelCheckpoint(filepath_best_featurenet,
                                                       monitor='acc',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto',
                                                       save_freq=1)])
        training_time = time.time() - start_time

        # load the weights of best performance
        featureNet.load_weights(filepath_best_featurenet)

        # get and save the learned feature, delete data due to memory limitation
        train_feature = featureNet_p.predict(train_data)
        del train_data
        val_feature = featureNet_p.predict(val_data)
        del val_data
        print('Retrieved training and validation features.')
        if dataset == 'sleep_edf_78':
            del featureNet
        print('Save feature of Fold #' + str(idx_fold) + ' to ' + filepath_features)
        np.savez(filepath_features,
            train_feature=train_feature,
            val_feature=val_feature,
            train_targets=train_targets,
            val_targets=val_targets,
            training_time=training_time
        )

        # Fold finish
        keras.backend.clear_session()
        del featureNet_p, train_targets, val_targets, train_feature, val_feature
        # in case of large dataset, also delete the model to free memory
        if dataset != 'sleep_edf_78':
            del featureNet
        gc.collect()

        print(128*'-')

    print('End of training FeatureNet.')
    print(128 * '#')
