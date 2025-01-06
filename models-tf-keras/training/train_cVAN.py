import gc
import time

from data_handler.cross_validation import kFoldGeneratorcVAN
from utils.utils import *
from utils.utils_io import ReadConfig
from models.cVAN import build_cVAN


def train_cVAN(idx_run=0, config_training='cVAN_training', dataset='ISRUC'):
    print('Start to train cVAN.')

    # training, model and data parameters
    train_conf = ReadConfig(config_training)
    cvan_conf = ReadConfig('cVAN')
    ds_conf = ReadConfig(f'cVAN_{dataset}')

    # train cVAN
    for idx_fold in range(train_conf['fold']):
        filepath_best_cvan = f'{ds_conf["path_output"]}cVAN_Best_idx_run_{idx_run}_idx_fold_{idx_fold}.h5'
        filepath_final_cvan = f'{ds_conf["path_output"]}cVAN_Final_idx_run_{idx_run}_idx_fold_{idx_fold}.h5'

        # skip the fold if the model file already exists
        if os.path.isfile(filepath_final_cvan):
            continue

        # optimizer（opt）
        opt = Instantiation_optim(train_conf['optimizer'], train_conf['learn_rate'], train_conf['weight_decay'])

        # get i th-fold data and label, create data generator in k-fold cross validation loop to save memory
        DataGenerator = kFoldGeneratorcVAN(dataset, ds_conf['path_preprocessed_data'], num_folds=train_conf['fold'], keep_data=False)
        x_interp_train, x_stft_train, y_train, x_interp_test, x_stft_test, y_test = DataGenerator.getFold(idx_fold)

        model = build_cVAN(
            cvan_conf['vocab_size'],
            cvan_conf['maxlen'],
            cvan_conf['num_class'],
            cvan_conf['d_model'],
            cvan_conf['num_heads'],
            cvan_conf['num_layers'],
            cvan_conf['ff_dim'],
            ds_conf['channels'],
            ds_conf['n_length'],
            optimizer=opt,
            weight_logit=train_conf['loss_weight_logit'],
            weight_sim=train_conf['loss_weight_sim'])

        print(128 * '-')
        print('Fold #', idx_fold)

        # train cVAN, monitor training accuracy to avoid data leakage (SMG)
        start_time = time.time()
        model.fit(
            x=[x_interp_train, x_stft_train],
            y=y_train,
            epochs=train_conf['epoch'],
            batch_size=train_conf['batch_size'],
            shuffle=True,
            validation_data=([x_interp_test, x_stft_test], y_test),
            verbose=2,
            callbacks=[keras.callbacks.ModelCheckpoint(filepath_best_cvan,
                                                       monitor='logit_output_accuracy',
                                                       verbose=0,
                                                       save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto',
                                                       save_freq=1),])
        training_time = time.time() - start_time

        # save the final model
        model.save(filepath_final_cvan)
        with open(filepath_best_cvan.replace('.h5', '_metadata.json'), "w") as f:
            json.dump({'training_time': training_time}, f)
        print(128 * '-')

        del model, x_interp_train, x_stft_train, y_train, x_interp_test, x_stft_test, y_test
        keras.backend.clear_session()
        gc.collect()

    print('End of training cVAN.')
    print(128 * '#')
