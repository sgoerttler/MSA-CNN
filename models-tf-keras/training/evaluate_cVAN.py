from models.FeatureNet import build_FeatureNet
from models.cVAN import build_cVAN
from data_handler.cross_validation import kFoldGeneratorcVAN
from utils.utils import *
from utils.utils_graph import get_chebyshev_polynomials
from utils.utils_io import ReadConfig, ConfusionMatrixWriter


def evaluate_cVAN(model_writer, idx_run=0, config_training='cVAN_training', dataset='ISRUC'):
    print('Start to evaluate cVAN.')

    # training, model and data parameters
    train_conf = ReadConfig(config_training)
    cvan_conf = ReadConfig('cVAN')
    ds_conf = ReadConfig(f'cVAN_{dataset}')

    if not os.path.exists(ds_conf['path_evaluation']):
        os.makedirs(ds_conf['path_evaluation'])

    # k-fold cross validation
    all_scores = []
    all_counts = []
    for idx_fold in range(train_conf['fold']):
        filepath_best_cvan = f'{ds_conf["path_output"]}cVAN_Best_idx_run_{idx_run}_idx_fold_{idx_fold}.h5'

        print('Fold #', idx_fold)

        # optimizer（opt）
        opt = Instantiation_optim(train_conf['optimizer'], train_conf['learn_rate'], train_conf['weight_decay'])

        # get i th-fold data and label, create data generator in k-fold cross validation loop to save memory
        DataGenerator = kFoldGeneratorcVAN(dataset, ds_conf['path_preprocessed_data'], num_folds=train_conf['fold'], keep_data=False)
        _, _, _, x_interp_test, x_stft_test, y_test = DataGenerator.getFold(idx_fold)

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

        # evaluate
        model.load_weights(filepath_best_cvan)
        with open(filepath_best_cvan.replace('.h5', '_metadata.json'), "r") as f:
            model_meta = json.load(f)
        model_writer.dict_row['training_time_cVAN'] = model_meta['training_time']
        _, _, val_acc, _ = model.evaluate([x_interp_test, x_stft_test], y_test, verbose=0)

        # predict
        predicts, _ = model.predict([x_interp_test, x_stft_test])
        print('Evaluate', val_acc)
        all_scores.append(val_acc)
        all_counts.append(y_test.shape[0])
        AllPred_temp = np.argmax(predicts, axis=1)
        AllTrue_temp = np.argmax(y_test, axis=1)

        # save confusion matrix to store all relevant information
        if idx_fold == 0:
            AllPred = AllPred_temp
            AllTrue = AllTrue_temp
            write_cm = ConfusionMatrixWriter(ds_conf['path_evaluation'])
        else:
            AllPred = np.concatenate((AllPred, AllPred_temp))
            AllTrue = np.concatenate((AllTrue, AllTrue_temp))
        write_cm.write_confusion_matrix_fold(AllTrue_temp, AllPred_temp, idx_fold=idx_fold, idx_run=idx_run)

        # get cVAN model complexity and record
        if idx_fold == 0:
            trainable_params = model.trainable_weights
            trainable_count = sum([tf.keras.backend.count_params(p) for p in trainable_params])
            model_writer.dict_row['parameters_cVAN'] = trainable_count
            flops = get_num_flops(model, num_inputs=2)
            print('total # flops cVAN:', flops)
            model_writer.dict_row['flops_cVAN'] = flops
        model_writer.dict_row['test_samples'] = len(y_test)
        model_writer.write_model_info(idx_fold, idx_run, write_cm.filename, ds_conf['path_evaluation'],
                                      train_conf['evaluation_file'], dataset)

        print(128 * '-')
        del model, x_interp_test, x_stft_test, y_test

    print("All folds' acc: ", all_scores)
    print("Average acc of each fold: ", np.average(all_scores, weights=all_counts))

    print('End of evaluating cVAN.')
    print(128 * '#')
