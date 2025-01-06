from models.FeatureNet import build_FeatureNet
from models.JKSTGCN import build_JKSTGCN
from utils.utils import *
from utils.utils_io import ReadConfig, ConfusionMatrixWriter


def evaluate_JK_STGCN(model_writer, idx_run=0, config_training='JK-STGCN_training', dataset='ISRUC'):
    print('Start to evaluate JKSTGCN.')

    # training, model and data parameters
    train_conf = ReadConfig(config_training)
    jkstgcn_conf = ReadConfig('JK-STGCN')
    ds_conf = ReadConfig(f'JK-STGCN_{dataset}')

    # build feature net only to measure model complexity
    if idx_run == 0:
        featureNet, featureNet_p = build_FeatureNet(channels=ds_conf['channels'])  # '_p' model is without the softmax layer
        trainable_params = featureNet_p.trainable_weights
        trainable_count = sum([tf.keras.backend.count_params(p) for p in trainable_params])
        model_writer.dict_row['parameters_FeatureNet'] = trainable_count
        flops = get_num_flops(featureNet_p)
        print('total # flops FeatureNet:', flops)
        model_writer.dict_row['flops_FeatureNet'] = flops

    # read data
    ReadList = np.load(ds_conf['path_preprocessed_data'], allow_pickle=True)
    Fold_Num = ReadList['Fold_len']
    print("Read data successfully")
    Fold_Num_c = Fold_Num + 1 - train_conf['context']
    print('Number of samples: ', np.sum(Fold_Num), '(with context:', np.sum(Fold_Num_c), ')')
    print('Dataset: ', dataset)
    print(128 * '-')

    if not os.path.exists(ds_conf['path_evaluation']):
        os.makedirs(ds_conf['path_evaluation'])

    # k-fold cross validation
    all_scores = []
    all_counts = []
    for idx_fold in range(train_conf['fold']):
        filepath_features = f'{ds_conf["path_feature"]}Feature_idx_run_{idx_run}_idx_fold_{idx_fold}.npz'
        filepath_best_jkstgcn = f'{ds_conf["path_output"]}JK_STGCN_Best_idx_run_{idx_run}_idx_fold_{idx_fold}.h5'

        print('Fold #', idx_fold)

        # get i th-fold feature and label
        Features = np.load(filepath_features, allow_pickle=True)
        val_feature = Features['val_feature']
        val_targets = Features['val_targets']
        model_writer.dict_row['training_time_features'] = Features['training_time']

        # use sliding window to add context
        print('Feature', val_feature.shape)
        val_feature, val_targets = AddContext_SingleSub(val_feature, val_targets, train_conf['context'])
        print('Feature with context:', val_feature.shape)

        model = build_JKSTGCN(
            jkstgcn_conf['cheb_k'],
            jkstgcn_conf['cheb_filters'],
            jkstgcn_conf['time_filters'],
            jkstgcn_conf['time_conv_strides'],
            jkstgcn_conf['time_conv_kernel'],
            (val_feature.shape[1:]),
            jkstgcn_conf['Globaldense'])

        # evaluate
        model.load_weights(filepath_best_jkstgcn)
        with open(filepath_best_jkstgcn.replace('.h5', '_metadata.json'), "r") as f:
            model_meta = json.load(f)
        model_writer.dict_row['training_time_JK_STGCN'] = model_meta['training_time']
        val_mse, val_acc = model.evaluate(val_feature, val_targets, verbose=0)

        # predict
        predicts = model.predict(val_feature)
        print('Evaluate', val_acc)
        all_scores.append(val_acc)
        all_counts.append(val_targets.shape[0])
        AllPred_temp = np.argmax(predicts, axis=1)
        AllTrue_temp = np.argmax(val_targets, axis=1)

        # save confusion matrix to store all relevant information
        if idx_fold == 0:
            AllPred = AllPred_temp
            AllTrue = AllTrue_temp
            write_cm = ConfusionMatrixWriter(ds_conf['path_evaluation'])
        else:
            AllPred = np.concatenate((AllPred, AllPred_temp))
            AllTrue = np.concatenate((AllTrue, AllTrue_temp))
        write_cm.write_confusion_matrix_fold(AllTrue_temp, AllPred_temp, idx_fold=idx_fold, idx_run=idx_run)

        # get JK-STGCN model complexity and record
        if idx_fold == 0:
            trainable_params = model.trainable_weights
            trainable_count = sum([tf.keras.backend.count_params(p) for p in trainable_params])
            model_writer.dict_row['parameters_JK-STGCN'] = trainable_count
            flops = get_num_flops(model)
            print('total # flops JK-STGCN:', flops)
            model_writer.dict_row['flops_JK-STGCN'] = flops
        model_writer.dict_row['test_samples'] = len(val_targets)
        model_writer.write_model_info(idx_fold, idx_run, write_cm.filename, ds_conf['path_evaluation'],
                                      train_conf['evaluation_file'], dataset)

        print(128 * '-')
        del model, val_feature, val_targets

    print("All folds' acc: ", all_scores)
    print("Average acc of each fold: ", np.average(all_scores, weights=all_counts))

    print('End of evaluating JKSTGCN.')
    print(128 * '#')
