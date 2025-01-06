from models.FeatureNet import build_FeatureNet
from models.MSTGCN import build_MSTGCN
from data_handler.cross_validation import DomainGenerator
from utils.utils import *
from utils.utils_graph import get_chebyshev_polynomials
from utils.utils_io import ReadConfig, ConfusionMatrixWriter


def evaluate_MSTGCN(model_writer, idx_run=0, config_training='MSTGCN_training', dataset='ISRUC'):
    print('Start to evaluate MSTGCN.')

    # training, model and data parameters
    train_conf = ReadConfig(config_training)
    mstgcn_conf = ReadConfig('MSTGCN')
    ds_conf = ReadConfig(f'MSTGCN_{dataset}')

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

    Dom_Generator = DomainGenerator(ds_conf['path_data'], dataset, train_conf['fold'], Fold_Num_c)

    if not os.path.exists(ds_conf['path_evaluation']):
        os.makedirs(ds_conf['path_evaluation'])

    # k-fold cross validation
    all_scores = []
    all_counts = []
    for idx_fold in range(train_conf['fold']):
        filepath_features = f'{ds_conf["path_feature"]}Feature_idx_run_{idx_run}_idx_fold_{idx_fold}.npz'
        filepath_best_mstgcn = f'{ds_conf["path_output"]}MSTGCN_Best_idx_run_{idx_run}_idx_fold_{idx_fold}.h5'

        print('Fold #', idx_fold)

        # optimizer（opt）
        opt = Instantiation_optim(train_conf['optimizer'], train_conf['learn_rate'])

        # Instantiation l1, l2 regularizer
        regularizer = Instantiation_regularizer(mstgcn_conf['l1'], mstgcn_conf['l1'])

        # get i th-fold feature and label
        Features = np.load(filepath_features, allow_pickle=True)
        val_feature = Features['val_feature']
        val_targets = Features['val_targets']
        model_writer.dict_row['training_time_features'] = Features['training_time']

        # use sliding window to add context
        print('Feature', val_feature.shape)
        val_feature, val_targets = AddContext_SingleSub(val_feature, val_targets, train_conf['context'])
        train_domain, val_domain = Dom_Generator.getFold(idx_fold)
        print('Feature with context:', val_feature.shape)

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
            opt, mstgcn_conf['GLalpha'],
            regularizer, mstgcn_conf['dropout'],
            train_conf['lambda_GRL'],
            num_classes=5,
            num_domain=train_domain.shape[1])

        # evaluate
        model.load_weights(filepath_best_mstgcn)
        with open(filepath_best_mstgcn.replace('.h5', '_metadata.json'), "r") as f:
            model_meta = json.load(f)
        model_writer.dict_row['training_time_MSTGCN'] = model_meta['training_time']
        val_mse, val_acc = model_p.evaluate(val_feature, val_targets, verbose=0)

        # predict
        predicts = model_p.predict(val_feature)
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

        # get MSTGCN model complexity and record
        if idx_fold == 0:
            trainable_params = model.trainable_weights
            trainable_count = sum([tf.keras.backend.count_params(p) for p in trainable_params])
            model_writer.dict_row['parameters_MSTGCN'] = trainable_count
            flops = get_num_flops(model)
            print('total # flops MSTGCN:', flops)
            model_writer.dict_row['flops_MSTGCN'] = flops
        model_writer.dict_row['test_samples'] = len(val_targets)
        model_writer.write_model_info(idx_fold, idx_run, write_cm.filename, ds_conf['path_evaluation'],
                                      train_conf['evaluation_file'], dataset)

        print(128 * '-')
        del model, val_feature, val_targets

    print("All folds' acc: ", all_scores)
    print("Average acc of each fold: ", np.average(all_scores, weights=all_counts))

    print('End of evaluating MSTGCN.')
    print(128 * '#')
