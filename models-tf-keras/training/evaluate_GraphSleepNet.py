from models.GraphSleepNet import build_GraphSleepNet
from data_handler.cross_validation import kFoldGenerator
from utils.utils import *
from utils.utils_graph import get_chebyshev_polynomials
from utils.utils_io import ReadConfig, ConfusionMatrixWriter


def evaluate_GraphSleepNet(model_writer, idx_run=0, config_training='GraphSleepNet_training', dataset='ISRUC'):
    print('Start to evaluate GraphSleepNet.')

    # training, model and data parameters
    train_conf = ReadConfig(config_training)
    graphsleepnet_conf = ReadConfig('GraphSleepNet')
    ds_conf = ReadConfig(f'GraphSleepNet_{dataset}')

    # read data
    ReadList = np.load(ds_conf['path_preprocessed_data'], allow_pickle=True)
    Fold_Num = ReadList['Fold_len']
    Fold_Data = ReadList['Fold_data']  # Data of each fold
    Fold_Label = ReadList['Fold_label']  # Labels of each fold
    print("Read data successfully")
    Fold_Num_c = Fold_Num + 1 - train_conf['context']
    print('Number of samples: ', np.sum(Fold_Num), '(with context:', np.sum(Fold_Num_c), ')')
    print('Dataset: ', dataset)
    print(128 * '-')

    DataGenerator = kFoldGenerator(dataset, x=Fold_Data, y=Fold_Label, len_parts=Fold_Num, num_folds=train_conf['fold'],
                                   keep_data=True)

    if not os.path.exists(ds_conf['path_evaluation']):
        os.makedirs(ds_conf['path_evaluation'])

    # k-fold cross validation
    all_scores = []
    all_counts = []
    for idx_fold in range(train_conf['fold']):
        filepath_best_graphsleepnet = f'{ds_conf["path_output"]}GraphSleepNet_Best_idx_run_{idx_run}_idx_fold_{idx_fold}.h5'

        print('Fold #', idx_fold)

        # optimizer（opt）
        opt = Instantiation_optim(train_conf['optimizer'], train_conf['learn_rate'])

        # Instantiation l1, l2 regularizer
        regularizer = Instantiation_regularizer(graphsleepnet_conf['l1'], graphsleepnet_conf['l1'])

        # get i th-fold data and label
        _, _, val_data, val_targets = DataGenerator.getFold(idx_fold, shuffle=(dataset == 'sleep_edf_78'))

        # use the feature to train GraphSleepNet
        print('Data', val_data.shape)
        val_data, val_targets = AddContext_SingleSub(val_data, val_targets, train_conf['context'])
        print('Data with context:', val_data.shape)

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

        # evaluate
        model.load_weights(filepath_best_graphsleepnet)
        with open(filepath_best_graphsleepnet.replace('.h5', '_metadata.json'), "r") as f:
            model_meta = json.load(f)
        model_writer.dict_row['training_time_GraphSleepNet'] = model_meta['training_time']
        val_mse, val_acc = model.evaluate(val_data, val_targets, verbose=0)

        # predict
        predicts = model.predict(val_data)
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

        # get GraphSleepNet model complexity and record
        if idx_fold == 0:
            trainable_params = model.trainable_weights
            trainable_count = sum([tf.keras.backend.count_params(p) for p in trainable_params])
            model_writer.dict_row['parameters_GraphSleepNet'] = trainable_count
            flops = get_num_flops(model)
            print('total # flops GraphSleepNet:', flops)
            model_writer.dict_row['flops_GraphSleepNet'] = flops
        model_writer.dict_row['test_samples'] = len(val_targets)
        model_writer.write_model_info(idx_fold, idx_run, write_cm.filename, ds_conf['path_evaluation'],
                                      train_conf['evaluation_file'], dataset)

        print(128 * '-')
        del model, val_data, val_targets

    print("All folds' acc: ", all_scores)
    print("Average acc of each fold: ", np.average(all_scores, weights=all_counts))

    print('End of evaluating GraphSleepNet.')
    print(128 * '#')
