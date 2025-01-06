import numpy as np
import os
from pathlib import Path
import pandas as pd
import datetime
from collections import OrderedDict
from scipy.stats import binomtest

from config_manager.config_utils import get_all_configs_keys


class OverviewWriter(object):

    def __init__(self, config_json, config, results_folder='results', results_overview_filename='overview_results',
                 first_config=True, save_results=True):
        self.time_start = datetime.datetime.now()
        self.time_string = self.time_start.strftime('%Y-%m-%d__%H-%M-%S')

        self.config_json = config_json
        self.config = config
        self.first_config = first_config
        self.save_results = save_results
        self.all_config_keys = get_all_configs_keys(self.config_json)

        self.results_folder = results_folder
        self.results_folder_overview = os.path.join(results_folder, 'overview')
        Path(self.results_folder_overview).mkdir(parents=True, exist_ok=True)
        self.results_overview_filename = results_overview_filename
        self.full_filepath_overview = os.path.join(self.results_folder_overview, self.results_overview_filename + '.csv')

        self.run_details = self.initialize_run_details()
        self.update_duration_run_details()
        self.columns_df_overview = self.get_columns_df_overview()
        self.df_overview = self.get_df_overview()
        self.df_overview_fill_nan_int_columns()

        self.train_loss_folds, self.train_accuracy_folds, self.train_samples_folds = [], [], []
        self.test_loss_folds, self.test_accuracy_folds, self.test_samples_folds = [], [], []

        # print('All previous configurations:\n', self.df_overview.to_string())

    def initialize_run_details(self):
        run_details = OrderedDict()
        run_details['main_results_filename'] = 'main_results__' + self.time_string
        run_details['individual_modes_filename'] = 'individual_modes__' + self.time_string
        run_details['sample_results_filename'] = 'sample_results__' + self.time_string
        run_details['trainable_params'] = -1
        run_details['non_trainable_params'] = -1
        run_details['MACs'] = -1
        run_details['train_loss'] = -1.
        run_details['train_accuracy'] = -1.
        run_details['test_loss'] = -1.
        run_details['test_accuracy'] = -1.
        run_details['test_accuracy_lower_bound'] = -1.
        run_details['test_accuracy_upper_bound'] = -1.
        run_details['test_samples'] = -1
        run_details['total_samples'] = -1
        run_details['total_datapoints'] = -1
        run_details['time_start'] = self.time_start
        run_details['duration'] = -1.
        run_details['duration_seconds'] = -1.
        run_details['folds_completed'] = 0
        run_details['run_complete'] = False
        if self.first_config:
            run_details['idx_experiment'] = 0
        return run_details

    def get_columns_df_overview(self):
        run_details_keys = list(self.run_details.keys())
        columns_df_overview = [run_details_keys[0]] + self.all_config_keys + run_details_keys[1:]
        for column_df_overview in columns_df_overview.copy():
            if 'data_folder' in column_df_overview:
                columns_df_overview.remove(column_df_overview)
        return columns_df_overview

    def get_df_overview(self):
        try:
            df_overview = pd.read_csv(self.full_filepath_overview, index_col=0)
            columns_df_overview_copy = self.columns_df_overview.copy()
            self.columns_df_overview = df_overview.columns.tolist()
            for column in columns_df_overview_copy:
                if column not in self.columns_df_overview:
                    self.columns_df_overview.append(column)
                    df_overview[column] = np.nan
            if self.first_config:
                self.run_details['idx_experiment'] = df_overview['idx_experiment'].iloc[-1] + 1
            else:
                self.run_details['idx_experiment'] = df_overview['idx_experiment'].iloc[-1]
        except FileNotFoundError:
            df_overview = pd.DataFrame(columns=self.columns_df_overview)
            if self.save_results:
                df_overview.to_csv(self.full_filepath_overview)
        except IndexError:
            df_overview = pd.DataFrame(columns=self.columns_df_overview)
            if self.save_results:
                df_overview.to_csv(self.full_filepath_overview)
        return df_overview

    def df_overview_fill_nan_int_columns(self):
        int_columns = [
            'test_samples',
            'total_samples',
            'graph_k_polynomial',
            'trainable_params',
            'non_trainable_params',
            'graph_num_true_modes'
        ]
        for key_int_column in int_columns:
            try:
                self.df_overview[key_int_column] = self.df_overview[key_int_column].fillna(-1).astype(int)
            except KeyError:
                pass

    def update_results_overview(self, verbose=1):
        config_overview = self.config.copy()
        for key in self.config.keys():
            if key not in self.columns_df_overview:
                config_overview.pop(key)
            if self.config[key] is None:
                self.config[key] = np.nan
        run_details_overview = self.run_details.copy()
        for key in self.run_details.keys():
            if key not in self.columns_df_overview:
                run_details_overview.pop(key)
        row_dict = {**config_overview, **run_details_overview}
        data_row = pd.DataFrame.from_dict(row_dict, orient='index').transpose()
        # data_row = data_row.astype({'test_samples': 'int'})
        data_row = data_row.astype({'length_time_series': 'int'})

        self.df_overview_extended = pd.concat([self.df_overview, data_row], join='outer', ignore_index=False)
        self.df_overview_extended = self.df_overview_extended.reset_index(drop=True)

        if verbose > 0:
            print('Last five configurations:\n',
                  self.df_overview_extended.iloc[max(len(self.df_overview_extended.index) - 5, 0):, :].to_string())

        if self.save_results:
            self.df_overview_extended.to_csv(self.full_filepath_overview)

    def update_duration_run_details(self):
        time_end = datetime.datetime.now()
        self.run_details['duration'] = time_end - self.time_start
        self.run_details['duration_seconds'] = (time_end - self.time_start).total_seconds()

    def reset_folds(self):
        self.train_loss_folds, self.train_accuracy_folds, self.train_samples_folds = [], [], []
        self.test_loss_folds, self.test_accuracy_folds, self.test_samples_folds = [], [], []

    def write_results(self, history, idx_k, num_train, num_test):
        self.train_loss_folds.append(history['train_loss'][-1])
        self.train_accuracy_folds.append(history['train_accuracy'][-1])
        self.train_samples_folds.append(num_train)
        self.test_loss_folds.append(history['test_loss'][-1])
        self.test_accuracy_folds.append(history['test_accuracy'][-1])
        self.test_samples_folds.append(num_test)

        self.update_duration_run_details()
        self.run_details['train_loss'] = np.average(self.train_loss_folds, weights=self.train_samples_folds)
        # data_writer.run_details['train_samples'] = sum(train_samples_folds)
        self.run_details['train_accuracy'] = np.average(self.train_accuracy_folds, weights=self.train_samples_folds)
        self.run_details['test_loss'] = np.average(self.test_loss_folds, weights=self.test_samples_folds)
        self.run_details['test_accuracy'] = np.average(self.test_accuracy_folds, weights=self.test_samples_folds)
        all_test_samples = sum(self.test_samples_folds)
        CI = binomtest(int(self.run_details['test_accuracy'] * all_test_samples),
                       n=all_test_samples).proportion_ci()
        self.run_details['test_accuracy_lower_bound'] = CI.low
        self.run_details['test_accuracy_upper_bound'] = CI.high
        self.run_details['test_samples'] = all_test_samples
        self.run_details['folds_completed'] = idx_k + 1
        self.update_results_overview()


class EpochWriter(object):

    def __init__(self, results_folder, main_results_filename, sample_results_filename, save_results=True):
        self.results_folder = os.path.join(results_folder, 'epoch_details')
        Path(self.results_folder).mkdir(parents=True, exist_ok=True)
        self.main_results_filename = main_results_filename
        self.sample_results_filename = sample_results_filename
        self.save_results = save_results

        if self.save_results:
            self.data = self.initialize_data()

    def initialize_data(self):
        data = OrderedDict()
        data['fold'] = -1
        data['epoch'] = -1
        data['train_loss'] = -1.
        data['train_accuracy'] = -1.
        data['train_samples'] = -1
        data['test_loss'] = -1.
        data['test_accuracy'] = -1.
        data['test_samples'] = -1
        return data

    def update_data(self, idx_k, epoch):
        data_row = pd.DataFrame.from_dict(self.data, orient='index').transpose()
        data_row = data_row.astype({'epoch': 'int'})
        if epoch != 0:
            data_row['fold'] = None
            data_row['train_samples'] = None
            data_row['test_samples'] = None

        create_csv = (idx_k == 0) and (epoch == 0)
        data_row.to_csv(os.path.join(self.results_folder, self.main_results_filename + '.csv'),
                        mode=['a', 'w'][int(create_csv)], header=create_csv, index=False)
