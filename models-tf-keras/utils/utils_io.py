import pandas as pd
import os
import numpy as np
import json

from sklearn import metrics


class ModelWriter:
    def __init__(self):
        self.df_model = pd.DataFrame()
        self.dict_row = {}

    def write_model_info(self, idx_fold, idx_run, filename_cm, path, filename, dataset):
        self.dict_row['filename_cm'] = filename_cm
        self.dict_row['idx_run'] = idx_run
        self.dict_row['idx_fold'] = idx_fold
        self.dict_row['dataset'] = dataset
        self.df_model = pd.concat([self.df_model, pd.DataFrame([self.dict_row])], ignore_index=True)
        self.df_model.to_csv(os.path.join(path, filename))


class ConfusionMatrixWriter:
    def __init__(self, folder):
        self.folder = folder
        self.filename = None

    def write_confusion_matrix_fold(self, y_true, y_pred, idx_fold, idx_run):
        cm = metrics.confusion_matrix(y_true, y_pred)
        self.filename = f'ConfusionMatrix_idx_run_{idx_run}_idx_fold_{idx_fold}.npz'
        np.savez(os.path.join(self.folder, self.filename), cm)


def ReadConfig(file_name):
    config_path = 'configs' + os.sep
    config_abs_path = os.path.join(config_path,file_name)+'.json'
    with open(config_abs_path, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
    return json_data
