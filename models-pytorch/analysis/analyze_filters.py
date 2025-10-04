import numpy as np
import os
from pathlib import Path
import argparse
import pandas as pd
import torch
from scipy.signal import welch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config_manager.config_utils import get_num_channels
from data_handler.utils import onehot_to_unique
from data_io.loader import read_data
from data_io.preprocessor import channel_selection, do_filtering
from models.MSA_CNN import MSA_CNN
from models.EEGNet import EEGNet
from utils.utils import add_legend, transparent_to_opaque


class DataProcessingFilterSpectra(object):
    """This class is used to process the data for the filter spectra analysis based on the trained models."""

    def __init__(self, config, state_dict):
        self.config = config
        self.state_dict = state_dict

        # load trained MSA_CNN model
        self.config['num_channels'] = get_num_channels(self.config)
        self.config['return_conv1'] = True

        if 'MSA_CNN' in self.config['model']:
            self.model = MSA_CNN(self.config)
            for idx_scale in range(self.config['num_filter_scales']):
                self.state_dict[f'msm.multi_scale_convolution.convs1.{idx_scale}.bias'] *= 0
        elif self.config['model'] == 'EEGNet':
            self.model = EEGNet(self.config)
        self.model.load_state_dict(self.state_dict)
        self.model.eval()

        # set return_conv1 to True to get the first convolutional layer of the model
        self.model.config['return_conv1'] = True

        self.S_aggr = self.get_S_aggr(var_type='numpy')
        self.filter_spectra = self.get_individual_filter_spectra(var_type='numpy')

    def get_data(self):
        X, y, df_sample_info = read_data(self.config)

        if any(x in self.config.get('preprocessing', 'no_filter') for x in ['lowpass', 'highpass', 'bandpass']):
            X = do_filtering(X, self.config['preprocessing'])

        if 'channel_selection' in self.config.keys():
            X, self.config = channel_selection(X, self.config)

        return X, y

    def get_S_aggr(self, var_type='numpy'):
        filter_freqs, filter_freqs_all = self.get_filter_freqs(return_all=True)

        S_aggr = np.zeros((len(filter_freqs), len(filter_freqs_all)))
        for idx_unique, unique_freq in enumerate(filter_freqs):
            idcs = np.where(filter_freqs_all == unique_freq)[0]
            S_aggr[idx_unique, idcs] = 1
        S_aggr /= np.sum(S_aggr, axis=1, keepdims=True)

        if var_type == 'torch_tensor':
            S_aggr = torch.tensor(S_aggr, dtype=torch.float32)
        return S_aggr

    def get_aggr_filter_spectra(self, filter_spectra, S_aggr):
        filter_spectra_aggr = S_aggr @ filter_spectra
        return filter_spectra_aggr

    def get_individual_filter_spectra(self, var_type='numpy'):
        if 'MSA_CNN' in self.config['model']:
            len_fft_filter = self.state_dict['msm.multi_scale_convolution.convs1.0.weight'].shape[-1] // 2 + 1
            filter_spectra = np.zeros(len_fft_filter * self.config['num_filter_scales'])

            for idx_scale in range(self.config['num_filter_scales']):
                weights = self.state_dict[f'msm.multi_scale_convolution.convs1.{idx_scale}.weight']
                weights = weights.cpu().numpy().squeeze()  # filter_pos x time (8 x 15)

                filter_y = np.fft.fft(weights, axis=1)[:, :len_fft_filter].T  # freq x filter_pos (8 x 8)
                filter_y = np.abs(filter_y)

                filter_spectra[idx_scale * len_fft_filter:(idx_scale + 1) * len_fft_filter] = np.mean(filter_y, axis=1)
        elif self.config['model'] == 'EEGNet':
            len_fft_filter = self.state_dict['conv1.weight'].shape[-1] // 2
            filter_spectra = np.zeros(len_fft_filter)

            weights = self.state_dict[f'conv1.weight']
            weights = weights.cpu().numpy().squeeze()  # filter_pos x time (8 x 50)

            filter_y = np.fft.fft(weights, axis=1)[:, :len_fft_filter].T  # freq x filter_pos (25 x 8)
            filter_y = np.abs(filter_y)

            filter_spectra[:len_fft_filter] = np.mean(filter_y, axis=1)

        if var_type == 'torch_tensor':
            filter_spectra = torch.tensor(filter_spectra, dtype=torch.float32)
        return filter_spectra

    def get_class_deviations(self, X, y):
        y_unique = onehot_to_unique(y)

        psdi_mean = np.zeros((len(np.unique(y_unique)), self.config['num_channels'], 129))
        psdi_std = np.zeros((len(np.unique(y_unique)), self.config['num_channels'], 129))

        X = X.squeeze()  # labels x channels x time
        for idx_channel in range(self.config['num_channels']):
            for idx_label, label in enumerate(np.unique(y_unique)):
                mask_label = (y_unique == label)
                dev_freqs, psdi = welch(X[mask_label, idx_channel, :], axis=-1, fs=100)

                psdi_mean[idx_label, idx_channel, :] = np.mean(np.sqrt(psdi), axis=0)
                psdi_std[idx_label, idx_channel, :] = np.std(np.sqrt(psdi), axis=0)

        # channels x dev_freqs
        psd_std_between = np.std((psdi_mean), axis=0)
        psd_std_within = np.mean((psdi_std), axis=0)

        class_deviations = psd_std_between / psd_std_within

        return dev_freqs, class_deviations

    def get_filter_freqs(self, sampling_freq=100, return_all=False):
        if 'MSA_CNN' in self.config['model']:
            for idx_scale in range(self.config['num_filter_scales']):
                xf = np.fft.fftfreq(self.config['kernel_1'], 1 / (sampling_freq / (2 ** idx_scale)))
                len_fft_filter = self.config['kernel_1'] // 2 + 1

                if idx_scale == 0:
                    filter_freqs_all = xf[:len_fft_filter]
                else:
                    filter_freqs_all = np.hstack((filter_freqs_all, xf[:len_fft_filter]))

        elif self.config['model'] == 'EEGNet':
            xf = np.fft.fftfreq(50, 1 / sampling_freq)
            len_fft_filter = 50 // 2

            filter_freqs_all = xf[:len_fft_filter]

        if return_all:
            return np.sort(np.unique(filter_freqs_all)), filter_freqs_all
        return np.sort(np.unique(filter_freqs_all))

    def get_correlations(self, filter_freqs, filter_spectra, dev_freqs, class_deviations, S_aggr):
        idcs_matching = self.get_idcs_matching(filter_freqs, dev_freqs)
        corrs = np.zeros((self.config['num_channels']))

        # Use inverted S_aggr to compute correlations with correct weighting
        if isinstance(S_aggr, torch.Tensor):
            S_aggr_copy = S_aggr.clone()
        else:
            S_aggr_copy = S_aggr.copy()
        S_aggr_copy[S_aggr_copy > 0] = 1
        for idx_channel in range(self.config['num_channels']):
            corrs[idx_channel] = np.corrcoef(((S_aggr_copy.T @ filter_spectra)[S_aggr_copy.T @ filter_freqs <= 12]),
                                             ((S_aggr_copy.T @ class_deviations[idx_channel, idcs_matching])[
                                                 S_aggr_copy.T @ filter_freqs <= 12]))[0, 1]
        return corrs

    @staticmethod
    def get_idcs_matching(array_small, array_large):
        idcs = np.zeros(len(array_small), dtype=int)
        for idx_value, value in enumerate(array_small):
            # Find the index of the closest value in the larger array
            closest_idx = np.argmin(np.abs(array_large - value))
            idcs[idx_value] = closest_idx
        return idcs


class PlottingFilterSpectra(object):
    """This class is used to plot the filter spectra and the between-class variances for the ISRUC-S3 and Sleep-EDF-20
    datasets."""

    def __init__(self, model_name):
        self.model_name = None
        self.idx_model = None
        self.set_model_name(model_name)
        self.datasets = ['ISRUC', 'sleep_edf_20']
        self.dataset_labels = {'ISRUC': 'ISRUC-S3', 'sleep_edf_20': 'Sleep-EDF-20'}
        self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 5.5), gridspec_kw={'hspace': 0.1, 'wspace': 0.1}, )
        self.figure_formatting()

    def set_model_name(self, model_name):
        self.model_name = model_name
        self.idx_model = ['MSA_CNN', 'EEGNet'].index(model_name)

    def figure_formatting(self):
        props = dict(facecolor=transparent_to_opaque('slategrey', alpha=0.025), boxstyle='round,pad=0.4')

        for idx_ds in range(2):
            self.axs[1, idx_ds].set_xlabel('frequency (Hz)')
            for idx_model, model_name in enumerate(['MSA-CNN', 'EEGNet']):
                self.axs[0, idx_ds].set_xticklabels([])
                if idx_ds == 0:
                    self.axs[idx_model, idx_ds].set_ylabel('rescaled spectrum')
                elif idx_ds == 1:
                    self.axs[idx_model, idx_ds].set_yticklabels([])

                self.axs[idx_model, idx_ds].text(0.5, 0.90, model_name, ha='center', va='center',
                                                 transform=self.axs[idx_model, idx_ds].transAxes, bbox=props)

        legend_colors = ['tab:blue', 'tab:blue', transparent_to_opaque('darkviolet', 0.1)]
        legend_ls = ['-', '--', '']
        add_legend(legend_colors, legend_ls, ['filter spectrum', 'betw.-class var.', 'F3-A2'],
                   self.axs[0, 0], loc='upper left', bbox_to_anchor=(0.025, 1))
        add_legend(legend_colors, legend_ls, ['filter spectrum', 'betw.-class var.', 'Fpz-Cz'],
                   self.axs[0, 1], loc='upper left', bbox_to_anchor=(0.025, 1))

    def plot_dataset(self, idx_ds, filter_freqs, filter_spectra, freqs, class_deviations, S_aggr):
        self.plot_filter_spectra(idx_ds, filter_freqs, filter_spectra, freqs, class_deviations, S_aggr)
        self.plot_eeg_waves(max(filter_freqs))
        self.set_limits(idx_ds, max(filter_freqs))
        self.set_titles(idx_ds)

    def plot_filter_spectra(self, idx_ds, filter_freqs, filter_spectra, freqs, class_deviations, S_aggr):
        if idx_ds == 0:
            idcs_color = [0, 0, 0, 0, 1, 1, 3, 3, 2]
            idx_channel = 2
        elif idx_ds == 1:
            idcs_color = [0, 1, 3, 2]
            idx_channel = 0

        if isinstance(S_aggr, torch.Tensor):
            S_aggr_copy = S_aggr.clone()
        else:
            S_aggr_copy = S_aggr.copy()
        S_aggr_copy[S_aggr_copy > 0] = 1

        self.axs[self.idx_model, idx_ds].plot(filter_freqs, filter_spectra / np.mean(
            (S_aggr_copy.T @ filter_spectra)[(S_aggr_copy.T @ filter_freqs) <= 12]),
                                              color=self.color_cycle[idcs_color[0]], linestyle='-')
        mask_freqs = (freqs >= min(filter_freqs)) & (freqs <= max(filter_freqs))
        mask_freqs_low_freq = (freqs >= min(filter_freqs)) & (freqs <= 12)
        self.axs[self.idx_model, idx_ds].plot(freqs[mask_freqs], class_deviations[idx_channel, :][mask_freqs] / np.mean(
            class_deviations[idx_channel, :][mask_freqs_low_freq]),
                                              color=self.color_cycle[idcs_color[0]], linestyle='--')

    def plot_eeg_waves(self, max_filter_freqs):
        for idx_ds in range(2):
            self.axs[self.idx_model, idx_ds].axvspan(0.5, 4, ymin=-0.1, ymax=9, facecolor='darkviolet', alpha=0.05,
                                                     label='Delta', edgecolor='grey')
            self.axs[self.idx_model, idx_ds].axvspan(4, 8, ymin=-0.1, ymax=9, facecolor='cornflowerblue', alpha=0.05,
                                                     label='Theta', edgecolor='none')
            self.axs[self.idx_model, idx_ds].axvspan(8, 12, ymin=-0.1, ymax=9, facecolor='forestgreen', alpha=0.05,
                                                     label='Alpha', edgecolor='grey')
            self.axs[self.idx_model, idx_ds].axvspan(12, 30, ymin=-0.1, ymax=9, facecolor='orange', alpha=0.05,
                                                     label='Beta', edgecolor='none')
            self.axs[self.idx_model, idx_ds].axvspan(30, max_filter_freqs - 0.1, ymin=-0.1, ymax=9, facecolor='tomato',
                                                     alpha=0.05, label='Gamma', edgecolor='grey')

            self.axs[self.idx_model, idx_ds].text(0.5 + (4 - 0.5) / 2, 0.15, 'δ', color='darkviolet', ha='center',
                                                  va='center')
            self.axs[self.idx_model, idx_ds].text(4 + (8 - 4) / 2, 0.15, 'θ', color='cornflowerblue', ha='center',
                                                  va='center')
            self.axs[self.idx_model, idx_ds].text(8 + (12 - 8) / 2, 0.15, 'α', color='forestgreen', ha='center',
                                                  va='center')
            self.axs[self.idx_model, idx_ds].text(12 + (30 - 12) / 2, 0.15, 'β', color='orange', ha='center',
                                                  va='center')
            self.axs[self.idx_model, idx_ds].text(30 + (max_filter_freqs - 30) / 2, 0.15, 'γ', color='tomato',
                                                  ha='center', va='center')

    def set_limits(self, idx_ds, max_filter_freqs):
        self.axs[self.idx_model, idx_ds].set_xlim([0, max_filter_freqs])
        self.axs[self.idx_model, idx_ds].set_ylim([0, 2.75])

    def set_titles(self, idx_ds):
        self.axs[0, idx_ds].set_title(['ISRUC-S3', 'Sleep-EDF-20'][idx_ds], fontsize=10)

    def save(self, save_directory='figures/scale_analysis/'):
        plt.rcParams['pdf.fonttype'] = 42
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        self.fig.savefig(f'{save_directory}{self.model_name}_filter_spectra.pdf', bbox_inches='tight', pad_inches=0.0)


class PlottingCorrelationsPerformances(object):
    """This class is used to plot the correlations between the filter spectra and the between-class variances and
    compare them to the univariate classification performances, which are read from the overview file."""

    def __init__(self, model_name, file_name_overview_univariate, dir_results='results', dir_overview='overview'):
        self.model_name = model_name
        self.dir_results = dir_results
        self.dir_overview = dir_overview
        self.file_path = self.get_file_path(file_name_overview_univariate)
        self.datasets = ['ISRUC', 'sleep_edf_20']
        self.dataset_labels = {'ISRUC': 'ISRUC-S3', 'sleep_edf_20': 'Sleep-EDF-20'}
        self.x_labels, self.x_label_mapping, self.idcs_rearrange_ISRUC = self.get_x_labels()

        self.marker_o_s = {'o': 30, 's': 20, '*': 50}
        self.idcs_color = {'ISRUC': [0, 0, 0, 0, 1, 1, 3, 3, 2], 'sleep_edf_20': [0, 1, 3, 2]}
        self.color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.color_cycle[1] = transparent_to_opaque(self.color_cycle[0], 0.7)

        self.fig, self.axs = self.get_figure()
        self.set_figure_formatting()

    def get_file_path(self, file_name_overview_univariate):
        return os.path.join(self.dir_results, self.dir_overview, file_name_overview_univariate)

    def get_figure(self):
        width_ratio = 1.25
        height_ratio = 1.3
        fig = plt.figure(figsize=(6.4, 5.8))
        gs = GridSpec(2, 2, figure=fig, width_ratios=[width_ratio, 1], height_ratios=[height_ratio, 1], wspace=0.18,
                      hspace=0.1)
        axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)])
        return fig, axs

    def set_figure_formatting(self):
        self.axs[0, 0].text(-0.27, 0.5, 'correlation (filter spectrum\nvs between-class variation)',
                            va='center', ha='center', rotation='vertical', transform=self.axs[0, 0].transAxes)
        self.axs[1, 0].text(-0.27, 0.5, 'performance\n(classification acc.)',
                            va='center', ha='center', rotation='vertical', transform=self.axs[1, 0].transAxes)
        self.fig.text(0.5, -0.0, 'channel', ha='center')

        for idx_ds in range(2):
            self.axs[0, idx_ds].spines[['right', 'top']].set_visible(False)
            self.axs[1, idx_ds].spines[['right', 'top']].set_visible(False)

            self.set_ticks(idx_ds)
            self.set_titles(idx_ds)

        legend_ls = ['o', '*']
        add_legend(['grey', 'grey'], legend_ls, ['MSA-CNN', 'EEGNet'], self.axs[0, 0],
                   loc='lower left', s=[self.marker_o_s[ls_i] for ls_i in legend_ls])

    def plot_dataset(self, idx_ds, corrs_i):
        self.plot_correlations(idx_ds, corrs_i)
        self.plot_performance(idx_ds, corrs_i)

    def plot_correlations(self, idx_ds, corrs_i):
        if self.model_name == 'MSA_CNN':
            marker = 'o'
        elif self.model_name == 'EEGNet':
            marker = '*'
        if idx_ds == 0:
            corrs_i_rearr = corrs_i[self.idcs_rearrange_ISRUC]
            self.axs[0, 0].scatter(np.arange(4), corrs_i_rearr[:4], color=self.color_cycle[0], marker=marker,
                                   s=self.marker_o_s[marker])
            self.axs[0, 0].scatter(np.arange(4, 6), corrs_i_rearr[4:6], color=self.color_cycle[1], marker=marker,
                                   s=self.marker_o_s[marker])
            self.axs[0, 0].scatter(np.arange(6, 8), corrs_i_rearr[6:8], color=self.color_cycle[3], marker=marker,
                                   s=self.marker_o_s[marker])
            self.axs[0, 0].scatter([8], [corrs_i_rearr[8]], color=self.color_cycle[2], marker=marker,
                                   s=self.marker_o_s[marker])
            self.axs[0, 0].plot(corrs_i_rearr[:4], color=self.color_cycle[0])
            self.axs[0, 0].plot(np.arange(4, 6), corrs_i_rearr[4:6], color=self.color_cycle[1])
            self.axs[0, 0].plot(np.arange(6, 8), corrs_i_rearr[6:8], color=self.color_cycle[3])
        elif idx_ds == 1:
            for idx_channel, corr_i in enumerate(corrs_i):
                self.axs[0, 1].scatter(idx_channel, corr_i,
                                       color=self.color_cycle[self.idcs_color['sleep_edf_20'][idx_channel]],
                                       marker=marker,
                                       s=self.marker_o_s[marker])

        self.set_plot_elements(idx_ds, corrs_i)

    def plot_performance(self, idx_ds, corrs_i):
        if self.model_name == 'MSA_CNN':
            marker = 'o'
        elif self.model_name == 'EEGNet':
            marker = '*'

        df_univariate = pd.read_csv(self.file_path, index_col=0)
        if df_univariate['model'].iloc[-1] == 'EEGNet':
            df_univariate = df_univariate[df_univariate['idx_experiment'] == 3]
        print(df_univariate.to_string())

        y_values = self.get_y_values(df_univariate, self.x_labels[self.datasets[idx_ds]], break_loc=[0.7, None][idx_ds],
                                     break_shift=[0.15, None][idx_ds])

        for idx_channel, y_value in enumerate(y_values):
            self.axs[1, idx_ds].scatter(idx_channel, y_value,
                                        color=self.color_cycle[self.idcs_color[self.datasets[idx_ds]][idx_channel]],
                                        marker=marker, s=self.marker_o_s[marker])

        if idx_ds == 0:
            self.axs[1, 0].plot(y_values[:4],
                                color=self.color_cycle[0])
            self.axs[1, 0].plot(np.arange(4, 6), y_values[4:6],
                                color=self.color_cycle[1])
            self.axs[1, 0].plot(np.arange(6, 8), y_values[6:8],
                                color=self.color_cycle[3])

        self.set_limits(idx_ds, corrs_i)

    def set_plot_elements(self, idx_ds, corrs_i):
        self.axs[0, idx_ds].hlines(0, -1, len(corrs_i) + 1, color='grey', linestyle='--')

    def get_y_values(self, df_univariate, x_labels, break_loc=None, break_shift=None):
        y_values = []
        for x_label in x_labels:
            mask = df_univariate['channel_selection'] == self.x_label_mapping[x_label]
            y_value = df_univariate['test_accuracy'][mask].mean()
            if break_loc is not None and break_shift is not None:
                if y_value < break_loc:
                    y_value += break_shift
            y_values.append(y_value)
        return np.array(y_values)

    def set_limits(self, idx_ds, corrs_i):
        self.axs[0, idx_ds].set_xlim([0 - 0.4, len(corrs_i) - 1 + 0.4])
        self.axs[0, idx_ds].set_ylim([-0.65, 1])
        self.axs[1, idx_ds].set_xlim([0 - 0.4, len(corrs_i) - 1 + 0.4])
        if idx_ds == 0:
            ylim = [0.5, 0.85]
        elif idx_ds == 1:
            ylim = [0.5, 0.85]
        self.axs[1, idx_ds].set_ylim(ylim)

    def set_ticks(self, idx_ds):
        if idx_ds == 1:
            self.axs[0, idx_ds].set_yticklabels([])
            self.axs[1, idx_ds].set_yticklabels([])
            self.axs[1, idx_ds].set_xticklabels([])

        xticks = np.arange(len(self.x_labels[self.datasets[idx_ds]]))
        self.axs[0, idx_ds].set_xticks(xticks, [])
        self.axs[1, idx_ds].set_xticks(xticks, [])

        for tick, label in zip(xticks, self.x_labels[self.datasets[idx_ds]]):
            self.axs[1, idx_ds].text(tick + 0.03, -0.02, label, ha='right', va='top', rotation=30,
                                     transform=self.axs[1, idx_ds].get_xaxis_transform())

    def set_titles(self, idx_ds):
        self.axs[0, idx_ds].set_title(['ISRUC-S3', 'Sleep-EDF-20'][idx_ds], fontsize=10)

    def save(self, save_directory='figures/scale_analysis/'):
        plt.rcParams['pdf.fonttype'] = 42
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        self.fig.savefig(f'{save_directory}{self.model_name}_correlation_datasets.pdf', bbox_inches='tight',
                         pad_inches=0.0)

    @staticmethod
    def get_x_labels():
        x_labels_ISRUC = ['C3-A2', 'C4-A1', 'F3-A2', 'F4-A1', 'O1-A2', 'O2-A1', 'lEOG-A2', 'rEOG-A1', 'EMG chin']
        x_labels_sleep_edf_20 = ['Fpz-Cz', 'Pz-Oz', 'hEOG']
        x_labels = {'ISRUC': np.array(x_labels_ISRUC), 'sleep_edf_20': np.array(x_labels_sleep_edf_20)}
        x_label_mapping = {key: key for key in x_labels_ISRUC + x_labels_sleep_edf_20}
        x_label_mapping['lEOG-A2'] = 'LOC-A2'
        x_label_mapping['rEOG-A1'] = 'ROC-A1'
        x_label_mapping['EMG chin'] = 'chin EMG'
        x_label_mapping['Fpz-Cz'] = 'EEG Fpz-Cz'
        x_label_mapping['Pz-Oz'] = 'EEG Pz-Oz'
        x_label_mapping['hEOG'] = 'EOG'
        idcs_rearrange_ISRUC = [2, 3, 0, 1, 4, 5, 6, 7, 8]
        x_labels['ISRUC'] = x_labels['ISRUC'][idcs_rearrange_ISRUC]
        return x_labels, x_label_mapping, idcs_rearrange_ISRUC


def get_analysis_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_1", type=str)
    parser.add_argument("--model_1_ISRUC", type=str)
    parser.add_argument("--model_1_sleep_edf_20", type=str)
    parser.add_argument("--file_name_overview_univariate_1", type=str)
    parser.add_argument("--model_name_2", type=str, default="none")
    parser.add_argument("--model_2_ISRUC", type=str, default="none")
    parser.add_argument("--model_2_sleep_edf_20", type=str, default="none")
    parser.add_argument("--file_name_overview_univariate_2", type=str, default="none")
    parser.add_argument("--results_folder", type=str, default="results")
    parser.add_argument("--overview_folder", type=str, default="overview")
    parser.add_argument("--models_folder", type=str, default="models")
    args, _ = parser.parse_known_args()
    return vars(args)


def main_analysis_filters(model_name=None, model_ISRUC=None, model_sleep_edf_20=None,
                          file_name_overview_univariate=None,
                          results_folder=None, models_folder=None, plotting_filter_spectra=None,
                          plotting_corrs_performs=None, plot_figure=True):
    """This function is used to carry out the filter spectrum analysis presented in "Retrieving Filter Spectra in CNN
    for Explainable Sleep Stage Classification" given the trained model in various configurations. Specifically, it
    uses the trained multivariate model for the ISRUC-S3 and Sleep-EDF-20 datasets to analyze the filter spectra, the
    raw datasets to compute the between-class variance, and the univariate classification results to compare the filter
    spectra - between-class variance correlation with the classification performance. The results are visualized and
    saved in the figures folder."""

    # Load the trained models
    model_full_states = {
        'ISRUC': torch.load(os.path.join(results_folder, models_folder, model_ISRUC)),
        'sleep_edf_20': torch.load(os.path.join(results_folder, models_folder, model_sleep_edf_20))
    }

    if plotting_filter_spectra is None:
        plotting_filter_spectra = PlottingFilterSpectra(model_name=model_name)
    else:
        plotting_filter_spectra.set_model_name(model_name)
    if plotting_corrs_performs is None:
        plotting_corrs_performs = PlottingCorrelationsPerformances(
            model_name=model_name, file_name_overview_univariate=file_name_overview_univariate)
    else:
        plotting_corrs_performs.model_name = model_name
        plotting_corrs_performs.file_path = plotting_corrs_performs.get_file_path(file_name_overview_univariate)

    for idx_ds, (dataset, full_state) in enumerate(model_full_states.items()):
        print(f'Processing dataset {dataset}...')

        proc_filter_spectra = DataProcessingFilterSpectra(full_state['config'], full_state['state_dict'])

        # Get results for figure 1: filter spectra and between-class variances
        S_aggr = proc_filter_spectra.get_S_aggr()

        filter_freqs = proc_filter_spectra.get_filter_freqs()
        filter_spectra = proc_filter_spectra.get_individual_filter_spectra()
        filter_spectra_aggr = proc_filter_spectra.get_aggr_filter_spectra(filter_spectra, S_aggr)
        X, y = proc_filter_spectra.get_data()
        dev_freqs, class_deviations = proc_filter_spectra.get_class_deviations(X, y)

        # Get results for figure 2: comparison of filter spectra - between-class variance correlation with
        # classification performance
        corrs = proc_filter_spectra.get_correlations(filter_freqs, filter_spectra_aggr, dev_freqs, class_deviations,
                                                     proc_filter_spectra.S_aggr)

        plotting_filter_spectra.plot_dataset(idx_ds, filter_freqs, filter_spectra_aggr, dev_freqs, class_deviations,
                                             proc_filter_spectra.S_aggr)
        plotting_corrs_performs.plot_dataset(idx_ds, corrs)

    if plot_figure:
        plotting_filter_spectra.save()
        plotting_corrs_performs.save()
        plt.show()
    return plotting_filter_spectra, plotting_corrs_performs


if __name__ == '__main__':
    main_analysis_filters()
