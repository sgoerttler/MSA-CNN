import os
import glob
import numpy as np
import math
from scipy.signal import stft
from scipy.fftpack import fft

from ..utils import unique_to_onehot, get_fold_idx


class Preprocessor:
    def __init__(self, dataset, data_in_folder, data_save_folder, save_mode='np_array', channel_selection=None):
        self.dataset = dataset
        self.data_in_folder = data_in_folder
        if 'sleep_edf' in self.dataset:
            self.data_in_folder_prep = os.path.join(self.data_in_folder, 'cleaned')
        self.data_save_folder = data_save_folder
        self.save_mode = save_mode
        if channel_selection is None:
            if dataset == 'ISRUC':
                self.channel_selection = np.arange(10)
            elif 'sleep_edf' in dataset:
                self.channel_selection = np.arange(6)
        else:
            self.channel_selection = channel_selection

        self.data_out_X = []
        self.data_out_y = []
        self.data_out_fold_len = []
        self.data_out_idx_fold = []
        self.data_out_idx_part = []
        self.data_out_idx_epoch = []

    def preprocess(self, modes, Xy_only=False):
        """Logic for preprocessing the datasets. Implemented by sgoerttler."""

        # check if more than one preprocessing mode is selected, if so loop over modes
        if not isinstance(modes, (list, tuple)):
            modes = [modes]

        print(f"Preprocessing data for dataset: {self.dataset}\n"
              f"Mode(s): {', '.join(modes)}")
        if self.dataset == 'ISRUC':
            file_path = os.path.join(self.data_in_folder, 'ISRUC.npz')
            data = np.load(file_path, allow_pickle=True)
            for idx_fold, (X, y) in enumerate(zip(data['Fold_data'], data['Fold_label'])):
                print('Fold:', idx_fold)
                self.preprocess_i(X, y, idx_part=idx_fold, modes=modes, Xy_only=Xy_only)
        elif 'sleep_edf' in self.dataset:
            parts = []
            files = glob.glob(os.path.join(self.data_in_folder_prep, f"*.npz"))
            for file in files:
                parts.append(os.path.split(file)[-1][:5])
            for idx_part, part in enumerate(np.unique(parts)):
                print('Participant:', part)
                files_part = glob.glob(os.path.join(self.data_in_folder_prep, f"{part}*.npz"))
                for idx, files_part_i in enumerate(files_part):
                    data_in = np.load(files_part_i)
                    # swap axes of data to match ISRUC data shape
                    X = np.swapaxes(data_in['x'], 1, 2)
                    y = data_in['y']
                    self.preprocess_i(X, y, idx_part=idx_part, modes=modes, Xy_only=Xy_only)

        # convert data to array depending on the implementation, fold length not required in this case
        if self.save_mode == 'np_array':
            self.data_out_X = np.concatenate(self.data_out_X, axis=0)
            self.data_out_y = np.concatenate(self.data_out_y, axis=0)
            if not Xy_only:
                self.data_out_idx_fold = np.concatenate(self.data_out_idx_fold, axis=0)
                self.data_out_idx_part = np.concatenate(self.data_out_idx_part, axis=0)
                self.data_out_idx_epoch = np.concatenate(self.data_out_idx_epoch, axis=0)

    def preprocess_i(self, X, y=None, idx_part=None, modes='reshaping', Xy_only=False):

        if len(modes) > 1:
            X_preps_i = []
        for idx_mode, mode in enumerate(modes):
            if mode == 'reshaping':
                X_prep_i = X
            elif mode == 'stft':
                X_prep_i = self.process_batch_signal(X, n_length=[100, 50][int(self.dataset == 'sleep_edf_78')])
            elif mode == 'interpolation':
                X_prep_i = np.apply_along_axis(
                    lambda x: np.interp(np.linspace(0, len(x), 500), np.arange(len(x)), x), axis=2, arr=X)
            elif mode == 'de':
                X_prep_i = np.array([self.DE_PSD(X_sample)[0] for X_sample in X])
            elif mode == 'psd':
                X_prep_i = np.array([self.DE_PSD(X_sample)[1] for X_sample in X])

            if mode == 'stft':
                X_prep_i = X_prep_i[:, :, :, self.channel_selection]
            else:
                X_prep_i = X_prep_i[:, self.channel_selection, :]

            if len(modes) > 1:
                X_preps_i.append(X_prep_i)

        self.data_out_X.append(X_preps_i if len(modes) > 1 else X_prep_i)
        if y is not None:
            if np.prod(y.shape) == y.shape[0]:
                y_onehot = unique_to_onehot(y, num_classes=5)
            else:
                y_onehot = y
            self.data_out_y.append(y_onehot)
        if not Xy_only:
            self.data_out_fold_len.append(X_prep_i.shape[0])
            self.data_out_idx_fold.append(np.ones(X_prep_i.shape[0], dtype=int) * get_fold_idx(idx_part))
            self.data_out_idx_part.append(np.ones(X_prep_i.shape[0], dtype=int) * idx_part)
            self.data_out_idx_epoch.append(np.arange(X_prep_i.shape[0]))

    def process_batch_signal(self, signals, n_length=100):
        """Process a batch of signals. Implemented by Zhanjiang."""
        signal_spectrogram_list = []
        for record in signals:
            for i in range(len(record)):
                if (i == 0):
                    spectrogram = self.get_spectrogram(record[i], n_length)
                else:
                    spectrogram = np.concatenate((spectrogram, self.get_spectrogram(record[i], n_length)), axis=-1)

            img = np.abs(spectrogram)
            img = np.pad(img, ((0, 0), (2000 // n_length - 1, 2000 // n_length), (0, 0)), 'constant', constant_values=(0))
            img = self.avg_pooling(img, (100 // n_length) ** 2, axis=1)
            signal_spectrogram_list.append(img)

        return np.array(signal_spectrogram_list)

    def DE_PSD(self, data, stft_para=None):
        '''
        compute DE and PSD (original function from ziyujia)
        --------
        input:  data [n*m]          n electrodes, m time points
                stft_para.stftn     frequency domain sampling rate
                stft_para.fStart    start frequency of each frequency band
                stft_para.fEnd      end frequency of each frequency band
                stft_para.window    window length of each sample point(seconds)
                stft_para.fs        original frequency
        output: psd,DE [n*l*k]        n electrodes, l windows, k frequency bands
        '''

        # the parameters to extract DE and PSD
        if stft_para is None:
            stft_para = {
                'stftn': 3000,
                'fStart': [0.5, 2, 4, 6, 8, 11, 14, 22, 31],
                'fEnd': [4, 6, 8, 11, 14, 22, 31, 40, 50],
                'fs': 100,
                'window': 30,
            }

        # initialize the parameters
        STFTN = stft_para['stftn']
        fStart = stft_para['fStart']
        fEnd = stft_para['fEnd']
        fs = stft_para['fs']
        window = stft_para['window']

        fStartNum = np.zeros([len(fStart)], dtype=int)
        fEndNum = np.zeros([len(fEnd)], dtype=int)
        for i in range(0, len(stft_para['fStart'])):
            fStartNum[i] = int(fStart[i] / fs * STFTN)
            fEndNum[i] = int(fEnd[i] / fs * STFTN)

        n = data.shape[0]

        psd = np.zeros([n, len(fStart)])
        de = np.zeros([n, len(fStart)])

        # Hanning window
        Hlength = window * fs
        Hwindow = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (Hlength + 1)) for n in range(1, Hlength + 1)])

        dataNow = data[0:n]
        for j in range(0, n):
            temp = dataNow[j]
            Hdata = temp * Hwindow
            FFTdata = fft(Hdata, STFTN)
            magFFTdata = abs(FFTdata[0:int(STFTN / 2)])
            for p in range(0, len(fStart)):
                E = 0
                for p0 in range(fStartNum[p] - 1, fEndNum[p]):
                    E = E + magFFTdata[p0] * magFFTdata[p0]
                E = E / (fEndNum[p] - fStartNum[p] + 1)

                psd[j][p] = E
                de[j][p] = math.log(max(100 * E, 1e-12), 2)

        return de, psd

    @staticmethod
    def get_spectrogram(waveform, n_length):
        """Compute the spectrogram of a 1D waveform. Implemented by Zhanjiang."""
        _, _, spectrogram = stft(waveform, fs=1.0, nperseg=n_length,
                                 window='hann', nfft=None, noverlap=None, return_onesided=False)
        spectrogram = np.abs(spectrogram)
        # Obtain the magnitude of the STFT.
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).

        spectrogram = spectrogram[..., np.newaxis]
        return spectrogram

    @staticmethod
    def avg_pooling(x, downsampling_factor, axis=0):
        """Perform average pooling. Implemented by Zhanjiang."""
        x_newshape = np.array(x.shape)
        x_newshape[axis] = x_newshape[axis] // downsampling_factor
        x_newshape = np.insert(x_newshape, axis + 1, downsampling_factor)
        x = x.reshape(x_newshape).mean(axis=axis + 1)
        return x

    def save_preprocessed_data(self, suffix, keep_suffix_y=True, Xy_only=False):
        if self.save_mode == 'np_array':
            # save data per sample, which are homogenous and can be saved as an array
            np.save(os.path.join(self.data_save_folder, f'{self.dataset}_{suffix}_X.npy'.replace('__', '_')),
                    self.data_out_X)
            if keep_suffix_y:
                filename = f'{self.dataset}_{suffix}_y.npy'.replace('__', '_')
            else:
                filename = f'{self.dataset}_y.npy'
            np.save(os.path.join(self.data_save_folder, filename), self.data_out_y)
            if not Xy_only:
                np.save(os.path.join(self.data_save_folder, f'{self.dataset}_idx_fold.npy'), self.data_out_idx_fold)
                np.save(os.path.join(self.data_save_folder, f'{self.dataset}_idx_part.npy'), self.data_out_idx_part)
                np.save(os.path.join(self.data_save_folder, f'{self.dataset}_idx_epoch.npy'), self.data_out_idx_epoch)
        elif self.save_mode == 'list':
            # save data not per sample, but per participant, which means that the data is inhomogenous
            fold_data = np.asarray(self.data_out_X, dtype=object)
            fold_label = np.asarray(self.data_out_y, dtype=object)
            fold_len = np.asarray(self.data_out_fold_len, dtype=object)
            np.savez(os.path.join(self.data_save_folder, f'{self.dataset}_{suffix}.npz'.replace('_.', '.')),
                     Fold_data=fold_data,
                     Fold_label=fold_label,
                     Fold_len=fold_len)
        print(f'Saved preprocessed data to folder {self.data_save_folder}\n')
        self.reset_data()

    def reset_data(self):
        self.data_out_X = []
        self.data_out_y = []
        self.data_out_idx_fold = []
        self.data_out_fold_len = []


def preprocessing_cvan(dataset, data_in_folder, data_out_folder, zip_modes=False):
    if dataset == 'ISRUC':
        channel_selection = np.arange(10)
    elif 'sleep_edf' in dataset:
        channel_selection = [0, 1, 2, 4]
    preprocessor = Preprocessor(dataset,
                                data_in_folder,
                                data_out_folder,
                                save_mode='np_array',
                                channel_selection=channel_selection)
    if zip_modes:
        preprocessor.preprocess(['interpolation', 'stft'])
        preprocessor.save_preprocessed_data(suffix='interpolation_stft')
    else:
        preprocessor.preprocess('interpolation')
        preprocessor.save_preprocessed_data(suffix='interpolation')
        preprocessor.preprocess('stft', Xy_only=True)
        preprocessor.save_preprocessed_data(suffix='stft', Xy_only=True)


def preprocessing_graphsleepnet(dataset, data_in_folder, data_out_folder):
    save_mode = 'list'
    if dataset == 'ISRUC':
        channel_selection = np.arange(10)
    elif 'sleep_edf' in dataset:
        channel_selection = [0, 1, 2, 4]
    preprocessor = Preprocessor(dataset,
                                data_in_folder,
                                data_out_folder,
                                save_mode=save_mode,
                                channel_selection=channel_selection)
    preprocessor.preprocess('de')
    preprocessor.save_preprocessed_data('de')


def preprocessing_reshape(dataset, data_in_folder, data_out_folder, channel_selection=None, save_mode='np_array', suffix=''):
    # set default channel selection, convert to list if not already
    if channel_selection is None:
        if dataset == 'ISRUC':
            channel_selection = np.arange(10)
        elif 'sleep_edf' in dataset:
            channel_selection = [0, 1, 2, 4]
    if not isinstance(channel_selection, (list, tuple)):
        channel_selection = [channel_selection]

    preprocessor = Preprocessor(dataset, data_in_folder, data_out_folder,
                                save_mode=save_mode, channel_selection=channel_selection)
    preprocessor.preprocess('reshaping')
    preprocessor.save_preprocessed_data(suffix)


def preprocessing_single_channel(dataset, data_in_folder, data_out_folder, channel=None):
    channel_names_isruc = ['C3', 'C4', 'F3', 'F4', 'O1', 'O2', 'L-EOG', 'R-EOG', 'EMG', 'ECG']
    channel_names_sleep_edf = ['Fpz-Cz', 'Pz-Oz', 'EOG', 'EMG']
    if dataset == 'ISRUC':
        if channel is None:
            channel = 'C4'  # default single channel
        channel_selection = [channel_names_isruc.index(channel)]
    elif 'sleep_edf' in dataset:
        if channel is None:
            channel = 'Fpz-Cz'  # default single channel
        channel_selection = [channel_names_sleep_edf.index(channel)]
    preprocessor = Preprocessor(dataset, data_in_folder, data_out_folder,
                                save_mode='np_array', channel_selection=channel_selection)
    preprocessor.preprocess('reshaping')
    preprocessor.save_preprocessed_data(channel, keep_suffix_y=False)


def main():
    ds_folder = {'ISRUC': 'ISRUC',
                 'sleep_edf_20': 'Sleep_EDF_20',
                 'sleep_edf_78': 'Sleep_EDF_78'}
    single_channel = {'ISRUC': 'C4',
                      'sleep_edf_20': 'Fpz-Cz',
                      'sleep_edf_78': 'Fpz-Cz'}

    for dataset in ['ISRUC', 'sleep_edf_20', 'sleep_edf_78']:
        data_in_folder = '../../data/dataset_name/'.replace('dataset_name', ds_folder[dataset])
        data_out_folder = '../../data/dataset_name/'.replace('dataset_name', ds_folder[dataset])

        preprocessing_cvan(dataset, data_in_folder, data_out_folder)
        preprocessing_graphsleepnet(dataset, data_in_folder, data_out_folder)
        preprocessing_reshape(dataset, data_in_folder, data_out_folder, save_mode='np_array')
        preprocessing_reshape(dataset, data_in_folder, data_out_folder, save_mode='list')
        if dataset == 'ISRUC':
            preprocessing_reshape(dataset, data_in_folder, data_out_folder, channel_selection=np.arange(9), suffix='without_ECG')
        preprocessing_single_channel(dataset, data_in_folder, data_out_folder, channel=single_channel[dataset])


if __name__ == "__main__":
    main()
