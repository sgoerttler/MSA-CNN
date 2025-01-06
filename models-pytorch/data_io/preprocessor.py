import scipy.signal as signal


def do_filtering(X, preprocess_mode, sampling_rate=100, order=4):
    nyquist = 0.5 * sampling_rate
    frequency_cutoff, filter_type = preprocess_mode.split('Hz_')
    frequency_cutoff = int(frequency_cutoff)
    b, a = signal.butter(order, frequency_cutoff / nyquist, btype=filter_type)
    for idx_sample in range(X.shape[0]):
        X[idx_sample, :, :, :] = signal.filtfilt(b, a, X[idx_sample, :, :, :], axis=2)
    return X


def channel_selection(X, config, channel_cardinality_in='reduced'):
    """
    Select channels from the data based on the configuration settings.
    ISRUC channels: C3-A2, C4-A1, F3-A2, F4-A1, O1-A2, O2-A1, LOC-A2 (EOG), ROC-A1 (EOG), chin EMG, [ECG]
    sleep-edf channels: EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, [Resp oro-nasal], EMG submental, [Temp rectal]
    Data input channel cardinality 'reduced' assumes that channels in square brackets are not present in the prepared
    data to save memory. Select 'full' if all channels are present.
    """
    if config['channel_selection'] == 'all':
        print(X.shape)
        print(X.shape[-2])
        idcs_select = list(range(X.shape[-2]))
    elif config['channel_selection'] == 'EEG/EOG/EMG':
        if 'ISRUC' in config['data']:
            idcs_select = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        elif 'sleep_edf' in config['data']:
            if channel_cardinality_in == 'full':
                idcs_select = [0, 1, 2, 4]
            elif channel_cardinality_in == 'reduced':
                idcs_select = [0, 1, 2, 3]
            else:
                raise ValueError(f'Unknown channel cardinality: {channel_cardinality_in}')
        else:
            raise ValueError(f'Unknown dataset: {config["data"]}')
    elif config['channel_selection'].isnumeric():
        if 'ISRUC' in config['data']:
            idcs_select = [1, 2, 5, 7, 8, 0, 3, 4, 6, 9][:int(config['channel_selection'])]
        elif 'sleep_edf' in config['data']:
            if channel_cardinality_in == 'full':
                idcs_select = [0, 1, 2, 4][:int(config['channel_selection'])]
            elif channel_cardinality_in == 'reduced':
                idcs_select = [0, 1, 2, 3][:int(config['channel_selection'])]
            else:
                raise ValueError(f'Unknown channel cardinality: {channel_cardinality_in}')
        else:
            raise ValueError(f'Unknown dataset: {config["data"]}')
    elif config['channel_selection'] == 'EEG Fpz-Cz':  # sleep-edf
        idcs_select = [0]
    elif config['channel_selection'] == 'C4-A1':  # ISRUC
        idcs_select = [1]
    elif config['channel_selection'] == 'F4-A1':  # ISRUC
        idcs_select = [3]
    else:
        raise ValueError(f'Unknown channel selection: {config["channel_selection"]}')

    if len(X.shape) == 3:
        X = X[:, idcs_select, :]
    elif len(X.shape) == 4:
        X = X[:, :, idcs_select, :]

    return X, config
