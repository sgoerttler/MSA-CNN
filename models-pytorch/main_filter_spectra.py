import sys
import os
import pandas as pd

from main import main
from analysis.analyze_filters import main_analysis_filters, get_analysis_args
from utils.utils import set_working_directory


def get_file_name_overview(model='multivariate'):
    dir_overview = os.path.join('results', 'overview')
    file_name_prefix = 'overview_results_MSA_CNN_fourier_filters'
    files = [f for f in os.listdir(dir_overview) if f.startswith(file_name_prefix)]
    if model == 'univariate':
        return next((f for f in files if 'univariate' in f), None)
    else:
        return next((f for f in files if 'univariate' not in f), None)


def get_model_names(file_name_overview):
    dir_overview = os.path.join('results', 'overview')
    df = pd.read_csv(os.path.join(dir_overview, file_name_overview), index_col=0)

    # get last completed run for each dataset
    file_name_ISRUC = df.loc[(df['data'] == 'ISRUC') & df['run_complete'], 'main_results_filename'].values[-1]
    file_name_sleep_edf_20 = df.loc[(df['data'] == 'sleep_edf_20') & df['run_complete'], 'main_results_filename'].values[-1]
    return file_name_ISRUC, file_name_sleep_edf_20


def main_filter_spectra():
    """This function combines running the univariate and multivariate models for the filter spectra analysis and the
    analysis itself. The models can either be run from scratch or specified by providing arguments."""
    set_working_directory()
    args_analysis = get_analysis_args()

    # run the multivariate model on both datasets if the models are not specified
    if args_analysis['model_ISRUC'] is None or args_analysis['model_sleep_edf_20'] is None:
        sys.argv.insert(1, os.path.join('configs', 'MSA_CNN_fourier_filters.json'))
        main()

        args_analysis['file_name_overview'] = get_file_name_overview()
        args_analysis['model_ISRUC'], args_analysis['model_sleep_edf_20'] = (
            get_model_names(args_analysis['file_name_overview']))

    # run the univariate model for all channels on both datasets if the model overview file is not specified
    if args_analysis['file_name_overview_univariate'] is None:
        sys.argv.insert(1, os.path.join('configs', 'MSA_CNN_fourier_filters_univariate.json'))
        main()

        args_analysis['file_name_overview_univariate'] = get_file_name_overview(model='univariate')

    # run the frequency analysis with the trained univariate and multivariate models
    main_analysis_filters(args_analysis)


if __name__ == '__main__':
    main_filter_spectra()
