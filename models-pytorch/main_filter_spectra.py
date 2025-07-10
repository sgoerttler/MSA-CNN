import sys
import os
import pandas as pd

from main import main
from analysis.analyze_filters import main_analysis_filters, get_analysis_args
from utils.utils import set_working_directory


def get_file_name_overview(model_name='MSA_CNN', model='multivariate'):
    dir_overview = os.path.join('results', 'overview')
    if 'MSA_CNN' in model_name:
        file_name_prefix = f'overview_results_MSA_CNN_fourier_filters'
    else:
        file_name_prefix = f'overview_results_{model_name}_fourier_filters'
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

    # loop over one or two models
    plotting_filter_spectra, plotting_corrs_performs, plot_figure = None, None, False
    for idx_model in [1, 2]:
        if args_analysis['model_name_2'] == 'none':
            plot_figure = True
            if idx_model == 2:
                continue
        elif idx_model == 2:
            plot_figure = True

        # run the multivariate model on both datasets if the models are not specified
        if args_analysis[f'model_{idx_model}_ISRUC'] is None or args_analysis[f'model_{idx_model}_sleep_edf_20'] is None:
            sys.argv.insert(1, os.path.join('configs', f'{args_analysis[f"model_name_{idx_model}"]}_fourier_filters.json'))
            main()

            args_analysis[f'file_name_overview_{idx_model}'] = get_file_name_overview()
            args_analysis[f'model_{idx_model}_ISRUC'], args_analysis[f'model_{idx_model}_sleep_edf_20'] = (
                get_model_names(args_analysis[f'file_name_overview_{idx_model}']))

        # run the univariate model for all channels on both datasets if the model overview file is not specified
        if args_analysis[f'file_name_overview_univariate_{idx_model}'] is None:
            sys.argv.insert(1, os.path.join('configs', f'{args_analysis[f"model_name_{idx_model}"]}_fourier_filters_univariate_{idx_model}.json'))
            main()

            args_analysis[f'file_name_overview_univariate_{idx_model}'] = get_file_name_overview(model_name=args_analysis[f'model_name_{idx_model}'],
                                                                                    model='univariate')

        # run the frequency analysis with the trained univariate and multivariate models
        plotting_filter_spectra, plotting_corrs_performs = main_analysis_filters(
            args_analysis[f'model_name_{idx_model}'], args_analysis[f'model_{idx_model}_ISRUC'],
            args_analysis[f'model_{idx_model}_sleep_edf_20'], args_analysis[f'file_name_overview_univariate_{idx_model}'],
            args_analysis['results_folder'], args_analysis['models_folder'], plotting_filter_spectra, plotting_corrs_performs, plot_figure=plot_figure)


if __name__ == '__main__':
    main_filter_spectra()
