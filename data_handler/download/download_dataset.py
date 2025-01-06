"""Download sleep stage classification dataset from the internet. Implementation by sgoerttlers."""

import subprocess
import os
import shutil


def check_tool(tool_name):
    if shutil.which(tool_name) is None:
        raise EnvironmentError(f"{tool_name} command line tool is required to download the dataset, but not found. "
                               f"Please install it.")


def download_dataset(dataset):
    check_tool('wget')
    if dataset == 'ISRUC':
        check_tool('unrar')

    print(f'Downloading {dataset} dataset...')
    filename = {
        'ISRUC': 'download_isruc_s3',
        'sleep_edf_20': 'download_sleep_edf_20',
        'sleep_edf_78': 'download_sleep_edf_78'
    }
    script_path = os.path.join(os.path.dirname(__file__), f'{filename[dataset]}.sh')
    subprocess.run(['bash', script_path], check=True)
