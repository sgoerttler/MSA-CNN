import numpy as np
import re
import ast


def unique(sequence):
    """Return unique elements of a sequence while preserving order."""
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


def array2str(arr, precision=None):
    """Convert numpy array to string."""
    s = np.array_str(arr, precision=precision)

    # remove unnecessary characters
    s.replace('\n', ',')
    s = re.sub('\[ +', '[', s.strip())
    s = re.sub('[,\s]+', ',', s)

    return s


def str2array(s):
    """Convert numpy array saved as string back to numpy array."""
    return np.array(ast.literal_eval(s), dtype=int)


def print_df(df, num_head=int(1e9), num_tail=int(1e9), repeat_header=True):
    """Print dataframe with header colored and repeated after body."""
    header, body = df.head(num_head).tail(num_tail).to_string().split('\n')[0], '\n'.join(df.head(num_head).tail(num_tail).to_string().split('\n')[1:])
    print(f'\033[94m{header}\033[0m')
    print(body)
    if repeat_header:
        print(f'\033[94m{header}\033[0m')
    print()
