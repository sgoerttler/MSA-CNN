import os
import numpy as np
import re
import ast

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors


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


def add_legend(colors, linestyles, labels, plot_axis, loc=None, bbox_to_anchor=None, s=None, return_legend=False, prop=None, framealpha=None):
    custom_lines = []
    if s is None:
        s = [50] * len(colors)
    xlim = plot_axis.get_xlim()
    ylim = plot_axis.get_ylim()
    for color, linestyle, si in zip(colors, linestyles, s):
        if linestyle == 'full':
            custom_lines.append(Patch(facecolor=color))
        elif (linestyle == 'o') or (linestyle == 'v') or (linestyle == '.') or (linestyle == '*') or (linestyle == 'D'):
            custom_lines.append(plot_axis.scatter(1e9, 0, marker=linestyle, s=si, color=color))
        elif linestyle == '':
            custom_lines.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=0))
        else:
            custom_lines.append(Line2D([0], [0], color=color, linestyle=linestyle))

    plot_axis.set_xlim(xlim)
    plot_axis.set_ylim(ylim)
    if return_legend:
        legend = plot_axis.legend(custom_lines, labels, loc=loc, bbox_to_anchor=bbox_to_anchor, prop=prop, framealpha=framealpha)
        return legend
    else:
        plot_axis.legend(custom_lines, labels, loc=loc, bbox_to_anchor=bbox_to_anchor, prop=prop, framealpha=framealpha)


def transparent_to_opaque(color, alpha):
    color = np.array(matplotlib.colors.to_rgb(color))
    inv_color = 1 - color
    return 1 - (inv_color * alpha)


def set_working_directory():
    cwd = os.getcwd()
    if 'models-pytorch' not in cwd:
        os.chdir('models-pytorch')
