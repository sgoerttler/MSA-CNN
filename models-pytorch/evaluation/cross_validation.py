import numpy as np


def custom_cv(idx_fold, k):
    """Get custom cross-validation iterator."""
    if k == 1:
        # return all data for training and testing, no cross-validation
        yield 0, \
              np.argwhere(np.array(idx_fold) != -1).flatten(), \
              np.argwhere(np.array(idx_fold) != -1).flatten()
    else:
        # cross-validation
        for k_i in range(k):
            yield k_i, \
                  np.argwhere(np.array(idx_fold) != k_i).flatten(), \
                  np.argwhere(np.array(idx_fold) == k_i).flatten()
