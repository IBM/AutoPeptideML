import numpy as np

from sklearn.preprocessing import KBinsDiscretizer


def discretizer(labels: np.ndarray, n_bins: int = 5) -> np.ndarray:
    if labels is None:
        return None
    elif len(np.unique(labels)) > 0.5 * len(labels):
        if len(labels.shape) < 2:
            labels = labels.reshape(-1, 1)
        try:
            disc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal',
                                    quantile_method='linear')
        except TypeError:
            disc = KBinsDiscretizer(n_bins=n_bins, encode='ordinal')
        labels = disc.fit_transform(labels)
        return labels
    else:
        return labels
