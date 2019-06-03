import argparse

import pandas as pd
import numpy as np


from .n01_utils import mean_df
from .n04_dataset import DATA_ROOT


def binarize_prediction(probabilities, threshold: float, argsorted=None,
                        min_labels=1, max_labels=10):
    """ Return matrix of 0/1 predictions, same shape as probabilities.
    """
    assert probabilities.shape[1] == N_CLASSES
    if argsorted is None:
        argsorted = probabilities.argsort(axis=1)
    max_mask = _make_mask(argsorted, max_labels)
    min_mask = _make_mask(argsorted, min_labels)
    prob_mask = probabilities > threshold
    return (max_mask & prob_mask) | min_mask


def _make_mask(argsorted, top_n: int):
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = [i // top_n for i in range(len(col_indices))]
    mask[row_indices, col_indices] = 1
    return mask


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('predictions', nargs='+')
    arg('output')
    arg('--threshold', type=float, default=0.2)
    args = parser.parse_args()
    sample_submission = pd.read_csv(
        DATA_ROOT / 'sample_submission.csv', index_col='id')
    dfs = []
    for prediction in args.predictions:
        df = pd.read_hdf(prediction, index_col='id')
        df = df.reindex(sample_submission.index)
        dfs.append(df)
    df = pd.concat(dfs)
    df = mean_df(df)
    df[:] = binarize_prediction(df.values, threshold=args.threshold)
    df = df.apply(get_classes, axis=1)
    df.name = 'attribute_ids'
    df.to_csv(args.output, header=True)


def get_classes(item):
    return ' '.join(cls for cls, is_present in item.items() if is_present)


if __name__ == '__main__':
    main()
