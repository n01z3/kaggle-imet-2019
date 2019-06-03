import argparse

import pandas as pd
import numpy as np

from imet.n04_dataset import DATA_ROOT, N_CLASSES


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--predictions', type=str, default='se_resnext50/test.h5')
    arg('--threshold', type=float, default=0.13)
    arg('--output', type=str, default='submission.csv')
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_hdf(args.predictions, 'prob')
    print(df.head())

    arr = df.drop('id', axis=1).values

    attrs = []
    for i in range(arr.shape[0]):
        index = np.where(arr[i] > args.threshold)[0]
        attrs.append(' '.join(index.astype(str)))

    print(attrs[:10])

    subm = pd.DataFrame()
    subm['id'] = df['id']
    subm['attribute_ids'] = attrs
    subm.to_csv(args.output, index=False)
    print(subm.shape)


if __name__ == '__main__':
    main()
