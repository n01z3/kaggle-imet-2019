import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch import cuda
from torch.utils.data import DataLoader
from tqdm import tqdm

from imet.n01_utils import (
    ThreadingDataLoader as DataLoader,
    ON_KAGGLE,
)
from imet.n02_models import se_resnext50, se_resnext101, effnet_b3, densenet161
from imet.n03_transforms import TTA2
from imet.n04_dataset import TTADataset, N_CLASSES, DATA_ROOT

MODELS = {"se_resnext101": se_resnext101, "se_resnext50": se_resnext50, 'eff_b3': effnet_b3, 'dn161': densenet161}


class TTAAveraging:
    def __init__(self, tta_size: int, num_classes: int):
        super().__init__()
        self._tta_size = tta_size
        self._num_classes = num_classes

        self._image_ids = []
        self._predicted_probas = None

    def add_predictions(self, image_ids, predictions):
        predictions = torch.sigmoid(predictions).data.cpu().numpy()
        predictions = predictions.reshape(
            len(image_ids), self._tta_size, self._num_classes
        )

        self._image_ids.extend(image_ids)

        predictions = np.mean(predictions, axis=1)
        if self._predicted_probas is None:
            self._predicted_probas = predictions.copy()
        else:
            self._predicted_probas = np.append(
                self._predicted_probas, predictions.copy(), axis=0
            )

    def build_predictions_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(data=self._predicted_probas)
        df["id"] = self._image_ids
        return df


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--model", default="eff_b3")
    arg("--checkpoint", default="../weights/eff_b3.pth")
    arg("--crop-size", type=int, default=640)
    arg("--scale-size", type=int, default=320)

    arg("--batch-size", type=int, default=16)
    arg("--step", type=int, default=1)
    arg("--workers", type=int, default=2 if ON_KAGGLE else 4)
    arg("--clean", action="store_true")
    arg("--tta", type=int, default=2)
    arg("--debug", action="store_true")
    arg("--limit", type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    model = MODELS[args.model]()
    model.load_state_dict(torch.load(args.checkpoint))

    use_cuda = cuda.is_available()
    if use_cuda:
        model = model.cuda()

    model.eval()

    dataset = TTADataset(
        DATA_ROOT, TTA2(args.crop_size, args.crop_size, args.scale_size)
    )
    tloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    tta_averaging = TTAAveraging(tta_size=args.tta, num_classes=N_CLASSES)

    with torch.set_grad_enabled(False):
        for data in tqdm(tloader, total=len(tloader)):
            images = data["image"]
            _, _, c, h, w = images.size()
            images = images.view(-1, c, h, w)
            images = images.cuda(non_blocking=True)

            predictions = model(images)
            tta_averaging.add_predictions(data["id"], predictions)

    df = tta_averaging.build_predictions_dataframe()
    os.makedirs(args.model, exist_ok=True)

    df.to_hdf(os.path.join(os.path.basename(args.checkpoint).split('.')[0], f"test.h5"), "prob", index_label="id")
    print(f"Saved predictions for {args.model}")


if __name__ == "__main__":
    main()
