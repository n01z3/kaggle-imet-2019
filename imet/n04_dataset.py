import os
from glob import glob
from typing import Callable

import cv2

from .n01_utils import ON_KAGGLE

N_CLASSES = 1103
DATA_ROOT = "../input/imet-2019-fgvc6/test" if ON_KAGGLE else "/media/n01z3/fast/dataset/imet/test"


class TTADataset:
    def __init__(self, root: str, image_transform: Callable):
        self._paths = sorted(glob(os.path.join(root, "*.png")))
        self._image_transform = image_transform

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx):
        path = self._paths[idx]
        image_id = os.path.basename(path).replace(".png", "")

        image = cv2.imread(path)[:, :, ::-1]
        image = self._image_transform(image=image)["image"]
        return {"id": image_id, "image": image}
