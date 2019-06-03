from typing import Dict

import cv2
import numpy as np
import torch
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
)


class CenterCrop:
    def __init__(self, height, width):
        self._height = height
        self._width = width

    def __call__(self, image):
        h, w, _ = image.shape
        cropped_image = self._center_crop(
            image, min(self._height, h), min(self._width, w)
        )
        return cropped_image

    @staticmethod
    def _center_crop(image: np.ndarray, height, width):
        h, w, _ = image.shape
        assert h >= height
        assert w >= width
        i = (h - height) // 2
        j = (w - width) // 2
        return image[i : i + height, j : j + width]


class Resize:
    def __init__(self, height, width):
        self._height = height
        self._width = width

    def __call__(self, image):
        return cv2.resize(
            image, (self._width, self._height), interpolation=cv2.INTER_LINEAR
        )


def post_transform():
    return Compose(
        [
            ToTensor(),
            Normalize(
                mean=np.array([0.485, 0.456, 0.406]),
                std=np.array([0.229, 0.224, 0.225]),
            ),
        ]
    )


class TTA2:
    def __init__(self, crop_height=640, crop_width=640, scale_size=320):
        self._crop = CenterCrop(height=crop_height, width=crop_width)
        self._resize = Resize(height=scale_size, width=scale_size)
        self._final = post_transform()

    def __call__(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        image1 = self._resize(self._crop(image.copy()))
        image2 = self._resize(self._crop(cv2.flip(image.copy(), 1)))
        return {"image": torch.stack((self._final(image1), self._final(image2)))}


train_transform = Compose([RandomCrop(288), RandomHorizontalFlip()])

test_transform = Compose([RandomCrop(288), RandomHorizontalFlip()])

tensor_transform = Compose(
    [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
