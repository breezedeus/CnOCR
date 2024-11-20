# coding: utf-8
# Copyright (C) 2021-2023, [Breezedeus](https://github.com/breezedeus).
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import random
import logging
import traceback

import cv2
import torch
import numpy as np

try:
    import albumentations as alb
    from albumentations.pytorch import ToTensorV2
    from albumentations.core.transforms_interface import ImageOnlyTransform
except ImportError:
    raise ImportError(f'Please install the dev version of cnocr by: `pip install cnocr[dev]`')

from ..utils import normalize_img_array

logger = logging.getLogger(__name__)


class Erosion(ImageOnlyTransform):
    """
    From Nougat: https://github.com/facebookresearch/nougat/blob/main/nougat/transforms.py .
    Apply erosion operation to an image.

    Erosion is a morphological operation that shrinks the white regions in a binary image.

    Args:
        scale (int or tuple/list of int): The scale or range for the size of the erosion kernel.
            If an integer is provided, a square kernel of that size will be used.
            If a tuple or list is provided, it should contain two integers representing the minimum
            and maximum sizes for the erosion kernel.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, scale, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
        )
        img = cv2.erode(img, kernel, iterations=1)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        return img


class Dilation(ImageOnlyTransform):
    """
    From Nougat: https://github.com/facebookresearch/nougat/blob/main/nougat/transforms.py .
    Apply dilation operation to an image.

    Dilation is a morphological operation that expands the white regions in a binary image.

    Args:
        scale (int or tuple/list of int): The scale or range for the size of the dilation kernel.
            If an integer is provided, a square kernel of that size will be used.
            If a tuple or list is provided, it should contain two integers representing the minimum
            and maximum sizes for the dilation kernel.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, scale, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
        )
        img = cv2.dilate(img, kernel, iterations=1)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        return img


class Bitmap(ImageOnlyTransform):
    """
    From Nougat: https://github.com/facebookresearch/nougat/blob/main/nougat/transforms.py .
    Apply a bitmap-style transformation to an image.

    This transformation replaces all pixel values below a certain threshold with a specified value.

    Args:
        value (int, optional): The value to replace pixels below the threshold with. Default is 0.
        lower (int, optional): The threshold value below which pixels will be replaced. Default is 200.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, value=0, lower=200, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.lower = lower
        self.value = value

    def apply(self, img, **params):
        img = img.copy()
        img[img < self.lower] = self.value
        return img


class RandomStretchAug(ImageOnlyTransform):
    """保持高度不变的情况下，对图像的宽度进行随机拉伸"""

    def __init__(
            self, min_ratio=0.9, max_ratio=1.1, min_width=8, always_apply=False, p=1
        ):
        super().__init__(always_apply=always_apply, p=p)
        self.min_width = min_width
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def apply(self, img, **params):
        h, w = img.shape[:2]
        new_w_ratio = self.min_ratio + random.random() * (
            self.max_ratio - self.min_ratio
        )
        new_w = max(int(w * new_w_ratio), self.min_width)
        return alb.Resize(height=h, width=new_w).apply(img)


class CustomRandomCrop(ImageOnlyTransform):
    """从图像的四个边缘随机裁剪"""

    def __init__(self, crop_size, always_apply=False, p=1.0):
        """
        Initializes a new instance of the CustomRandomCrop class.

        Parameters:
            crop_size (tuple): The size of the crop, in pixels, (height, width).
            always_apply (bool): Whether to always apply the crop. Defaults to False.
            p (float): The probability of applying the crop. Defaults to 1.0.
        """
        super().__init__(always_apply=always_apply, p=p)
        self.crop_size = crop_size

    def cal_params(self, img):
        ori_h, ori_w = img.shape[:2]
        for _ in range(10):
            h_top, h_bot = (
                random.randint(0, self.crop_size[0]),
                random.randint(0, self.crop_size[0]),
            )
            w_left, w_right = (
                random.randint(0, self.crop_size[1]),
                random.randint(0, self.crop_size[1]),
            )
            h = ori_h - h_top - h_bot
            w = ori_w - w_left - w_right
            if h < max(ori_h * 0.5, 4) or w < max(ori_w * 0.5, 4):
                continue

            return h_top, w_left, h, w

        return 0, 0, ori_h, ori_w

    def apply(self, img, **params):
        h_top, w_left, h, w = self.cal_params(img)
        out = cv2.resize(
            img[h_top : h_top + h, w_left : w_left + w], img.shape[:2][::-1]
        )
        if img.ndim > out.ndim:
            out = np.expand_dims(out, axis=-1)
        return out


class TransparentOverlay(ImageOnlyTransform):
    """模仿标注笔的标注效果。"""

    def __init__(
        self, max_height_ratio, max_width_ratio, alpha, always_apply=False, p=1.0
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.max_height_ratio = max_height_ratio
        self.max_width_ratio = max_width_ratio
        self.alpha = alpha

    def apply(self, img, x=0, y=0, height=0, width=0, color=(0, 0, 0), **params):
        if min(height, width) < 2:
            return img
        original_c = img.shape[2]

        # 确保图片有四个通道（RGBA）
        if img.shape[2] < 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        # 创建一个与图片大小相同的覆盖层
        overlay = img.copy()

        # 在覆盖层上涂色
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)

        # 结合覆盖层和原图片
        img = cv2.addWeighted(overlay, self.alpha, img, 1 - self.alpha, 0)

        # Convert the image back to the original number of channels
        if original_c != img.shape[2]:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img

    @property
    def targets_as_params(self):
        return ['image']

    def get_params_dependent_on_targets(self, params):
        img = params['image']
        height, width, _ = img.shape

        # Compute the actual pixel values for the maximum height and width
        max_height = int(height * self.max_height_ratio)
        max_width = int(width * self.max_width_ratio)

        x = np.random.randint(0, max(width - max_width, 1))
        y = np.random.randint(0, max(height - max_height, 1))
        rect_width = np.random.randint(0, max(max_width, 1))
        rect_height = np.random.randint(0, max(max_height, 1))

        color = [np.random.randint(0, 256) for _ in range(3)]

        return {
            'x': x,
            'y': y,
            'width': rect_width,
            'height': rect_height,
            'color': color,
        }


class ToSingleChannelGray(ImageOnlyTransform):
    def apply(self, img, **params):  # -> [H, W, 1]
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        if img.shape[2] != 1:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return gray[:, :, np.newaxis]  # Add an extra channel dimension
        else:
            return img


class CustomNormalize(ImageOnlyTransform):
    def apply(self, img, **params):  # -> [H, W, 1]
        return normalize_img_array(img)


class TransformWrapper(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, ori_image: torch.Tensor) -> torch.Tensor:
        """
        把albumentations的transform转换成torchvision的transform。

        Args:
            image (torch.Tensor): with shape [C, H, W]

        Returns: torch.Tensor, with shape [C, H, W]

        """
        image = ori_image.numpy()
        image = image.transpose((1, 2, 0))  # to: [H, W, C]
        try:
            out = self.transform(image=image)['image']
        except Exception as e:
            logger.error(f"Error when transforming one image with shape: {image.shape}")
            # print call stacktrace
            traceback.print_exc()
            logger.error(e)
            return ori_image.to(torch.float32)

        if image.ndim > out.ndim:
            out = np.expand_dims(out, axis=-1)
        out = torch.from_numpy(out.transpose((2, 0, 1)))  # to: [C, H, W]
        return out


_train_alb_transform = alb.Compose(
    [
        CustomRandomCrop(crop_size=(8, 10), always_apply=False, p=0.8),
        alb.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.1),
        TransparentOverlay(1.0, 0.1, alpha=0.4, always_apply=False, p=0.2),  # 半透明的矩形框覆盖
        alb.Affine(shear={"x": (0, 3), "y": (-3, 0)}, cval=(255, 255, 255), p=0.03),
        alb.ShiftScaleRotate(
            shift_limit_x=(0, 0.04),
            shift_limit_y=(0, 0.03),
            scale_limit=(-0.15, 0.03),
            rotate_limit=2,
            border_mode=0,
            interpolation=2,
            value=(255, 255, 255),
            p=0.03,
        ),
        alb.GridDistortion(
            distort_limit=0.05,
            border_mode=0,
            interpolation=2,
            value=(255, 255, 255),
            p=0.04,
        ),
        alb.Compose(
            [
                alb.Affine(
                    translate_px=(0, 2), always_apply=True, cval=(255, 255, 255)
                ),
                alb.ElasticTransform(
                    p=1,
                    alpha=50,
                    sigma=120 * 0.1,
                    alpha_affine=0.1,  #120 * 0.01,
                    border_mode=0,
                    value=(255, 255, 255),
                ),
            ],
            p=0.1,
        ),
        alb.RandomBrightnessContrast(0.1, 0.1, True, p=0.1),
        alb.ImageCompression(95, p=0.3),
        alb.GaussNoise(20, p=0.2),
        alb.GaussianBlur((3, 3), p=0.1),
        alb.Emboss(p=0.3, alpha=(0.2, 0.5), strength=(0.2, 0.7)),
        alb.OpticalDistortion(
            always_apply=False,
            p=0.2,
            distort_limit=(-0.05, 0.05),
            shift_limit=(-0.05, 0.05),
            interpolation=0,
            border_mode=0,
            value=(0, 0, 0),
            mask_value=None,
        ),
        # alb.Sharpen(always_apply=False, p=0.3, alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
        RandomStretchAug(min_ratio=0.5, max_ratio=1.5, p=0.2, always_apply=False),
        alb.InvertImg(p=0.3),
        ToSingleChannelGray(always_apply=True),
        CustomNormalize(always_apply=True),
        # alb.Normalize(0.456045, 0.224567, always_apply=True),
        # ToTensorV2(),
    ]
)

train_transform = TransformWrapper(_train_alb_transform)

_ft_alb_transform = alb.Compose(
    [
        CustomRandomCrop(crop_size=(4, 4), always_apply=False, p=0.8),
        alb.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.1),
        TransparentOverlay(1.0, 0.1, alpha=0.4, always_apply=False, p=0.2),  # 半透明的矩形框覆盖
        alb.RandomBrightnessContrast(0.1, 0.1, True, p=0.1),
        alb.ImageCompression(95, p=0.3),
        alb.GaussNoise(20, p=0.2),
        alb.GaussianBlur((3, 3), p=0.1),
        alb.Emboss(p=0.3, alpha=(0.2, 0.5), strength=(0.2, 0.7)),
        alb.OpticalDistortion(
            always_apply=False,
            p=0.2,
            distort_limit=(-0.05, 0.05),
            shift_limit=(-0.05, 0.05),
            interpolation=0,
            border_mode=0,
            value=(0, 0, 0),
            mask_value=None,
        ),
        RandomStretchAug(min_ratio=0.8, max_ratio=1.2, p=0.2, always_apply=False),
        alb.InvertImg(p=0.3),
        ToSingleChannelGray(always_apply=True),
        CustomNormalize(always_apply=True),
        # alb.Normalize(0.456045, 0.224567, always_apply=True),
        # ToTensorV2(),
    ]
)

ft_transform = TransformWrapper(_ft_alb_transform)

_test_alb_transform = alb.Compose(
    [
        CustomRandomCrop(crop_size=(6, 8), p=0.8),
        ToSingleChannelGray(always_apply=True),
        CustomNormalize(always_apply=True),
        # alb.Normalize(0.456045, 0.224567, always_apply=True),
    ]
)

test_transform = TransformWrapper(_test_alb_transform)
