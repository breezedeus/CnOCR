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
#
# use datasets from https://github.com/huggingface/datasets

from datasets import Dataset, Image
import numpy as np
import torch
import os
from pathlib import Path

from .consts import IMG_STANDARD_HEIGHT
from .utils import read_tsv_file, pad_img_seq


def preprocess(img):
    # img = np.expand_dims(np.array(img.convert('L')), 0)  # -> [1, H, W]
    # return resize_img(img, return_torch=False)
    # NOTE: MUST return Image.Image instead of np.ndarray or torch.Tensor
    img = img.convert('L')
    ori_width, ori_height = img.size

    min_width = 8
    ratio = ori_height / IMG_STANDARD_HEIGHT
    target_w = max(int(ori_width / ratio), min_width)
    target_w_h = (target_w, IMG_STANDARD_HEIGHT)
    return img.resize(target_w_h)


def apply_transforms(img, transforms):
    """Apply transforms to a single image."""
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, 0)
    return transforms(torch.from_numpy(img))


def create_transform_func(transforms):
    """Create a transform function that can be pickled."""
    def transform_func(examples):
        outs = []
        for img in examples['image']:
            outs.append(apply_transforms(img, transforms))
        examples['transformed_image'] = outs
        return examples
    return transform_func


def gen_dataset(
    index_fp, img_folder=None, transforms=None, mode='train', num_workers=None
) -> Dataset:
    """
    Generates a dataset based on the provided index file path.

    Args:
        index_fp (str | Path): The file path to the index file.
        img_folder (str, optional): The folder path for the images. Defaults to None.
        transforms (callable, optional): The transforms to apply to the images. Defaults to None.
        mode (str, optional): The mode of the dataset (train, val, test, etc.). Defaults to 'train'.
        num_workers (int, optional): The number of workers for data loading. Defaults to None.

    Returns:
        Dataset: The generated dataset.
    """
    img_fp_list, labels_list = read_tsv_file(index_fp, '\t', img_folder, mode)

    if mode != 'test':
        # 根据 labels 的长度进行排序
        sorted_indices = sorted(
            range(len(labels_list)), key=lambda x: len(labels_list[x])
        )
        img_fp_list = [img_fp_list[i] for i in sorted_indices]
        labels_list = [labels_list[i] for i in sorted_indices]

    dataset = Dataset.from_dict(
        {'image': img_fp_list, 'labels': labels_list}
    ).cast_column("image", Image())

    def map_func(examples):
        examples['image'] = [preprocess(img) for img in examples['image']]
        return examples

    if num_workers <= 0:
        num_workers = None
    dataset = dataset.map(map_func, batched=True, num_proc=num_workers)

    if transforms is not None:
        dataset.set_transform(create_transform_func(transforms))
    return dataset


def collate_fn(examples):
    img_list = []
    labels_list = []
    for example in examples:
        img_list.append((example["transformed_image"]))
        labels_list.append(example["labels"])

    label_lengths = torch.tensor([len(labels) for labels in labels_list])
    img_lengths = torch.tensor([img.size(2) for img in img_list])
    imgs = pad_img_seq(img_list)
    return imgs, img_lengths, labels_list, label_lengths
