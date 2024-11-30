# coding: utf-8
# Copyright (C) 2022-2023, [Breezedeus](https://github.com/breezedeus).
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
# Credits: adapted from https://github.com/PaddlePaddle/PaddleOCR

import os
import logging
from typing import Union, Optional, Collection, List, Tuple
from pathlib import Path
import math

import numpy as np
from PIL import Image
from cnstd.utils import get_model_file

from ..utils import resize_img, data_dir, read_img
from ..recognizer import Recognizer
from .postprocess import build_post_process
from .utility import create_predictor
from .consts import PP_SPACE
from ..consts import MODEL_VERSION, AVAILABLE_MODELS, DOWNLOAD_SOURCE


logger = logging.getLogger(__name__)


class PPRecognizer(Recognizer):
    def __init__(
        self,
        model_name: str = 'ch_PP-OCRv3',
        *,
        cand_alphabet: Optional[Union[Collection, str]] = None,
        model_fp: Optional[str] = None,
        root: Union[str, Path] = data_dir(),
        rec_image_shape: str = "3, 32, 320",
        use_space_char: bool = True,
        **kwargs
    ):
        """
        来自 ppocr 的文本识别器。

        Args:
            model_name (str): 模型名称。默认为 `ch_PP-OCRv3`
            cand_alphabet (Optional[Union[Collection, str]]): 待识别字符所在的候选集合。默认为 `None`，表示不限定识别字符范围
            model_fp (Optional[str]): 如果不使用系统自带的模型，可以通过此参数直接指定所使用的模型文件（'.ckpt' 文件）
            root (Union[str, Path]): 模型文件所在的根目录
                Linux/Mac下默认值为 `~/.cnocr`，表示模型文件所处文件夹类似 `~/.cnocr/2.3/ppocr`
                Windows下默认值为 `C:/Users/<username>/AppData/Roaming/cnocr/2.3/ppocr`
            rec_image_shape (str): 输入图片尺寸，无需更改使用默认值即可。默认值：`"3, 32, 320"`
            use_space_char (bool): 是否使用空格字符，无需更改使用默认值即可。默认值：`True`
            **kwargs:
                ort_providers (List[str]): 使用 ONNX 模型时，使用此参数指定 `onnxruntime` 识别模型运行的设备。未指定则使用默认值（优先使用 GPU）。
        """
        self.rec_image_shape = [int(v) for v in rec_image_shape.split(",")]
        self.rec_algorithm = 'CRNN'
        self._model_name = model_name
        self._model_backend = 'onnx'

        vocab_fp = AVAILABLE_MODELS.get_vocab_fp(self._model_name, self._model_backend)
        self._assert_and_prepare_model_files(model_fp, root)
        logger.info('use model: %s' % self._model_fp)
        postprocess_params = {
            'name': 'CTCLabelDecode',
            'character_dict_path': vocab_fp,
            'use_space_char': use_space_char,
            'cand_alphabet': cand_alphabet,
        }
        self.postprocess_op = build_post_process(postprocess_params)
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            self.config,
        ) = create_predictor(
            self._model_fp, 'rec', ort_providers=kwargs.get('ort_providers')
        )
        self.use_onnx = True

    def _assert_and_prepare_model_files(self, model_fp, root):
        if model_fp is not None and not os.path.isfile(model_fp):
            raise FileNotFoundError('can not find model file %s' % model_fp)

        if model_fp is not None:
            self._model_fp = model_fp
            return

        root = os.path.join(root, MODEL_VERSION)
        self._model_dir = os.path.join(root, PP_SPACE)
        model_fp = os.path.join(self._model_dir, '%s_rec_infer.onnx' % self._model_name)
        if not os.path.isfile(model_fp):
            logger.warning('can not find model file %s' % model_fp)
            if (self._model_name, self._model_backend) not in AVAILABLE_MODELS:
                raise NotImplementedError(
                    '%s is not a downloadable model'
                    % ((self._model_name, self._model_backend),)
                )
            url = AVAILABLE_MODELS.get_url(self._model_name, self._model_backend)
            get_model_file(
                url, self._model_dir, download_source=DOWNLOAD_SOURCE
            )  # download the .zip file and unzip

        self._model_fp = model_fp

    def resize_norm_img(self, img, max_wh_ratio):
        """

        Args:
            img (): with shape of (height, width, channel)
            max_wh_ratio ():

        Returns:

        """
        imgC, imgH, imgW = self.rec_image_shape

        assert imgC == img.shape[2]
        imgW = int((32 * max_wh_ratio))
        if self.use_onnx:
            w = self.input_tensor.shape[3:][0]
            if w is not None and w > 0:
                imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        # resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resize_img(
            img.transpose((2, 0, 1)), target_h_w=(imgH, resized_w), return_torch=False
        ).transpose((1, 2, 0))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def recognize(
        self, img_list: List[Union[str, Path, np.ndarray]], batch_size: int = 1,
    ) -> List[Tuple[str, float]]:
        """
        Batch recognize characters from a list of one-line-characters images.

        Args:
            img_list (List[Union[str, Path, np.ndarray]]):
                list of images, in which each element should be a line image array with np.ndarray.
                Each element should be a tensor with values ranging from 0 to 255,
                and with shape [height, width] or [height, width, channel].
                The optional channel should be 1 (gray image) or 3 (RGB-format color image).
                注：img_list 不宜包含太多图片，否则同时导入这些图片会消耗很多内存。
            batch_size: 待处理图片很多时，需要分批处理，每批图片的数量由此参数指定。默认为 `1`。

        Returns:
            list: list of (chars, prob), such as
            [('第一行', 0.80), ('第二行', 0.75), ('第三行', 0.9)]
        """
        if len(img_list) == 0:
            return []

        img_list = [self._prepare_img(img) for img in img_list]

        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        for beg_img_no in range(0, img_num, batch_size):
            end_img_no = min(img_num, beg_img_no + batch_size)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                if self.rec_algorithm != "SRN" and self.rec_algorithm != "SAR":
                    norm_img = self.resize_norm_img(
                        img_list[indices[ino]], max_wh_ratio
                    )
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            input_dict = dict()
            input_dict[self.input_tensor.name] = norm_img_batch
            outputs = self.predictor.run(self.output_tensors, input_dict)
            preds = outputs[0]

            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        return rec_res

    def _prepare_img(self, img_fp: Union[str, Path, np.ndarray]) -> np.ndarray:
        """

        Args:
            img_fp (Union[str, Path, np.ndarray]):
                image array with type torch.Tensor or np.ndarray,
                with shape [height, width] or [height, width, channel].
                channel should be 1 (gray image) or 3 (color image).

        Returns:
            np.ndarray: with shape (height, width, 3), scale [0, 255]
        """
        img = img_fp
        if isinstance(img_fp, (str, Path)):
            if not os.path.isfile(img_fp):
                raise FileNotFoundError(img_fp)
            img = read_img(img_fp, gray=False)

        if len(img.shape) == 3 and img.shape[2] == 1:
            # (H, W, 1) -> (H, W)
            img = img.squeeze(-1)
        if len(img.shape) == 2:
            # (H, W) -> (H, W, 3)
            img = np.array(Image.fromarray(img).convert('RGB'))
        elif img.shape[2] != 3:
            raise ValueError(
                'only images with shape [height, width, 1] (gray images), '
                'or [height, width, 3] (RGB-formated color images) are supported'
            )

        return img
