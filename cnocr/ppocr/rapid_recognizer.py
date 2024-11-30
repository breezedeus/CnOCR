# coding: utf-8
# Copyright (C) 2022-2024, [Breezedeus](https://github.com/breezedeus).
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.

import os
import logging
from typing import Union, Optional, List, Tuple
from pathlib import Path

import numpy as np
from rapidocr_onnxruntime.ch_ppocr_rec.text_recognize import TextRecognizer
from cnstd.utils import prepare_model_files

from ..utils import data_dir, read_img
from ..recognizer import Recognizer
from .consts import PP_SPACE
from ..consts import MODEL_VERSION, AVAILABLE_MODELS


logger = logging.getLogger(__name__)


class RapidRecognizer(Recognizer):
    def __init__(
        self,
        model_name: str = "ch_PP-OCRv3",
        *,
        model_fp: Optional[str] = None,
        root: Union[str, Path] = data_dir(),
        context: str = "cpu",  # ['cpu', 'gpu']
        rec_image_shape: str = "3, 48, 320",
        **kwargs
    ):
        """
        基于 rapidocr_onnxruntime 的文本识别器。

        Args:
            model_name (str): 模型名称。默认为 `ch_PP-OCRv3`
            model_fp (Optional[str]): 如果不使用系统自带的模型，可以通过此参数直接指定所使用的模型文件（'.onnx' 文件）
            root (Union[str, Path]): 模型文件所在的根目录
            context (str): 使用的设备。默认为 `cpu`，可选 `gpu`
            rec_image_shape (str): 输入图片尺寸，无需更改使用默认值即可。默认值：`"3, 32, 320"`
            **kwargs: 其他参数
        """
        self.rec_image_shape = [int(v) for v in rec_image_shape.split(",")]
        self._model_name = model_name
        self._model_backend = "onnx"
        use_gpu = context.lower() not in ("cpu", "mps")

        self._assert_and_prepare_model_files(model_fp, root)

        config = {
            "use_cuda": use_gpu,
            "rec_img_shape": self.rec_image_shape,
            "rec_batch_num": 6,
            "model_path": self._model_fp,
        }
        self.recognizer = TextRecognizer(config)

    def _assert_and_prepare_model_files(self, model_fp, root):
        if model_fp is not None and not os.path.isfile(model_fp):
            raise FileNotFoundError("can not find model file %s" % model_fp)

        if model_fp is not None:
            self._model_fp = model_fp
            return

        root = os.path.join(root, MODEL_VERSION)
        self._model_dir = os.path.join(root, PP_SPACE, self._model_name)
        model_fp = os.path.join(self._model_dir, "%s_rec_infer.onnx" % self._model_name)
        if not os.path.isfile(model_fp):
            logger.warning("can not find model file %s" % model_fp)
            if (self._model_name, self._model_backend) not in AVAILABLE_MODELS:
                raise NotImplementedError(
                    "%s is not a downloadable model"
                    % ((self._model_name, self._model_backend),)
                )
            remote_repo = AVAILABLE_MODELS.get_value(
                self._model_name, self._model_backend, "repo"
            )
            model_fp = prepare_model_files(model_fp, remote_repo)

        self._model_fp = model_fp
        logger.info("use model: %s" % self._model_fp)

    def recognize(
        self, img_list: List[Union[str, Path, np.ndarray]], batch_size: int = 6
    ) -> List[Tuple[str, float]]:
        """
        识别图片中的文字。
        Args:
            img_list: 支持以下格式的图片数据：
                + 图片路径
                + 已经从图片文件中读入的数据
            batch_size: 待处理图片数据的批大小。

        Returns:
            列表，每个元素是对应图片的识别结果，由 (text, score) 组成，其中：
                + text: 识别出的文本
                + score: 识别结果的得分
        """
        if not isinstance(img_list, (list, tuple)):
            img_list = [img_list]

        self.recognizer.rec_batch_num = batch_size

        img_data_list = []
        for img in img_list:
            if isinstance(img, (str, Path)):
                img = read_img(img, gray=False)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = img[..., ::-1]  # RGB to BGR
            img_data_list.append(img)

        results, _ = self.recognizer(img_data_list)
        return results

    def recognize_one_line(
        self, img: Union[str, Path, np.ndarray]
    ) -> Tuple[str, float]:
        """
        识别图片中的一行文字。
        Args:
            img: 支持以下格式的图片数据：
                + 图片路径
                + 已经从图片文件中读入的数据

        Returns:
            (text, score)：
                + text: 识别出的文本
                + score: 识别结果的得分
        """
        results = self.recognize([img])
        return results[0]
