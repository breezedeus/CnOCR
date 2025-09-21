# coding: utf-8
# Copyright (C) 2022-2025, [Breezedeus](https://github.com/breezedeus).
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.

import os
import logging
from typing import Union, Optional, List, Tuple
from pathlib import Path

import numpy as np
from rapidocr import EngineType, LangRec, ModelType, OCRVersion
from rapidocr.utils.typings import TaskType
from rapidocr.ch_ppocr_rec import TextRecognizer, TextRecInput
from cnstd.utils import prepare_model_files

from ..utils import data_dir, read_img
from ..recognizer import Recognizer
from .consts import PP_SPACE
from ..consts import MODEL_VERSION, AVAILABLE_MODELS


logger = logging.getLogger(__name__)


class Config(dict):
    DEFAULT_CFG = {
        "engine_type": EngineType.ONNXRUNTIME,
        "lang_type": LangRec.CH,
        "model_type": ModelType.MOBILE,
        "ocr_version": OCRVersion.PPOCRV5,
        "task_type": TaskType.REC,
        "model_path": None,
        "model_dir": None,
        "rec_keys_path": None,
        "rec_img_shape": [3, 48, 320],
        "rec_batch_num": 6,
        "engine_cfg": {
            "intra_op_num_threads": -1,
            "inter_op_num_threads": -1,
            "enable_cpu_mem_arena": False,
            "cpu_ep_cfg": {"arena_extend_strategy": "kSameAsRequested"},
            "use_cuda": False,
            "cuda_ep_cfg": {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
            "use_dml": False,
            "dm_ep_cfg": None,
            "use_cann": False,
            "cann_ep_cfg": {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "npu_mem_limit": 21474836480,
                "op_select_impl_mode": "high_performance",
                "optypelist_for_implmode": "Gelu",
                "enable_cann_graph": True,
            },
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__()
        data = dict(*args, **kwargs)
        for k, v in data.items():
            if isinstance(v, dict):
                v = Config(v)
            self[k] = v

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class RapidRecognizer(Recognizer):
    def __init__(
        self,
        model_name: str = "ch_PP-OCRv5",
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
            model_name (str): 模型名称。默认为 `ch_PP-OCRv5`
            model_fp (Optional[str]): 如果不使用系统自带的模型，可以通过此参数直接指定所使用的模型文件（'.onnx' 文件）
            root (Union[str, Path]): 模型文件所在的根目录
            context (str): 使用的设备。默认为 `cpu`，可选 `gpu`
            rec_image_shape (str): 输入图片尺寸，无需更改使用默认值即可。默认值：`"3, 48, 320"`
            **kwargs: 其他参数
        """
        self.rec_image_shape = [int(v) for v in rec_image_shape.split(",")]
        self._model_name = model_name
        self._model_backend = "onnx"
        use_gpu = context.lower() not in ("cpu", "mps")

        self._assert_and_prepare_model_files(model_fp, root)

        config = Config.DEFAULT_CFG
        ## add custom font path
        if 'font_path' in kwargs:
            config['font_path'] = kwargs['font_path']
        config["engine_cfg"]["use_cuda"] = use_gpu
        if "engine_cfg" in kwargs:
            config["engine_cfg"].update(kwargs["engine_cfg"])
        config["rec_img_shape"] = self.rec_image_shape
        config["model_path"] = self._model_fp
        # 从 model_name 中获取 model_type 和 ocr_version
        config["model_type"] = ModelType.SERVER if "server" in model_name else ModelType.MOBILE
        config["ocr_version"] = OCRVersion.PPOCRV5 if "v5" in model_name else OCRVersion.PPOCRV4

        config = Config(config)
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
        self,
        img_list: List[Union[str, Path, np.ndarray]],
        batch_size: int = 6,
        return_word_box: bool = False,
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

        rec_input = TextRecInput(img=img_data_list, return_word_box=return_word_box)
        try:
            results = self.recognizer(rec_input)
            return [(txt, score) for txt, score in zip(results.txts, results.scores)]
        except Exception as e:
            logger.error(f"Error recognizing image: {e}")
            return []

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
