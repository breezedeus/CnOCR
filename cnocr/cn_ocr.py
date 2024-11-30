# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).
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

import os
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Union, List, Any, Dict, Optional, Collection
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from cnstd.consts import AVAILABLE_MODELS as DET_AVAILABLE_MODELS
from cnstd import CnStd
from cnstd.utils import data_dir as det_data_dir

from .consts import AVAILABLE_MODELS as REC_AVAILABLE_MODELS
from .utils import data_dir, read_img
from .line_split import line_split
from .recognizer import Recognizer
from .ppocr import PPRecognizer, RapidRecognizer, PP_SPACE

logger = logging.getLogger(__name__)

DET_MODLE_NAMES, _ = zip(*DET_AVAILABLE_MODELS.all_models())
DET_MODLE_NAMES = set(DET_MODLE_NAMES)


@dataclass
class OcrResult(object):
    text: str
    score: float
    position: Optional[np.ndarray] = None
    cropped_img: np.ndarray = None

    def to_dict(self):
        res = deepcopy(self.__dict__)
        if self.position is None:
            res.pop('position')
        if self.cropped_img is None:
            res.pop('cropped_img')
        return res


class CnOcr(object):
    def __init__(
        self,
        rec_model_name: str = 'densenet_lite_136-gru',
        *,
        det_model_name: str = 'ch_PP-OCRv4_det',
        cand_alphabet: Optional[Union[Collection, str]] = None,
        context: str = 'cpu',  # ['cpu', 'gpu', 'cuda']
        rec_model_fp: Optional[str] = None,
        rec_model_backend: str = 'onnx',  # ['pytorch', 'onnx']
        rec_vocab_fp: Optional[Union[str, Path]] = None,
        rec_more_configs: Optional[Dict[str, Any]] = None,
        rec_root: Union[str, Path] = data_dir(),
        det_model_fp: Optional[str] = None,
        det_model_backend: str = 'onnx',  # ['pytorch', 'onnx']
        det_more_configs: Optional[Dict[str, Any]] = None,
        det_root: Union[str, Path] = det_data_dir(),
        **kwargs,
    ):
        """
        识别模型初始化函数。

        Args:
            rec_model_name (str): 识别模型名称。默认为 `densenet_lite_136-gru`
            det_model_name (str): 检测模型名称。默认为 `ch_PP-OCRv4_det`
            cand_alphabet (Optional[Union[Collection, str]]): 待识别字符所在的候选集合。默认为 `None`，表示不限定识别字符范围
            context (str): 'cpu', or 'gpu'。表明预测时是使用CPU还是GPU。默认为 `cpu`。
                此参数仅在 `model_backend=='pytorch'` 时有效。
            rec_model_fp (Optional[str]): 如果不使用系统自带的识别模型，可以通过此参数直接指定所使用的模型文件（'.ckpt' 文件）
            rec_model_backend (str): 'pytorch', or 'onnx'。表明识别时是使用 PyTorch 版本模型，还是使用 ONNX 版本模型。
                同样的模型，ONNX 版本的预测速度一般是 PyTorch 版本的2倍左右。默认为 'onnx'。
            rec_vocab_fp (Optional[Union[str, Path]]): 识别字符集合的文件路径，即 `label_cn.txt` 文件路径。取值为 `None` 表示使用系统设定的词表。
                若训练的自有模型更改了字符集，看通过此参数传入新的字符集文件路径。
            rec_more_configs (Optional[Dict[str, Any]]): 识别模型初始化时传入的其他参数。
            rec_root (Union[str, Path]): 识别模型文件所在的根目录。
                Linux/Mac下默认值为 `~/.cnocr`，表示模型文件所处文件夹类似 `~/.cnocr/2.3/densenet_lite_136-gru`。
                Windows下默认值为 `C:/Users/<username>/AppData/Roaming/cnocr`。
            det_model_fp (Optional[str]): 如果不使用系统自带的检测模型，可以通过此参数直接指定所使用的模型文件（'.ckpt' 文件）
            det_model_backend (str): 'pytorch', or 'onnx'。表明检测时是使用 PyTorch 版本模型，还是使用 ONNX 版本模型。
                同样的模型，ONNX 版本的预测速度一般是 PyTorch 版本的2倍左右。默认为 'onnx'。
            det_more_configs (Optional[Dict[str, Any]]): 识别模型初始化时传入的其他参数。
            det_root: 检测模型文件所在的根目录。
                Linux/Mac下默认值为 `~/.cnstd`，表示模型文件所处文件夹类似 `~/.cnstd/1.2/db_resnet18`
                Windows下默认值为 `C:/Users/<username>/AppData/Roaming/cnstd`。
            **kwargs: 目前未被使用。

        Examples:
            使用默认参数：
            >>> ocr = CnOcr()

            使用指定模型：
            >>> ocr = CnOcr('densenet_lite_136-gru')

            识别时只考虑数字：
            >>> ocr = CnOcr(rec_model_name='densenet_lite_136-gru', det_model_name='naive_det', cand_alphabet='0123456789')

            只检测和识别水平文字：
            >>> ocr = CnOcr(rec_model_name='densenet_lite_136-gru', det_model_name='db_shufflenet_v2_small', det_more_configs={'rotated_bbox': False})

        """
        if kwargs.get('model_name') is not None and rec_model_name is None:
            # 兼容前面的版本
            rec_model_name = kwargs.get('model_name')

        self.rec_space = REC_AVAILABLE_MODELS.get_space(
            rec_model_name, rec_model_backend
        )
        if self.rec_space is None:
            logger.warning(
                'no available model is found for name %s and backend %s'
                % (rec_model_name, rec_model_backend)
            )
            rec_model_backend = 'onnx' if rec_model_backend == 'pytorch' else 'pytorch'
            logger.warning(
                'trying to use name %s and backend %s'
                % (rec_model_name, rec_model_backend)
            )
            self.rec_space = REC_AVAILABLE_MODELS.get_space(
                rec_model_name, rec_model_backend
            )

        if self.rec_space == REC_AVAILABLE_MODELS.CNOCR_SPACE:
            rec_cls = Recognizer
        elif self.rec_space == PP_SPACE:
            rec_name = REC_AVAILABLE_MODELS.get_value(rec_model_name, rec_model_backend, 'recognizer')
            rec_cls = RapidRecognizer if rec_name == 'RapidRecognizer' else PPRecognizer
            if rec_vocab_fp is not None:
                logger.warning('param `vocab_fp` is invalid for %s models' % PP_SPACE)
        else:
            raise NotImplementedError(
                '%s is not supported currently' % ((rec_model_name, rec_model_backend),)
            )

        rec_more_configs = rec_more_configs or dict()
        self.rec_model = rec_cls(
            model_name=rec_model_name,
            model_backend=rec_model_backend,
            cand_alphabet=cand_alphabet,
            context=context,
            model_fp=rec_model_fp,
            root=rec_root,
            vocab_fp=rec_vocab_fp,
            **rec_more_configs,
        )

        self.det_model = None
        if det_model_name in DET_MODLE_NAMES:
            det_more_configs = det_more_configs or dict()
            self.det_model = CnStd(
                det_model_name,
                model_backend=det_model_backend,
                context=context,
                model_fp=det_model_fp,
                root=det_root,
                **det_more_configs,
            )

    def ocr(
        self,
        img_fp: Union[str, Path, Image.Image, torch.Tensor, np.ndarray],
        rec_batch_size=1,
        return_cropped_image=False,
        **det_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        识别函数。

        Args:
            img_fp (Union[str, Path, Image.Image, torch.Tensor, np.ndarray]): image file path;
                or loaded image by `Image.open()`;
                or color image torch.Tensor or np.ndarray,
                    with shape [height, width] or [height, width, channel].
                    channel should be 1 (gray image) or 3 (RGB formatted color image). scaled in [0, 255];
            rec_batch_size: `batch_size` when recognizing detected text boxes. Default: `1`.
            return_cropped_image: 是否返回检测出的文本框图片数据.
            **det_kwargs: kwargs for the detector model when calling its `detect()` function.
                - resized_shape: `int` or `tuple`, `tuple` 含义为 (height, width), `int` 则表示高宽都为此值；
                      检测前，先把原始图片resize到接近此大小（只是接近，未必相等）。默认为 `(768, 768)`。
                      注：这个取值对检测结果的影响较大，可以针对自己的应用多尝试几组值，再选出最优值。
                          例如 (512, 768), (768, 768), (768, 1024)等。
                - preserve_aspect_ratio: 对原始图片resize时是否保持高宽比不变。默认为 `True`。
                - min_box_size: 如果检测出的文本框高度或者宽度低于此值，此文本框会被过滤掉。默认为 `8`，也即高或者宽低于 `8` 的文本框会被过滤去掉。
                - box_score_thresh: 过滤掉得分低于此值的文本框。默认为 `0.3`。
                - batch_size: 待处理图片很多时，需要分批处理，每批图片的数量由此参数指定。默认为 `20`。

        Returns:
            list of detected texts, which element is a dict, with keys:
                - 'text' (str): 识别出的文本
                - 'score' (float): 识别结果的得分（置信度），取值范围为 `[0, 1]`；得分越高表示越可信
                - 'position' (np.ndarray or None): 检测出的文字对应的矩形框；np.ndarray, shape: (4, 2)，对应 box 4个点的坐标值 (x, y) ;
                  注：此值只有使用检测模型时才会存在，未使用检测模型（`det_model_name=='naive_det'`）时无此值
                - 'cropped_img' (np.ndarray): 当 `return_cropped_image==True` 时才会有此值。
                          对应 `position` 中被检测出的图片（RGB格式），会把倾斜的图片旋转为水平。
                          np.ndarray 类型，shape: (height, width, 3), 取值范围：[0, 255]；

            示例：
            ```
             [{'position': array([[ 31.,  28.],
                   [511.,  28.],
                   [511.,  55.],
                   [ 31.,  55.]], dtype=float32),
               'score': 0.8812797665596008,
               'text': '第一行'},
              {'position': array([[ 30.,  71.],
                    [541.,  71.],
                    [541.,  97.],
                    [ 30.,  97.]], dtype=float32),
               'score': 0.859879732131958,
               'text': '第二行'},
              {'position': array([[ 28., 110.],
                    [541., 111.],
                    [541., 141.],
                    [ 28., 140.]], dtype=float32),
               'score': 0.7850906848907471,
               'text': '第三行'}
             ]
            ```
        """
        if isinstance(img_fp, Image.Image):  # Image to np.ndarray
            img_fp = np.asarray(img_fp.convert('RGB'))

        if self.det_model is not None:
            return self._ocr_with_det_model(
                img_fp, rec_batch_size, return_cropped_image, **det_kwargs
            )

        img = self._prepare_img(img_fp)

        if min(img.shape[0], img.shape[1]) < 2:
            return []
        if img.mean() < 145:  # 把黑底白字的图片对调为白底黑字
            img = 255 - img
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = np.squeeze(img, axis=-1)
        line_imgs = line_split(img, blank=True)
        line_img_list = [line_img for line_img, _ in line_imgs]
        line_chars_list = self.ocr_for_single_lines(
            line_img_list, batch_size=rec_batch_size
        )
        if return_cropped_image:
            for _out, line_img in zip(line_chars_list, line_img_list):
                _out['cropped_img'] = line_img

        return line_chars_list

    def _ocr_with_det_model(
        self,
        img: Union[str, Path, torch.Tensor, np.ndarray],
        rec_batch_size: int,
        return_cropped_image: bool,
        **det_kwargs,
    ) -> List[Dict[str, Any]]:
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3 and img.shape[2] == 1:
                # (H, W, 1) -> (H, W)
                img = img.squeeze(-1)
            if len(img.shape) == 2:
                # (H, W) -> (H, W, 3)
                img = np.array(Image.fromarray(img).convert('RGB'))

        box_infos = self.det_model.detect(img, **det_kwargs)

        cropped_img_list = [
            box_info['cropped_img'] for box_info in box_infos['detected_texts']
        ]
        ocr_outs = self.ocr_for_single_lines(
            cropped_img_list, batch_size=rec_batch_size
        )
        results = []
        for box_info, ocr_out in zip(box_infos['detected_texts'], ocr_outs):
            _out = OcrResult(**ocr_out)
            _out.position = box_info['box']
            if return_cropped_image:
                _out.cropped_img = box_info['cropped_img']
            results.append(_out.to_dict())

        return results

    def _prepare_img(
        self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        """

        Args:
            img_fp (Union[str, Path, torch.Tensor, np.ndarray]):
                image array with type torch.Tensor or np.ndarray,
                with shape [height, width] or [height, width, channel].
                channel should be 1 (gray image) or 3 (color image).

        Returns:
            np.ndarray: with shape (height, width, 1), dtype uint8, scale [0, 255]
        """
        img = img_fp
        if isinstance(img_fp, (str, Path)):
            if not os.path.isfile(img_fp):
                raise FileNotFoundError(img_fp)
            img = read_img(img_fp, gray=False)

        if isinstance(img, torch.Tensor):
            img = img.numpy()

        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        elif len(img.shape) == 3:
            assert img.shape[2] in (1, 3)

        if img.dtype != np.dtype('uint8'):
            img = img.astype('uint8')
        return img

    def ocr_for_single_line(
        self, img_fp: Union[str, Path, torch.Tensor, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Recognize characters from an image with only one-line characters.

        Args:
            img_fp (Union[str, Path, torch.Tensor, np.ndarray]):
                image file path; or image torch.Tensor or np.ndarray,
                with shape [height, width] or [height, width, channel].
                The optional channel should be 1 (gray image) or 3 (color image).

        Returns:
            dict, with keys:
                - 'text' (str): 识别出的文本
                - 'score' (float): 识别结果的得分（置信度），取值范围为 `[0, 1]`；得分越高表示越可信

            示例：
            ```
             {'score': 0.8812797665596008,
              'text': '当前行'}
            ```
        """
        img = self._prepare_img(img_fp)
        res = self.ocr_for_single_lines([img])
        return res[0]

    def ocr_for_single_lines(
        self,
        img_list: List[Union[str, Path, torch.Tensor, np.ndarray]],
        batch_size: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Batch recognize characters from a list of one-line-characters images.

        Args:
            img_list (List[Union[str, Path, torch.Tensor, np.ndarray]]):
                list of images, in which each element should be a line image array,
                with type torch.Tensor or np.ndarray.
                Each element should be a tensor with values ranging from 0 to 255,
                and with shape [height, width] or [height, width, channel].
                The optional channel should be 1 (gray image) or 3 (color image).
                注：img_list 不宜包含太多图片，否则同时导入这些图片会消耗很多内存。
            batch_size: 待处理图片很多时，需要分批处理，每批图片的数量由此参数指定。默认为 `1`。

        Returns:
            list of detected texts, which element is a dict, with keys:
                - 'text' (str): 识别出的文本
                - 'score' (float): 识别结果的得分（置信度），取值范围为 `[0, 1]`；得分越高表示越可信

            示例：
            ```
             [{'score': 0.8812797665596008,
               'text': '第一行'},
              {'score': 0.859879732131958,
               'text': '第二行'},
              {'score': 0.7850906848907471,
               'text': '第三行'}
             ]
            ```
        """
        if len(img_list) == 0:
            return []

        img_list = [self._prepare_img(img) for img in img_list]
        outs = self.rec_model.recognize(img_list, batch_size=batch_size)

        results = []
        for text, score in outs:
            _out = OcrResult(text=text, score=score)
            results.append(_out.to_dict())

        return results
