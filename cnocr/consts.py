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

import os
import string
from collections import OrderedDict
from pathlib import Path
from typing import Tuple, Set, Dict, Any, Optional, Union
import logging
from copy import deepcopy

from .__version__ import __version__

logger = logging.getLogger(__name__)


# 模型版本只对应到第二层，第三层的改动表示模型兼容。
# 如: __version__ = '2.2.*'，对应的 MODEL_VERSION 都是 '2.2'
MODEL_VERSION = '.'.join(__version__.split('.', maxsplit=2)[:2])
DOWNLOAD_SOURCE = os.environ.get('CNOCR_DOWNLOAD_SOURCE', 'HF')

IMG_STANDARD_HEIGHT = 32
CN_VOCAB_FP = Path(__file__).parent.absolute() / 'label_cn.txt'
NUMBER_VOCAB_FP = Path(__file__).parent.absolute() / 'label_number.txt'

ENCODER_CONFIGS = {
    'densenet': {  # 长度压缩至 1/8（seq_len == 35），输出的向量长度为 4*128 = 512
        'growth_rate': 32,
        'block_config': [2, 2, 2, 2],
        'num_init_features': 64,
        'out_length': 512,  # 输出的向量长度为 4*128 = 512
    },
    'densenet_1112': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 1, 1, 2],
        'num_init_features': 64,
        'out_length': 400,
    },
    'densenet_1114': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 1, 1, 4],
        'num_init_features': 64,
        'out_length': 656,
    },
    'densenet_1122': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 1, 2, 2],
        'num_init_features': 64,
        'out_length': 464,
    },
    'densenet_1124': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 1, 2, 4],
        'num_init_features': 64,
        'out_length': 720,
    },
    'densenet_lite_113': {  # 长度压缩至 1/8（seq_len == 35），输出的向量长度为 2*136 = 272
        'growth_rate': 32,
        'block_config': [1, 1, 3],
        'num_init_features': 64,
        'out_length': 272,  # 输出的向量长度为 2*80 = 160
    },
    'densenet_lite_114': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 1, 4],
        'num_init_features': 64,
        'out_length': 336,
    },
    'densenet_lite_124': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 2, 4],
        'num_init_features': 64,
        'out_length': 368,
    },
    'densenet_lite_134': {  # 长度压缩至 1/8（seq_len == 35）
        'growth_rate': 32,
        'block_config': [1, 3, 4],
        'num_init_features': 64,
        'out_length': 400,
    },
    'densenet_lite_136': {  # 长度压缩至 1/8（seq_len == 35）; #params, with fc: 680 K, with gru: 1.4 M
        'growth_rate': 32,
        'block_config': [1, 3, 6],
        'num_init_features': 64,
        'out_length': 528,
    },
    'densenet_lite_246': {  # 长度压缩至 1/8（seq_len == 35）; #params, with fc: 831 K, with gru: 1.6 M
        'growth_rate': 32,
        'block_config': [2, 4, 6],
        'num_init_features': 64,
        'out_length': 576,
    },
    'densenet_lite_666': {  # 长度压缩至 1/8（seq_len == 35）; #params, with fc: 1.4 M, with gru: xxx
        'growth_rate': 32,
        'block_config': [6, 6, 6],
        'num_init_features': 64,
        'out_length': 704,
    },
    'densenet_lite_999': {  # 长度压缩至 1/8（seq_len == 35）; #params, with fc: 2.4 M, with gru: xxx
        'growth_rate': 32,
        'block_config': [9, 9, 9],
        'num_init_features': 64,
        'out_length': 1040,
    },
    'mobilenetv3_tiny': {'arch': 'tiny', 'out_length': 384,},
    'mobilenetv3_small': {'arch': 'small', 'out_length': 384,},
}

DECODER_CONFIGS = {
    'lstm': {'rnn_units': 128},
    'gru': {'rnn_units': 128, 'num_layers': 2},
    'gru_base': {'rnn_units': 256, 'num_layers': 2},
    'gru_large': {'rnn_units': 512, 'num_layers': 3},
    'fc': {'hidden_size': 128, 'dropout': 0.1},
    'fc_base': {'hidden_size': 256, 'dropout': 0.3},
    'fc_large': {'hidden_size': 512, 'dropout': 0.4},
}


HF_HUB_REPO_ID = "breezedeus/cnstd-cnocr-models"
HF_HUB_SUBFOLDER = "models/cnocr/%s" % MODEL_VERSION
PAID_HF_HUB_REPO_ID = "breezedeus/paid-models"
PAID_HF_HUB_SUBFOLDER = "cnocr/%s" % MODEL_VERSION
CN_OSS_ENDPOINT = (
    "https://sg-models.oss-cn-beijing.aliyuncs.com/cnocr/%s/" % MODEL_VERSION
)


def format_hf_hub_url(url: str, is_paid_model=False) -> dict:
    out_dict = {'filename': url}

    if is_paid_model:
        repo_id = PAID_HF_HUB_REPO_ID
        subfolder = PAID_HF_HUB_SUBFOLDER
    else:
        repo_id = HF_HUB_REPO_ID
        subfolder = HF_HUB_SUBFOLDER
        out_dict['cn_oss'] = CN_OSS_ENDPOINT
    out_dict.update(
        {'repo_id': repo_id, 'subfolder': subfolder,}
    )
    return out_dict


class AvailableModels(object):
    CNOCR_SPACE = '__cnocr__'

    # name: (epoch, url)
    FREE_MODELS = OrderedDict(
        {
            ('densenet_lite_136-gru', 'pytorch'): {
                'epoch': 4,
                'url': 'densenet_lite_136-gru.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('densenet_lite_136-gru', 'onnx'): {
                'epoch': 4,
                'url': 'densenet_lite_136-gru-onnx.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('scene-densenet_lite_136-gru', 'pytorch'): {
                'epoch': 4,
                'url': 'scene-densenet_lite_136-gru.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('scene-densenet_lite_136-gru', 'onnx'): {
                'epoch': 4,
                'url': 'scene-densenet_lite_136-gru-onnx.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('doc-densenet_lite_136-gru', 'pytorch'): {
                'epoch': 4,
                'url': 'doc-densenet_lite_136-gru.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('doc-densenet_lite_136-gru', 'onnx'): {
                'epoch': 4,
                'url': 'doc-densenet_lite_136-gru-onnx.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('number-densenet_lite_136-fc', 'onnx'): {
                'epoch': 23,
                'url': 'number-densenet_lite_136-fc-onnx.zip',
                'vocab_fp': NUMBER_VOCAB_FP,
            },
            ('number-densenet_lite_136-fc', 'pytorch'): {
                'epoch': 23,
                'url': 'number-densenet_lite_136-fc.zip',
                'vocab_fp': NUMBER_VOCAB_FP,
            },
        }
    )

    PAID_MODELS = OrderedDict(
        {
            ('densenet_lite_246-gru_base', 'pytorch'): {
                'epoch': 5,
                'url': 'densenet_lite_246-gru_base.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('densenet_lite_246-gru_base', 'onnx'): {
                'epoch': 5,
                'url': 'densenet_lite_246-gru_base-onnx.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('scene-densenet_lite_246-gru_base', 'pytorch'): {
                'epoch': 4,
                'url': 'scene-densenet_lite_246-gru_base.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('scene-densenet_lite_246-gru_base', 'onnx'): {
                'epoch': 4,
                'url': 'scene-densenet_lite_246-gru_base-onnx.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('doc-densenet_lite_246-gru_base', 'pytorch'): {
                'epoch': 4,
                'url': 'doc-densenet_lite_246-gru_base.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('doc-densenet_lite_246-gru_base', 'onnx'): {
                'epoch': 4,
                'url': 'doc-densenet_lite_246-gru_base-onnx.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('densenet_lite_666-gru_large', 'pytorch'): {
                'epoch': 4,
                'url': 'densenet_lite_666-gru_large.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('densenet_lite_666-gru_large', 'onnx'): {
                'epoch': 4,
                'url': 'densenet_lite_666-gru_large-onnx.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('scene-densenet_lite_666-gru_large', 'pytorch'): {
                'epoch': 5,
                'url': 'scene-densenet_lite_666-gru_large.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('scene-densenet_lite_666-gru_large', 'onnx'): {
                'epoch': 5,
                'url': 'scene-densenet_lite_666-gru_large-onnx.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('doc-densenet_lite_666-gru_large', 'pytorch'): {
                'epoch': 5,
                'url': 'doc-densenet_lite_666-gru_large.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('doc-densenet_lite_666-gru_large', 'onnx'): {
                'epoch': 5,
                'url': 'doc-densenet_lite_666-gru_large-onnx.zip',
                'vocab_fp': CN_VOCAB_FP,
            },
            ('number-densenet_lite_136-gru', 'pytorch'): {
                'epoch': 28,
                'url': 'number-densenet_lite_136-gru.zip',
                'vocab_fp': NUMBER_VOCAB_FP,
            },
            ('number-densenet_lite_136-gru', 'onnx'): {
                'epoch': 28,
                'url': 'number-densenet_lite_136-gru-onnx.zip',
                'vocab_fp': NUMBER_VOCAB_FP,
            },
            ('number-densenet_lite_666-gru_large', 'pytorch'): {
                'epoch': 28,
                'url': 'number-densenet_lite_666-gru_large.zip',
                'vocab_fp': NUMBER_VOCAB_FP,
            },
            ('number-densenet_lite_666-gru_large', 'onnx'): {
                'epoch': 28,
                'url': 'number-densenet_lite_666-gru_large-onnx.zip',
                'vocab_fp': NUMBER_VOCAB_FP,
            },
        }
    )

    CNOCR_MODELS = deepcopy(FREE_MODELS)
    CNOCR_MODELS.update(PAID_MODELS)
    OUTER_MODELS = {}

    def all_models(self) -> Set[Tuple[str, str]]:
        return set(self.CNOCR_MODELS.keys()) | set(self.OUTER_MODELS.keys())

    def __contains__(self, model_name_backend: Tuple[str, str]) -> bool:
        return model_name_backend in self.all_models()

    def register_models(self, model_dict: Dict[Tuple[str, str], Any], space: str):
        assert not space.startswith('__')
        for key, val in model_dict.items():
            if key in self.CNOCR_MODELS or key in self.OUTER_MODELS:
                logger.warning(
                    'model %s has already existed, and will be ignored' % key
                )
                continue
            val = deepcopy(val)
            val['space'] = space
            self.OUTER_MODELS[key] = val

    def get_space(self, model_name, model_backend) -> Optional[str]:
        if (model_name, model_backend) in self.CNOCR_MODELS:
            return self.CNOCR_SPACE
        elif (model_name, model_backend) in self.OUTER_MODELS:
            return self.OUTER_MODELS[(model_name, model_backend)]['space']
        return self.CNOCR_SPACE

    def get_vocab_fp(
        self, model_name: str, model_backend: str
    ) -> Optional[Union[str, Path]]:
        if (model_name, model_backend) in self.CNOCR_MODELS:
            return self.CNOCR_MODELS[(model_name, model_backend)]['vocab_fp']
        elif (model_name, model_backend) in self.OUTER_MODELS:
            return self.OUTER_MODELS[(model_name, model_backend)]['vocab_fp']
        else:
            logger.warning(
                'no predefined vocab_fp is found for model %s, use the default %s'
                % ((model_name, model_backend), CN_VOCAB_FP)
            )
            return CN_VOCAB_FP

    def get_value(self, model_name, model_backend, key) -> Optional[Any]:
        if (model_name, model_backend) in self.CNOCR_MODELS:
            info = self.CNOCR_MODELS[(model_name, model_backend)]
        elif (model_name, model_backend) in self.OUTER_MODELS:
            info = self.OUTER_MODELS[(model_name, model_backend)]
        else:
            logger.warning(
                'no url is found for model %s' % ((model_name, model_backend),)
            )
            return None
        return info.get(key)

    def get_epoch(self, model_name, model_backend) -> Optional[int]:
        if (model_name, model_backend) in self.CNOCR_MODELS:
            return self.CNOCR_MODELS[(model_name, model_backend)]['epoch']
        return None

    def get_url(self, model_name, model_backend) -> Optional[dict]:
        is_paid_model = False
        if (model_name, model_backend) in self.CNOCR_MODELS:
            url = self.CNOCR_MODELS[(model_name, model_backend)]['url']
            is_paid_model = (model_name, model_backend) in self.PAID_MODELS
        elif (model_name, model_backend) in self.OUTER_MODELS:
            url = self.OUTER_MODELS[(model_name, model_backend)]['url']
        else:
            logger.warning(
                'no url is found for model %s' % ((model_name, model_backend),)
            )
            return None
        url = format_hf_hub_url(url, is_paid_model=is_paid_model)
        return url


AVAILABLE_MODELS = AvailableModels()

# 候选字符集合
NUMBERS = string.digits + string.punctuation
ENG_LETTERS = string.digits + string.ascii_letters + string.punctuation
