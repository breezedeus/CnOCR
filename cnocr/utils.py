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
from __future__ import division, absolute_import, print_function

import hashlib
import os
from pathlib import Path
import logging
import platform
import requests
from typing import Union, Any, Tuple, List, Optional, Dict

from tqdm import tqdm
from PIL import Image, ImageOps
import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torchvision.utils import save_image
import torchvision.transforms.functional as F

from .consts import (
    ENCODER_CONFIGS,
    DECODER_CONFIGS,
    AVAILABLE_MODELS,
    IMG_STANDARD_HEIGHT,
)

fmt = '[%(levelname)s %(asctime)s %(funcName)s:%(lineno)d] %(' 'message)s '
logging.basicConfig(format=fmt)
logging.captureWarnings(True)
logger = logging.getLogger()


def set_logger(log_file=None, log_level=logging.INFO, log_file_level=logging.NOTSET):
    """
    Example:
        >>> set_logger(log_file)
        >>> logger.info("abc'")
    """
    log_format = logging.Formatter(fmt)
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != '':
        if not Path(log_file).parent.exists():
            os.makedirs(Path(log_file).parent)
        if isinstance(log_file, Path):
            log_file = str(log_file)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def check_context(context):
    if isinstance(context, str):
        return any([ctx in context.lower() for ctx in ('gpu', 'cpu', 'cuda')])
    if isinstance(context, list):
        if len(context) < 1:
            return False
        return all(isinstance(ctx, torch.device) for ctx in context)
    return isinstance(context, torch.device)


def data_dir_default():
    """

    :return: default data directory depending on the platform and environment variables
    """
    system = platform.system()
    if system == 'Windows':
        return os.path.join(os.environ.get('APPDATA'), 'cnocr')
    else:
        return os.path.join(os.path.expanduser("~"), '.cnocr')


def data_dir():
    """

    :return: data directory in the filesystem for storage, for example when downloading models
    """
    return os.getenv('CNOCR_HOME', data_dir_default())


def check_model_name(model_name):
    encoder_type, decoder_type = model_name.split('-')[-2:]
    assert encoder_type in ENCODER_CONFIGS
    assert decoder_type in DECODER_CONFIGS


def get_default_ort_providers():
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError('Please install onnxruntime first:\n'
                          '  - For CPU only, use `pip install onnxruntime`;\n'
                          '  - For GPU, use `pip install onnxruntime-gpu`.')

    providers = []
    if 'CPUExecutionProvider' in ort.get_available_providers():
        providers.append('CPUExecutionProvider')
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')  # GPU优先
    if len(providers) == 0:
        providers = ort.get_available_providers()
    return providers


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    sha1_file = sha1.hexdigest()
    l = min(len(sha1_file), len(sha1_hash))
    return sha1.hexdigest()[0:l] == sha1_hash[0:l]


def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download a given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if (
        overwrite
        or not os.path.exists(fname)
        or (sha1_hash and not check_sha1(fname, sha1_hash))
    ):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        logger.info('Downloading %s from %s...' % (fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None:  # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(
                    r.iter_content(chunk_size=1024),
                    total=int(total_length / 1024.0 + 0.5),
                    unit='KB',
                    unit_scale=False,
                    dynamic_ncols=True,
                ):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning(
                'File {} is downloaded but the content hash does not match. '
                'The repo may be outdated or download may be incomplete. '
                'If the "repo_url" is overridden, consider switching to '
                'the default repo.'.format(fname)
            )

    return fname


def read_charset(charset_fp):
    alphabet = []
    with open(charset_fp, encoding='utf-8') as fp:
        for line in fp:
            alphabet.append(line.rstrip('\n'))
    inv_alph_dict = {_char: idx for idx, _char in enumerate(alphabet)}
    if len(alphabet) != len(inv_alph_dict):
        from collections import Counter

        repeated = Counter(alphabet).most_common(len(alphabet) - len(inv_alph_dict))
        raise ValueError('repeated chars in vocab: %s' % repeated)

    return alphabet, inv_alph_dict


def read_tsv_file(fp, sep='\t', img_folder=None, mode='eval'):
    img_fp_list, labels_list = [], []
    num_fields = 2 if mode != 'test' else 1
    with open(fp) as f:
        for line in f:
            fields = line.strip('\n').split(sep)
            assert len(fields) == num_fields
            img_fp = (
                os.path.join(img_folder, fields[0])
                if img_folder is not None
                else fields[0]
            )
            img_fp_list.append(img_fp)

            if mode != 'test':
                labels = fields[1].split(' ')
                labels_list.append(labels)

    return (img_fp_list, labels_list) if mode != 'test' else (img_fp_list, None)


def read_img(path: Union[str, Path], gray=True) -> np.ndarray:
    """
    :param path: image file path
    :param gray: whether to return a gray image array
    :return:
        * when `gray==True`, return a gray image, with dim [height, width, 1], with values range from 0 to 255
        * when `gray==False`, return a color image, with dim [height, width, 3], with values range from 0 to 255
    """
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)  # 识别旋转后的图片（pillow不会自动识别）
    except Exception as e:
        raise FileNotFoundError(f'Error loading image: {path}')
        
    if gray:
        img = img.convert('L')
        return np.expand_dims(np.array(img), -1)
    else:
        img = img.convert('RGB')
        return np.array(img)


def save_img(img: Union[Tensor, np.ndarray], path):
    if not isinstance(img, Tensor):
        img = torch.from_numpy(img)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    # img *= 255
    # img = img.to(dtype=torch.uint8)
    save_image(img, path)

    # Image.fromarray(img).save(path)


def resize_img(
    img: np.ndarray,
    target_h_w: Optional[Tuple[int, int]] = None,
    min_width: int = 8,
    return_torch: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    rescale an image tensor with [Channel, Height, Width] to the given height value, and keep the ratio
    :param img: np.ndarray; should be [c, height, width]
    :param target_h_w: (height, width) of the target image or None
    :param min_width: int; minimum width after resized. Only used when `target_h_w` is None. Default 8
    :param return_torch: bool; whether to return a `torch.Tensor` or `np.ndarray`
    :return: image tensor with the given height. The resulting dim is [C, height, width]
    """
    ori_height, ori_width = img.shape[1:]
    if target_h_w is None:
        ratio = ori_height / IMG_STANDARD_HEIGHT
        target_w = max(int(ori_width / ratio), min_width)
        target_h_w = (IMG_STANDARD_HEIGHT, target_w)

    if (ori_height, ori_width) != target_h_w:
        # img = F.resize(torch.from_numpy(img), target_h_w, antialias=True)
        new_img = cv2.resize(img.transpose((1, 2, 0)), (target_h_w[1], target_h_w[0]))  # -> (H, W, C)
        if img.ndim > new_img.ndim:
            new_img = np.expand_dims(new_img, axis=-1)
        img = new_img.transpose((2, 0, 1))  # -> (C, H, W)
    if return_torch:
        img = torch.from_numpy(img)
    return img


def normalize_img_array(img: Union[Tensor, np.ndarray]):
    """ rescale """
    if isinstance(img, Tensor):
        img = img.to(dtype=torch.float32)
    else:
        img = img.astype('float32')
    # return (img - np.mean(img, dtype=dtype)) / 255.0
    return img / 255.0
    # return (img - np.median(img)) / (np.std(img, dtype=dtype) + 1e-6)  # 转完以后有些情况会变得不可识别


def gen_length_mask(lengths: torch.Tensor, mask_size: Union[Tuple, Any]):
    """ see how it is used """
    labels = torch.arange(mask_size[-1], device=lengths.device, dtype=torch.long)
    while True:
        if len(labels.shape) >= len(mask_size):
            break
        labels = labels.unsqueeze(0)
        lengths = lengths.unsqueeze(-1)
    mask = labels < lengths
    return ~mask


def pad_img_seq(img_list: List[torch.Tensor], padding_value=0) -> torch.Tensor:
    """
    Pad a list of variable width image Tensors with `padding_value`.

    :param img_list: each element has shape [C, H, W], where W is variable width
    :param padding_value: padding value, 0 by default
    :return: [B, C, H, W_max]
    """
    img_list = [img.permute((2, 0, 1)) for img in img_list]  # [W, C, H]
    imgs = pad_sequence(
        img_list, batch_first=True, padding_value=padding_value
    )  # [B, W_max, C, H]
    return imgs.permute((0, 2, 3, 1))  # [B, C, H, W_max]


def load_model_params(model, param_fp, device='cpu'):
    checkpoint = torch.load(param_fp, map_location=device)
    state_dict = checkpoint['state_dict']
    if all([param_name.startswith('model.') for param_name in state_dict.keys()]):
        # 表示导入的模型是通过 PlTrainer 训练出的 WrapperLightningModule，对其进行转化
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k.split('.', maxsplit=1)[1]] = v
    model.load_state_dict(state_dict)
    return model


def get_model_size(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def mask_by_candidates(
    logits: np.ndarray,
    candidates: Optional[Union[str, List[str]]],
    vocab: List[str],
    letter2id: Dict[str, int],
    ignored_tokens: List[int],
):
    if candidates is None:
        return logits

    _candidates = [letter2id[word] for word in candidates]
    _candidates.sort()
    _candidates = np.array(_candidates, dtype=int)

    candidates = np.zeros((len(vocab),), dtype=bool)
    candidates[_candidates] = True
    # candidates[-1] = True  # for cnocr, 间隔符号/填充符号，必须为真
    candidates[ignored_tokens] = True
    candidates = np.expand_dims(candidates, axis=(0, 1))  # 1 x 1 x (vocab_size+1)
    candidates = candidates.repeat(logits.shape[1], axis=1)

    masked = np.ma.masked_array(data=logits, mask=~candidates, fill_value=-100.0)
    logits = masked.filled()
    return logits


def draw_ocr_results(image_fp: Union[str, Path, Image.Image], ocr_outs, out_draw_fp, font_path):
    # Credits: adapted from https://github.com/PaddlePaddle/PaddleOCR
    import cv2
    from .ppocr.utility import draw_ocr_box_txt

    if isinstance(image_fp, (str, Path)):
        img = Image.open(image_fp).convert('RGB')
    else:
        img = image_fp

    txts = []
    scores = []
    boxes = []
    for _out in ocr_outs:
        txts.append(_out['text'])
        scores.append(_out['score'])
        boxes.append(_out['position'])

    draw_img = draw_ocr_box_txt(
        img, boxes, txts, scores, drop_score=0.0, font_path=font_path
    )

    cv2.imwrite(out_draw_fp, draw_img[:, :, ::-1])
    logger.info("The visualized image saved in {}".format(out_draw_fp))
