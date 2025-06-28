# coding: utf-8
# Copyright (C) 2021, [Breezedeus](https://github.com/breezedeus).
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
import pytest
from pathlib import Path
import logging

from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR
from rapidocr.utils import LoadImage
from rapidocr.ch_ppocr_rec import TextRecognizer, TextRecInput
from cnocr.ppocr.rapid_recognizer import RapidRecognizer, Config
from cnocr.utils import set_logger
from cnocr import CnOcr

logger = set_logger(log_level=logging.INFO)


def test_rec():
    config = Config(Config.DEFAULT_CFG)
    engine = TextRecognizer(config)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example_dir = Path(root_dir) / 'docs/examples'
    # img_path = example_dir / 'multi-line_cn1.png'
    img_path = example_dir / 'hybrid.png'
    img = LoadImage()(img_path)
    rec_input = TextRecInput(img=img, return_word_box=True)
    result = engine(rec_input)
    print(result)


def test_cnocr():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example_dir = Path(root_dir) / 'docs/examples'
    # img_path = example_dir / 'multi-line_cn1.png'
    img_path = example_dir / 'hybrid.png'
    ocr = CnOcr(det_model_name='ch_PP-OCRv5_det',  # 'ch_PP-OCRv5_det_server'
                rec_model_name='ch_PP-OCRv5',  # 'ch_PP-OCRv5_server'
                )
    result = ocr.ocr(img_path)
    print(result)


def test_rec_rapidocr():
    engine = RapidRecognizer(model_name="ch_PP-OCRv3")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example_dir = Path(root_dir) / 'docs/examples'
    # img_path = example_dir / 'multi-line_cn1.png'
    img_path = example_dir / 'hybrid.png'
    result = engine.recognize([img_path])
    print(result)


def test_whole_pipeline():
    engine = RapidOCR(
        params={
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.lang_type": LangDet.CH,
        "Det.model_type": ModelType.SERVER,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Rec.engine_type": EngineType.ONNXRUNTIME,
        "Rec.lang_type": LangRec.CH,
        "Rec.model_type": ModelType.SERVER,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.model_path": "models/rapid_ocr/ch_PP-OCRv5_server_rec_infer.onnx",
        # "Rec.model_path": "models/rapid_ocr/ch_PP-OCRv5_rec_mobile_infer/ch_PP-OCRv5_rec_mobile_infer.onnx",
        # "Rec.rec_keys_path": "models/rapid_ocr/ch_PP-OCRv5_rec_mobile_infer/ppocrv5_dict.txt",
    }
    )

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example_dir = Path(root_dir) / 'docs/examples'
    img_path = example_dir / 'hybrid.png'
    result = engine(img_path, )
    print(result)

    result.vis("vis_result.jpg")