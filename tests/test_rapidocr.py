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
import torch
from pathlib import Path

from rapidocr_onnxruntime import RapidOCR
from rapidocr_onnxruntime.utils import LoadImage
from rapidocr_onnxruntime.ch_ppocr_rec import TextRecognizer
from rapidocr_onnxruntime.ch_ppocr_det import TextDetector            
from rapidocr_onnxruntime.utils import LoadImage

def test_whole_pipeline():
    engine = RapidOCR(det_model_path="en_PP-OCRv3_det_infer.onnx", rec_model_path="en_PP-OCRv4_rec_infer.onnx")

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example_dir = Path(root_dir) / 'docs/examples'
    # img_path = example_dir / 'multi-line_cn1.png'
    img_path = example_dir / 'en_ticket.jpeg'
    result, elapse = engine(img_path)
    print(result)
    breakpoint()
    print(elapse)


def test_rec():
    config = {'intra_op_num_threads': -1, 'inter_op_num_threads': -1, 'use_cuda': False, 'use_dml': False, 'model_path': 'en_PP-OCRv4_rec_infer.onnx', 'rec_img_shape': [3, 48, 320], 'rec_batch_num': 6}
    # config = dict(det_model_path="en_PP-OCRv3_det_infer.onnx")
    engine = TextRecognizer(config)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    example_dir = Path(root_dir) / 'docs/examples'
    # img_path = example_dir / 'multi-line_cn1.png'
    img_path = example_dir / 'hybrid.png'
    result, elapse = engine(LoadImage()(img_path))
    print(result)