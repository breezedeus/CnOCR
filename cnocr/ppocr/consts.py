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

from pathlib import Path


VOCAB_DIR = Path(__file__).parent / "utils"

MODEL_LABELS_FILE_DICT = {
    ("ch_PP-OCRv3", "onnx"): {
        "vocab_fp": VOCAB_DIR / "ppocr_keys_v1.txt",  # 简体中英文
        "url": "ch_PP-OCRv3_rec_infer-onnx.zip",
    },
    ("ch_ppocr_mobile_v2.0", "onnx"): {
        "vocab_fp": VOCAB_DIR / "ppocr_keys_v1.txt",
        "url": "ch_ppocr_mobile_v2.0_rec_infer-onnx.zip",
    },
    ("en_number_mobile_v2.0", "onnx"): {
        "vocab_fp": VOCAB_DIR / "en_dict.txt",
        "url": "en_number_mobile_v2.0_rec_infer-onnx.zip",
    },
    ("chinese_cht_PP-OCRv3", "onnx"): {
        "vocab_fp": VOCAB_DIR / "chinese_cht_dict.txt",  # 繁体中文
        "url": "chinese_cht_PP-OCRv3_rec_infer-onnx.zip",
    },
    ("japan_PP-OCRv3", "onnx"): {
        "recognizer": "RapidRecognizer",
        "repo": "breezedeus/cnocr-ppocr-japan_PP-OCRv3",
    },
    ("korean_PP-OCRv3", "onnx"): {
        "recognizer": "RapidRecognizer",
        "repo": "breezedeus/cnocr-ppocr-korean_PP-OCRv3",
    },
    ("latin_PP-OCRv3", "onnx"): {
        "recognizer": "RapidRecognizer",
        "repo": "breezedeus/cnocr-ppocr-latin_PP-OCRv3",
    },
    ("arabic_PP-OCRv3", "onnx"): {
        "recognizer": "RapidRecognizer",
        "repo": "breezedeus/cnocr-ppocr-arabic_PP-OCRv3",
    },
    ("en_PP-OCRv3", "onnx"): {
        "vocab_fp": VOCAB_DIR / "en_dict.txt",  # 英文
        "url": "en_PP-OCRv3_rec_infer-onnx.zip",
        "recognizer": "RapidRecognizer",
        "repo": "breezedeus/cnocr-ppocr-en_PP-OCRv3",
    },
    ("en_PP-OCRv4", "onnx"): {
        "recognizer": "RapidRecognizer",
        "repo": "breezedeus/cnocr-ppocr-en_PP-OCRv4",
    },
    ("ch_PP-OCRv4", "onnx"): {
        "recognizer": "RapidRecognizer",
        "repo": "breezedeus/cnocr-ppocr-ch_PP-OCRv4",
    },
    ("ch_PP-OCRv4_server", "onnx"): {
        "recognizer": "RapidRecognizer",
        "repo": "breezedeus/cnocr-ppocr-ch_PP-OCRv4_server",
    },
}

PP_SPACE = "ppocr"
