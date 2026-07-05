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
from unittest.mock import patch

from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR
from rapidocr.utils.load_image import LoadImage
from rapidocr.ch_ppocr_rec import TextRecognizer, TextRecInput
from cnocr import CnOcr
from cnocr.ppocr.rapid_recognizer import RapidRecognizer, Config
from cnocr.utils import set_logger

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


@pytest.mark.skipif(
    not hasattr(OCRVersion, "PPOCRV6") or not hasattr(ModelType, "SMALL"),
    reason="PP-OCRv6 requires rapidocr>=3.9.0",
)
def test_rapid_recognizer_supports_ppocrv6_config():
    recognizer_calls = []
    prepare_calls = []

    def fake_text_recognizer(config):
        recognizer_calls.append(config)
        return lambda args: None

    def fake_prepare_model_files(model_fp, remote_repo):
        prepare_calls.append((model_fp, remote_repo))
        return model_fp

    with patch(
        "cnocr.ppocr.rapid_recognizer.TextRecognizer",
        side_effect=fake_text_recognizer,
    ), patch(
        "cnocr.ppocr.rapid_recognizer.prepare_model_files",
        side_effect=fake_prepare_model_files,
    ):
        recognizer = RapidRecognizer(model_name="multi_PP-OCRv6")

    assert recognizer_calls
    config = recognizer_calls[0]
    assert config.ocr_version == OCRVersion.PPOCRV6
    assert config.model_type == ModelType.SMALL
    assert config.lang_type == LangRec.CH
    assert config.model_path.endswith("PP-OCRv6_rec_small.onnx")
    assert config.model_root_dir == recognizer._model_dir
    assert prepare_calls == [
        (
            os.path.join(recognizer._model_dir, "PP-OCRv6_rec_small.onnx"),
            "breezedeus/cnocr-ppocr-multi_PP-OCRv6",
        )
    ]


@pytest.mark.skipif(
    not hasattr(OCRVersion, "PPOCRV6") or not hasattr(ModelType, "MEDIUM"),
    reason="PP-OCRv6 requires rapidocr>=3.9.0",
)
def test_rapid_recognizer_supports_ppocrv6_medium_config():
    recognizer_calls = []

    def fake_text_recognizer(config):
        recognizer_calls.append(config)
        return lambda args: None

    with patch(
        "cnocr.ppocr.rapid_recognizer.TextRecognizer",
        side_effect=fake_text_recognizer,
    ), patch(
        "cnocr.ppocr.rapid_recognizer.prepare_model_files",
        side_effect=lambda model_fp, remote_repo: model_fp,
    ):
        RapidRecognizer(model_name="multi_PP-OCRv6_medium")

    assert recognizer_calls[0].ocr_version == OCRVersion.PPOCRV6
    assert recognizer_calls[0].model_type == ModelType.MEDIUM


@pytest.mark.skipif(
    not hasattr(OCRVersion, "PPOCRV6") or not hasattr(ModelType, "SMALL"),
    reason="PP-OCRv6 requires rapidocr>=3.9.0",
)
def test_rapid_recognizer_supports_ppocrv6_lang_override():
    recognizer_calls = []

    def fake_text_recognizer(config):
        recognizer_calls.append(config)
        return lambda args: None

    with patch(
        "cnocr.ppocr.rapid_recognizer.TextRecognizer",
        side_effect=fake_text_recognizer,
    ), patch(
        "cnocr.ppocr.rapid_recognizer.prepare_model_files",
        side_effect=lambda model_fp, remote_repo: model_fp,
    ):
        RapidRecognizer(model_name="multi_PP-OCRv6", lang_type=LangRec.EN)

    assert recognizer_calls[0].lang_type == LangRec.EN


def test_cnocr_passes_explicit_rec_lang_type_to_rapid_recognizer(monkeypatch):
    recognizer_calls = []

    class FakeRapidRecognizer:
        def __init__(self, **kwargs):
            recognizer_calls.append(kwargs)
            self._lang_type = kwargs.get("lang_type")
            self._model_fp = "/tmp/model.onnx"

    monkeypatch.setattr("cnocr.cn_ocr.RapidRecognizer", FakeRapidRecognizer)

    CnOcr(
        rec_model_name="multi_PP-OCRv6_small",
        det_model_name="naive_det",
        rec_lang_type="en",
    )

    assert recognizer_calls[0]["lang_type"] == "en"


@pytest.mark.skipif(
    not hasattr(OCRVersion, "PPOCRV6") or not hasattr(ModelType, "MEDIUM"),
    reason="PP-OCRv6 requires rapidocr>=3.9.0",
)
def test_rapid_recognizer_supports_ppocrv6_string_lang_type():
    recognizer_calls = []

    def fake_text_recognizer(config):
        recognizer_calls.append(config)
        return lambda args: None

    with patch(
        "cnocr.ppocr.rapid_recognizer.TextRecognizer",
        side_effect=fake_text_recognizer,
    ), patch(
        "cnocr.ppocr.rapid_recognizer.prepare_model_files",
        side_effect=lambda model_fp, remote_repo: model_fp,
    ):
        RapidRecognizer(model_name="multi_PP-OCRv6_medium", lang_type="french")

    assert recognizer_calls[0].lang_type == "french"


@pytest.mark.skipif(
    not hasattr(OCRVersion, "PPOCRV6"),
    reason="PP-OCRv6 requires rapidocr>=3.9.0",
)
def test_rapid_recognizer_rejects_ppocrv6_multi_lang_type():
    with pytest.raises(ValueError, match="concrete lang_type"):
        RapidRecognizer(model_name="multi_PP-OCRv6", lang_type="multi")


@pytest.mark.skipif(
    not hasattr(OCRVersion, "PPOCRV6") or not hasattr(ModelType, "TINY"),
    reason="PP-OCRv6 requires rapidocr>=3.9.0",
)
def test_rapid_recognizer_rejects_ppocrv6_tiny_japan_lang_type():
    with pytest.raises(ValueError, match="Unsupported rec.lang_type='japan'"):
        RapidRecognizer(model_name="multi_PP-OCRv6_tiny", lang_type="japan")


@pytest.mark.skipif(
    not hasattr(OCRVersion, "PPOCRV6"),
    reason="PP-OCRv6 requires rapidocr>=3.9.0",
)
def test_rapid_recognizer_rejects_unknown_ppocrv6_model():
    with pytest.raises(NotImplementedError, match="not a downloadable model"):
        RapidRecognizer(model_name="unknown_PP-OCRv6")


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
