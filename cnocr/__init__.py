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

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from cnstd.consts import AVAILABLE_MODELS as DET_AVAILABLE_MODELS
from cnstd.utils import pil_to_numpy

from .consts import (
    MODEL_VERSION,
    AVAILABLE_MODELS as REC_AVAILABLE_MODELS,
    NUMBERS,
    ENG_LETTERS,
)
from .utils import read_img, set_logger
from .cn_ocr import CnOcr
from .recognizer import gen_model
from .line_split import line_split
from .classification import ImageClassifier
