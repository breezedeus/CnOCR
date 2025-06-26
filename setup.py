#!/usr/bin/env python3
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
from setuptools import find_packages, setup
from pathlib import Path

PACKAGE_NAME = "cnocr"

here = Path(__file__).parent

long_description = (here / "README.md").read_text(encoding="utf-8")

about = {}
exec(
    (here / PACKAGE_NAME.replace('.', os.path.sep) / "__version__.py").read_text(
        encoding="utf-8"
    ),
    about,
)

required = [
    "click",
    "tqdm",
    "torch>=1.8.0",
    "torchvision>=0.9.0",
    "numpy",
    "pytorch-lightning>=2.0.0",
    "wandb",
    "torchmetrics",
    "pillow>=5.3.0",
    "onnx",
    "cnstd>=1.2.6",
    "rapidocr>=3.0",
]
extras_require = {
    "ort-cpu": ["onnxruntime"],
    "ort-gpu": ["onnxruntime-gpu"],
    "serve": ["uvicorn[standard]", "fastapi", "python-multipart", "pydantic"],
    "dev": ["albumentations", "pip-tools", "pytest", "datasets[vision]"],
}

entry_points = """
[console_scripts]
cnocr = cnocr.cli:cli
cnocr-clf = cnocr.clf_cli:cli
"""

setup(
    name=PACKAGE_NAME,
    version=about['__version__'],
    description="Python3 package for Chinese/English OCR, with small pretrained models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='breezedeus',
    author_email='breezedeus@163.com',
    license='Apache 2.0',
    url='https://github.com/breezedeus/cnocr',
    platforms=["Mac", "Linux", "Windows"],
    packages=find_packages(),
    include_package_data=True,
    data_files=[
        (
            '',
            [
                'cnocr/label_cn.txt',
                'cnocr/label_number.txt',
                'cnocr/ppocr/utils/ppocr_keys_v1.txt',
                'cnocr/ppocr/utils/en_dict.txt',
                'cnocr/ppocr/utils/chinese_cht_dict.txt',
            ],
        )
    ],
    entry_points=entry_points,
    install_requires=required,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
