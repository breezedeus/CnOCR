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
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from cnstd.utils import pil_to_numpy, imsave

from cnocr import CnOcr, DET_AVAILABLE_MODELS, REC_AVAILABLE_MODELS
from cnocr.utils import set_logger, draw_ocr_results, download


logger = set_logger()
st.set_page_config(layout="wide")


def plot_for_debugging(rotated_img, one_out, box_score_thresh, crop_ncols, prefix_fp):
    import matplotlib.pyplot as plt
    import math

    rotated_img = rotated_img.copy()
    crops = [info['cropped_img'] for info in one_out]
    print('%d boxes are found' % len(crops))
    if len(crops) < 1:
        return
    ncols = crop_ncols
    nrows = math.ceil(len(crops) / ncols)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    for i, axi in enumerate(ax.flat):
        if i >= len(crops):
            break
        axi.imshow(crops[i])
    crop_fp = '%s-crops.png' % prefix_fp
    plt.savefig(crop_fp)
    print('cropped results are save to file %s' % crop_fp)

    for info in one_out:
        box, score = info.get('position'), info['score']
        if score < box_score_thresh:  # score < 0.5
            continue
        if box is not None:
            box = box.astype(int).reshape(-1, 2)
            cv2.polylines(rotated_img, [box], True, color=(255, 0, 0), thickness=2)
    result_fp = '%s-result.png' % prefix_fp
    imsave(rotated_img, result_fp, normalized=False)
    print('boxes results are save to file %s' % result_fp)


@st.cache(allow_output_mutation=True)
def get_ocr_model(det_model_name, rec_model_name, det_more_configs):
    det_model_name, det_model_backend = det_model_name
    rec_model_name, rec_model_backend = rec_model_name
    return CnOcr(
        det_model_name=det_model_name,
        det_model_backend=det_model_backend,
        rec_model_name=rec_model_name,
        rec_model_backend=rec_model_backend,
        det_more_configs=det_more_configs,
    )


def visualize_naive_result(img, det_model_name, std_out, box_score_thresh):
    if len(std_out) < 1:
        st.warning(f'未检测到文本！')
        return
    img = pil_to_numpy(img).transpose((1, 2, 0)).astype(np.uint8)

    plot_for_debugging(img, std_out, box_score_thresh, 2, './streamlit-app')
    st.subheader('Detection Result')
    if det_model_name == 'default_det':
        st.warning('⚠️ Warning: "default_det" 检测模型不返回文本框位置！')
    cols = st.columns([1, 7, 1])
    cols[1].image('./streamlit-app-result.png')

    st.subheader('Recognition Result')
    cols = st.columns([1, 7, 1])
    cols[1].image('./streamlit-app-crops.png')

    _visualize_ocr(std_out)


def _visualize_ocr(ocr_outs):
    st.empty()
    if len(ocr_outs) < 1:
        return
    ocr_res = OrderedDict({'文本': []})
    ocr_res['得分'] = []
    for out in ocr_outs:
        # cropped_img = out['cropped_img']  # 检测出的文本框
        ocr_res['得分'].append(out['score'])
        ocr_res['文本'].append(out['text'])
    st.table(ocr_res)


def visualize_result(img, ocr_outs):
    out_draw_fp = './streamlit-app-det-result.png'
    font_path = 'docs/fonts/simfang.ttf'
    if not os.path.exists(font_path):
        url = 'https://huggingface.co/datasets/breezedeus/cnocr-wx-qr-code/resolve/main/fonts/simfang.ttf'
        os.makedirs(os.path.dirname(font_path), exist_ok=True)
        download(url, path=font_path, overwrite=True)
    draw_ocr_results(img, ocr_outs, out_draw_fp, font_path)
    st.image(out_draw_fp)


def main():
    st.sidebar.header('模型设置')
    det_models = list(DET_AVAILABLE_MODELS.all_models())
    det_models.append(('naive_det', 'onnx'))
    det_models.sort()
    det_model_name = st.sidebar.selectbox(
        '选择检测模型', det_models, index=det_models.index(('ch_PP-OCRv4_det', 'onnx'))
    )

    all_models = list(REC_AVAILABLE_MODELS.all_models())
    all_models.sort()
    idx = all_models.index(('densenet_lite_136-gru', 'onnx'))
    rec_model_name = st.sidebar.selectbox('选择识别模型', all_models, index=idx)

    st.sidebar.subheader('检测参数')
    rotated_bbox = st.sidebar.checkbox('是否检测带角度文本框', value=True)
    use_angle_clf = st.sidebar.checkbox('是否使用角度预测模型校正文本框', value=False)
    new_size = st.sidebar.slider(
        'resize 后图片（长边）大小', min_value=124, max_value=4096, value=768
    )
    box_score_thresh = st.sidebar.slider(
        '得分阈值（低于阈值的结果会被过滤掉）', min_value=0.05, max_value=0.95, value=0.3
    )
    min_box_size = st.sidebar.slider(
        '框大小阈值（更小的文本框会被过滤掉）', min_value=4, max_value=50, value=10
    )
    # std = get_std_model(det_model_name, rotated_bbox, use_angle_clf)

    # st.sidebar.markdown("""---""")
    # st.sidebar.header('CnOcr 设置')
    det_more_configs = dict(rotated_bbox=rotated_bbox, use_angle_clf=use_angle_clf)
    ocr = get_ocr_model(det_model_name, rec_model_name, det_more_configs)

    st.markdown('# 开源Python OCR工具 ' '[CnOCR](https://github.com/breezedeus/cnocr)')
    st.markdown('> 详细说明参见：[CnOCR 文档](https://cnocr.readthedocs.io/) ；'
                '欢迎加入 [交流群](https://www.breezedeus.com/article/join-group) ；'
                '作者：[breezedeus](https://github.com/breezedeus) 。')
    st.markdown('')
    st.subheader('选择待检测图片')
    content_file = st.file_uploader('', type=["png", "jpg", "jpeg", "webp"])
    if content_file is None:
        st.stop()

    try:
        img = Image.open(content_file).convert('RGB')

        ocr_out = ocr.ocr(
            img,
            return_cropped_image=True,
            resized_shape=new_size,
            preserve_aspect_ratio=True,
            box_score_thresh=box_score_thresh,
            min_box_size=min_box_size,
        )
        if det_model_name[0] == 'naive_det':
            visualize_naive_result(img, det_model_name[0], ocr_out, box_score_thresh)
        else:
            visualize_result(img, ocr_out)

    except Exception as e:
        st.error(e)


if __name__ == '__main__':
    main()
