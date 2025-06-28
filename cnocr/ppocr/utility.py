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
# Credits: adapted from https://github.com/PaddlePaddle/PaddleOCR

import argparse
import os
import sys
import math
import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..utils import get_default_ort_providers

logger = logging.getLogger(__name__)


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def init_args():
    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--use_xpu", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--min_subgraph_size", type=int, default=15)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)

    # params for text detector
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str)
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")
    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    parser.add_argument("--det_sast_polygon", type=str2bool, default=False)

    # PSE parmas
    parser.add_argument("--det_pse_thresh", type=float, default=0)
    parser.add_argument("--det_pse_box_thresh", type=float, default=0.85)
    parser.add_argument("--det_pse_min_area", type=float, default=16)
    parser.add_argument("--det_pse_box_type", type=str, default='box')
    parser.add_argument("--det_pse_scale", type=int, default=1)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='CRNN')
    parser.add_argument("--rec_model_dir", type=str)
    parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="./cnocr/ppocr/utils/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument(
        "--vis_font_path", type=str, default="./docs/fonts/simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.5)

    # params for e2e
    parser.add_argument("--e2e_algorithm", type=str, default='PGNet')
    parser.add_argument("--e2e_model_dir", type=str)
    parser.add_argument("--e2e_limit_side_len", type=float, default=768)
    parser.add_argument("--e2e_limit_type", type=str, default='max')

    # PGNet parmas
    parser.add_argument("--e2e_pgnet_score_thresh", type=float, default=0.5)
    parser.add_argument(
        "--e2e_char_dict_path", type=str, default="./ppocr/utils/ic15_dict.txt")
    parser.add_argument("--e2e_pgnet_valid_set", type=str, default='totaltext')
    parser.add_argument("--e2e_pgnet_mode", type=str, default='fast')

    # params for text classifier
    parser.add_argument("--use_angle_cls", type=str2bool, default=False)
    parser.add_argument("--cls_model_dir", type=str)
    parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
    parser.add_argument("--label_list", type=list, default=['0', '180'])
    parser.add_argument("--cls_batch_num", type=int, default=6)
    parser.add_argument("--cls_thresh", type=float, default=0.9)

    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=10)
    parser.add_argument("--use_pdserving", type=str2bool, default=False)
    parser.add_argument("--warmup", type=str2bool, default=False)

    #
    parser.add_argument(
        "--draw_img_save_dir", type=str, default="./inference_results")
    parser.add_argument("--is_visualize", type=str2bool, default=True)
    parser.add_argument("--save_crop_res", type=str2bool, default=False)
    parser.add_argument("--crop_res_save_dir", type=str, default="./output")

    # multi-process
    parser.add_argument("--use_mp", type=str2bool, default=False)
    parser.add_argument("--total_process_num", type=int, default=1)
    parser.add_argument("--process_id", type=int, default=0)

    parser.add_argument("--benchmark", type=str2bool, default=False)
    parser.add_argument("--save_log_path", type=str, default="./log_output/")

    parser.add_argument("--show_log", type=str2bool, default=True)
    parser.add_argument("--use_onnx", type=str2bool, default=False)
    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def create_predictor(model_dir, mode, ort_providers=None):
    import onnxruntime as ort

    # if mode == "det":
    #     model_dir = args.det_model_dir
    # elif mode == 'cls':
    #     model_dir = args.cls_model_dir
    # elif mode == 'rec':
    #     model_dir = args.rec_model_dir
    # elif mode == 'table':
    #     model_dir = args.table_model_dir
    # else:
    #     model_dir = args.e2e_model_dir

    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)

    model_file_path = model_dir
    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(
            model_file_path))
    if ort_providers is None:
        ort_providers = get_default_ort_providers()
    logger.debug(f'ort providers: {ort_providers}')
    sess = ort.InferenceSession(model_file_path, providers=ort_providers)
    return sess, sess.get_inputs()[0], None, None


def get_output_tensors(args, mode, predictor):
    output_names = predictor.get_output_names()
    output_tensors = []
    if mode == "rec" and args.rec_algorithm == "CRNN":
        output_name = 'softmax_0.tmp_0'
        if output_name in output_names:
            return [predictor.get_output_handle(output_name)]
        else:
            for output_name in output_names:
                output_tensor = predictor.get_output_handle(output_name)
                output_tensors.append(output_tensor)
    else:
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
    return output_tensors


# def get_infer_gpuid():
#     if os.name == 'nt':
#         try:
#             return int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
#         except KeyError:
#             return 0
#     if not paddle.fluid.core.is_compiled_with_rocm():
#         cmd = "env | grep CUDA_VISIBLE_DEVICES"
#     else:
#         cmd = "env | grep HIP_VISIBLE_DEVICES"
#     env_cuda = os.popen(cmd).readlines()
#     if len(env_cuda) == 0:
#         return 0
#     else:
#         gpu_id = env_cuda[0].strip().split("=")[1]
#         return int(gpu_id[0])


# def draw_e2e_res(dt_boxes, strs, img_path):
#     src_im = cv2.imread(img_path)
#     for box, str in zip(dt_boxes, strs):
#         box = box.astype(np.int32).reshape((-1, 1, 2))
#         cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
#         cv2.putText(
#             src_im,
#             str,
#             org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
#             fontFace=cv2.FONT_HERSHEY_COMPLEX,
#             fontScale=0.7,
#             color=(0, 255, 0),
#             thickness=1)
#     return src_im
#
#
# def draw_text_det_res(dt_boxes, img_path):
#     src_im = cv2.imread(img_path)
#     for box in dt_boxes:
#         box = np.array(box).astype(np.int32).reshape(-1, 2)
#         cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
#     return src_im


# def resize_img(img, input_size=600):
#     """
#     resize img and limit the longest side of the image to input_size
#     """
#     img = np.array(img)
#     im_shape = img.shape
#     im_size_max = np.max(im_shape[0:2])
#     im_scale = float(input_size) / float(im_size_max)
#     img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
#     return img


# def draw_ocr(image,
#              boxes,
#              txts=None,
#              scores=None,
#              drop_score=0.5,
#              font_path="./docs/fonts/simfang.ttf"):
#     """
#     Visualize the results of OCR detection and recognition
#     args:
#         image(Image|array): RGB image
#         boxes(list): boxes with shape(N, 4, 2)
#         txts(list): the texts
#         scores(list): txxs corresponding scores
#         drop_score(float): only scores greater than drop_threshold will be visualized
#         font_path: the path of font which is used to draw text
#     return(array):
#         the visualized img
#     """
#     if scores is None:
#         scores = [1] * len(boxes)
#     box_num = len(boxes)
#     for i in range(box_num):
#         if scores is not None and (scores[i] < drop_score or
#                                    math.isnan(scores[i])):
#             continue
#         box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
#         image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
#     if txts is not None:
#         img = np.array(resize_img(image, input_size=600))
#         txt_img = text_visual(
#             txts,
#             scores,
#             img_h=img.shape[0],
#             img_w=600,
#             threshold=drop_score,
#             font_path=font_path)
#         img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
#         return img
#     return image
#

def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     show_score=False,
                     drop_score=0.5,
                     font_path="./docs/fonts/simfang.ttf"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt, score) in enumerate(zip(boxes, txts, scores)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        # box 是 ndarray，  它的 dtype 必须是 float32
        box = box.astype(np.float32)
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][
            1])**2)
        box_width = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][
            1])**2)
        if show_score:
            txt = txt + ':' + str(score)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_bbox = font.getbbox(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_bbox[3] - char_bbox[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)


def text_visual(texts,
                scores,
                img_h=400,
                img_w=600,
                threshold=0.,
                font_path="./docs/fonts/simfang.ttf"):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):
    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)


# def base64_to_cv2(b64str):
#     import base64
#     data = base64.b64decode(b64str.encode('utf8'))
#     data = np.fromstring(data, np.uint8)
#     data = cv2.imdecode(data, cv2.IMREAD_COLOR)
#     return data
#
#
# def draw_boxes(image, boxes, scores=None, drop_score=0.5):
#     if scores is None:
#         scores = [1] * len(boxes)
#     for (box, score) in zip(boxes, scores):
#         if score < drop_score:
#             continue
#         box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
#         image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
#     return image


# def get_rotate_crop_image(img, points):
#     '''
#     img_height, img_width = img.shape[0:2]
#     left = int(np.min(points[:, 0]))
#     right = int(np.max(points[:, 0]))
#     top = int(np.min(points[:, 1]))
#     bottom = int(np.max(points[:, 1]))
#     img_crop = img[top:bottom, left:right, :].copy()
#     points[:, 0] = points[:, 0] - left
#     points[:, 1] = points[:, 1] - top
#     '''
#     assert len(points) == 4, "shape of points must be 4*2"
#     img_crop_width = int(
#         max(
#             np.linalg.norm(points[0] - points[1]),
#             np.linalg.norm(points[2] - points[3])))
#     img_crop_height = int(
#         max(
#             np.linalg.norm(points[0] - points[3]),
#             np.linalg.norm(points[1] - points[2])))
#     pts_std = np.float32([[0, 0], [img_crop_width, 0],
#                           [img_crop_width, img_crop_height],
#                           [0, img_crop_height]])
#     M = cv2.getPerspectiveTransform(points, pts_std)
#     dst_img = cv2.warpPerspective(
#         img,
#         M, (img_crop_width, img_crop_height),
#         borderMode=cv2.BORDER_REPLICATE,
#         flags=cv2.INTER_CUBIC)
#     dst_img_height, dst_img_width = dst_img.shape[0:2]
#     if dst_img_height * 1.0 / dst_img_width >= 1.5:
#         dst_img = np.rot90(dst_img)
#     return dst_img
#
#
def check_gpu(use_gpu):
    import torch

    if use_gpu and not torch.cuda.is_available():
        use_gpu = False
    return use_gpu


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF'}
    
    if os.path.isfile(img_file):
        try:
            with Image.open(img_file) as img:
                if img.format and img.format.lower() in [ext.lower() for ext in img_end]:
                    imgs_lists.append(img_file)
        except Exception:
            pass
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path):
                try:
                    with Image.open(file_path) as img:
                        if img.format and img.format.lower() in [ext.lower() for ext in img_end]:
                            imgs_lists.append(file_path)
                except Exception:
                    pass
    
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


# def check_and_read_gif(img_path):
#     if os.path.basename(img_path)[-3:] in ['gif', 'GIF']:
#         gif = cv2.VideoCapture(img_path)
#         ret, frame = gif.read()
#         if not ret:
#             logger.info("Cannot read {}. This gif image maybe corrupted.")
#             return None, False
#         if len(frame.shape) == 2 or frame.shape[-1] == 1:
#             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#         imgvalue = frame[:, :, ::-1]
#         return imgvalue, True
#     return None, False


if __name__ == '__main__':
    pass
