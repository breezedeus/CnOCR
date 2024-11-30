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

from __future__ import absolute_import, division, print_function
import os
import logging
import time
from collections import Counter
import json
import glob
from operator import itemgetter
from pathlib import Path
from multiprocessing import Process
import subprocess

import click
import numpy as np
import torchmetrics
from torchvision import transforms as T
import torch

from cnocr.consts import MODEL_VERSION, ENCODER_CONFIGS, DECODER_CONFIGS
from cnocr.utils import (
    set_logger,
    load_model_params,
    check_model_name,
    save_img,
    read_img,
    draw_ocr_results,
    read_charset,
)
from cnocr.data_utils.aug import NormalizeAug
from cnocr.trainer import PlTrainer, resave_model, Metrics
from cnocr import CnOcr, gen_model
from cnocr.recognizer import Recognizer

_CONTEXT_SETTINGS = {"help_option_names": ['-h', '--help']}
logger = set_logger(log_level=logging.INFO)

DEFAULT_MODEL_NAME = 'densenet_lite_136-gru'
LEGAL_MODEL_NAMES = {
    enc_name + '-' + dec_name
    for enc_name in ENCODER_CONFIGS.keys()
    for dec_name in DECODER_CONFIGS.keys()
}


@click.group(context_settings=_CONTEXT_SETTINGS)
def cli():
    pass


@cli.command('train')
@click.option(
    '-m',
    '--rec-model-name',
    type=str,
    default=DEFAULT_MODEL_NAME,
    help='识别模型名称。默认值为 `%s`' % DEFAULT_MODEL_NAME,
)
@click.option(
    '-i',
    '--index-dir',
    type=str,
    required=True,
    help='索引文件所在的文件夹，会读取文件夹中的 train.tsv 和 dev.tsv 文件',
)
@click.option(
    '--train-config-fp',
    type=str,
    required=True,
    help='识别模型训练使用的json配置文件，参考 `docs/examples/train_config.json`',
)
@click.option(
    "--finetuning", is_flag=True, help="是否为精调模式（精调模式使用更温柔的transform）。默认为 `False`",
)
@click.option(
    '-r',
    '--resume-from-checkpoint',
    type=str,
    default=None,
    help='恢复此前中断的训练状态，继续训练识别模型。所以文件中应该包含训练状态。默认为 `None`',
)
@click.option(
    '-p',
    '--pretrained-model-fp',
    type=str,
    default=None,
    help='导入的训练好的识别模型，作为模型初始值。'
    '优先级低于"--resume-from-checkpoint"，当传入"--resume-from-checkpoint"时，此传入失效。默认为 `None`',
)
def train(
    rec_model_name,
    index_dir,
    train_config_fp,
    finetuning,
    resume_from_checkpoint,
    pretrained_model_fp,
):
    """训练识别模型"""
    from cnocr.dataset import OcrDataModule

    from cnocr.data_utils.transforms import (
        train_transform,
        ft_transform,
        test_transform,
    )

    check_model_name(rec_model_name)
    # train_transform = T.Compose(
    #     [
    #         RandomStretchAug(min_ratio=0.5, max_ratio=1.5),
    #         # RandomCrop((8, 10)),
    #         T.RandomInvert(p=0.2),
    #         T.RandomApply([T.RandomRotation(degrees=1)], p=0.4),
    #         # T.RandomAutocontrast(p=0.05),
    #         # T.RandomPosterize(bits=4, p=0.3),
    #         # T.RandomAdjustSharpness(sharpness_factor=0.5, p=0.3),
    #         # T.RandomEqualize(p=0.3),
    #         # T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
    #         NormalizeAug(),
    #         # RandomPaddingAug(p=0.5, max_pad_len=72),
    #     ]
    # )
    # val_transform = NormalizeAug()
    val_transform = test_transform

    train_config = json.load(open(train_config_fp))

    data_mod = OcrDataModule(
        index_dir=index_dir,
        vocab_fp=train_config['vocab_fp'],
        img_folder=train_config['img_folder'],
        train_transforms=train_transform if not finetuning else ft_transform,
        val_transforms=val_transform,
        batch_size=train_config['batch_size'],
        train_bucket_size=train_config.get('train_bucket_size'),
        num_workers=train_config['num_workers'],
        pin_memory=train_config['pin_memory'],
    )
    data_mod.setup('')

    # train_ds = data_mod.train
    # for i in range(min(100, len(train_ds))):
    #     visualize_example(train_transform(train_ds[i][0]), 'debugs/train-1-%d' % i)
    #     visualize_example(train_transform(train_ds[i][0]), 'debugs/train-2-%d' % i)
    #     visualize_example(train_transform(train_ds[i][0]), 'debugs/train-3-%d' % i)
    # val_ds = data_mod.val
    # for i in range(min(10, len(val_ds))):
    #     visualize_example(val_transform(val_ds[i][0]), 'debugs/val-1-%d' % i)
    #     visualize_example(val_transform(val_ds[i][0]), 'debugs/val-2-%d' % i)
    #     visualize_example(val_transform(val_ds[i][0]), 'debugs/val-2-%d' % i)
    # return

    trainer = PlTrainer(
        train_config, ckpt_fn=['cnocr', 'v%s' % MODEL_VERSION, rec_model_name]
    )
    model = gen_model(rec_model_name, data_mod.vocab)
    logger.info(model)

    if pretrained_model_fp is not None:
        load_model_params(model, pretrained_model_fp)

    trainer.fit(
        model, datamodule=data_mod, resume_from_checkpoint=resume_from_checkpoint
    )


def visualize_example(example, fp_prefix):
    if not os.path.exists(os.path.dirname(fp_prefix)):
        os.makedirs(os.path.dirname(fp_prefix))
    image = example
    save_img(image, '%s-image.jpg' % fp_prefix)


@cli.command('predict')
@click.option(
    '-m',
    '--rec-model-name',
    type=str,
    default=DEFAULT_MODEL_NAME,
    help='识别模型名称。默认值为 %s' % DEFAULT_MODEL_NAME,
)
@click.option(
    '-b',
    '--rec-model-backend',
    type=click.Choice(['pytorch', 'onnx']),
    default='onnx',
    help='识别模型类型。默认值为 `onnx`',
)
@click.option(
    '-v',
    '--rec-vocab-fp',
    type=str,
    default=None,
    help='识别模型使用的词表。默认取值为 `None` 表示使用系统设定的词表',
)
@click.option(
    '-d',
    '--det-model-name',
    type=str,
    default='ch_PP-OCRv4_det',
    help='检测模型名称。默认值为 ch_PP-OCRv4_det',
)
@click.option(
    '--det-model-backend',
    type=click.Choice(['pytorch', 'onnx']),
    default='onnx',
    help='检测模型类型。默认值为 `onnx`',
)
@click.option(
    '--det-resized-shape', type=int, default=768, help='检测模型输入图像尺寸。默认值为 768',
)
@click.option(
    '-p',
    '--pretrained-model-fp',
    type=str,
    default=None,
    help='识别模型使用训练好的模型。默认为 `None`，表示使用系统自带的预训练模型',
)
@click.option(
    "-c",
    "--context",
    help="使用cpu还是 `gpu` 运行代码，也可指定为特定gpu，如`cuda:0`。默认为 `cpu`",
    type=str,
    default='cpu',
)
@click.option("-i", "--img-file-or-dir", required=True, help="输入图片的文件路径或者指定的文件夹")
@click.option(
    "-s",
    "--single-line",
    is_flag=True,
    help="是否输入图片只包含单行文字。对包含单行文字的图片，不做按行切分；否则会先对图片按行分割后再进行识别",
)
@click.option(
    "--draw-results-dir", default=None, help="画出的检测与识别效果图所存放的目录；取值为 `None` 表示不画图",
)
@click.option(
    "--draw-font-path", default='./docs/fonts/simfang.ttf', help="画出检测与识别效果图时使用的字体文件",
)
@click.option(
    "--show-details", is_flag=True, default=False, help="是否打印识别结果详情。默认值为 `False`",
)
@click.option(
    "--verbose", is_flag=True, default=False, help="是否打印详细日志信息。默认值为 `False`",
)
def predict(
    rec_model_name,
    rec_model_backend,
    rec_vocab_fp,
    det_model_name,
    det_model_backend,
    det_resized_shape,
    pretrained_model_fp,
    context,
    img_file_or_dir,
    single_line,
    draw_results_dir,
    draw_font_path,
    show_details,
    verbose,
):
    """模型预测""",
    if verbose:
        logger = set_logger(log_level=logging.DEBUG)
    else:
        logger = set_logger(log_level=logging.INFO)

    fp_list = []
    if os.path.isfile(img_file_or_dir):
        fp_list.append(img_file_or_dir)
    elif os.path.isdir(img_file_or_dir):
        fn_list = glob.glob1(img_file_or_dir, '*g')
        fp_list = [os.path.join(img_file_or_dir, fn) for fn in fn_list]
    else:
        raise ValueError(
            f'"{img_file_or_dir}" is not found, which must be a file or a directory'
        )
    if len(fp_list) == 0:
        raise ValueError(f'No image is found from "{img_file_or_dir}".')

    ocr = CnOcr(
        rec_model_name=rec_model_name,
        rec_model_backend=rec_model_backend,
        rec_vocab_fp=rec_vocab_fp,
        det_model_name=det_model_name,
        det_model_backend=det_model_backend,
        rec_model_fp=pretrained_model_fp,
        context=context,
        # det_more_configs={'rotated_bbox': False},
    )
    ocr_func = ocr.ocr_for_single_line if single_line else ocr.ocr
    ocr_kwargs = {}
    if not single_line:
        ocr_kwargs['resized_shape'] = det_resized_shape

    for fp in fp_list:
        start_time = time.time()
        logger.info('\n' + '=' * 10 + fp + '=' * 10)
        res = ocr_func(
            fp,
            **ocr_kwargs,
            # box_score_thresh=0.14,
            # min_box_size=20,
        )
        logger.info('time cost: %f' % (time.time() - start_time))
        if show_details:
            logger.info(res)
        else:
            logger.info('\n'.join([line_res['text'] for line_res in res]))
        if single_line:
            res = [res]

        if not single_line and draw_results_dir is not None:
            if not os.path.isfile(draw_font_path):
                logger.error(
                    'can not find the font file {}, so stop drawing ocr results'.format(
                        draw_font_path
                    )
                )
            else:
                os.makedirs(draw_results_dir, exist_ok=True)
                out_draw_fp = os.path.join(
                    draw_results_dir, os.path.basename(fp) + '-result.jpg'
                )
                draw_ocr_results(
                    fp, res, out_draw_fp=out_draw_fp, font_path=draw_font_path
                )


@cli.command('evaluate')
@click.option(
    '-m',
    '--rec-model-name',
    type=str,
    default=DEFAULT_MODEL_NAME,
    help='识别模型名称。默认值为 %s' % DEFAULT_MODEL_NAME,
)
@click.option(
    '-b',
    '--rec-model-backend',
    type=click.Choice(['pytorch', 'onnx']),
    default='onnx',
    help='识别模型类型。默认值为 `onnx`',
)
@click.option(
    '-v',
    '--rec-vocab-fp',
    type=str,
    default=None,
    help='识别模型使用的词表。默认取值为 `None` 表示使用系统设定的词表',
)
@click.option(
    '-p',
    '--pretrained-model-fp',
    type=str,
    default=None,
    help='识别模型使用训练好的模型。默认为 `None`，表示使用系统自带的预训练模型',
)
@click.option(
    "-c",
    "--context",
    help="使用cpu还是 `gpu` 运行代码，也可指定为特定gpu，如`cuda:0`。默认为 `cpu`",
    type=str,
    default='cpu',
)
@click.option(
    "-i",
    "--eval-index-fp",
    type=str,
    help='待评估文件所在的索引文件，格式与训练时训练集索引文件相同，每行格式为 `<图片路径>\t<以空格分割的labels>`',
    default='test.txt',
)
@click.option("--image-folder", required=True, help="图片所在文件夹，相对于索引文件中记录的图片位置")
@click.option("--batch-size", type=int, help="batch size. 默认值：128", default=128)
@click.option(
    '-o',
    '--output-dir',
    type=str,
    default='eval_results',
    help='存放评估结果的文件夹。默认值：`eval_results`',
)
@click.option(
    "--verbose", is_flag=True, help="whether to print details to screen",
)
def evaluate(
    rec_model_name,
    rec_model_backend,
    rec_vocab_fp,
    pretrained_model_fp,
    context,
    eval_index_fp,
    image_folder,
    batch_size,
    output_dir,
    verbose,
):
    """评估模型效果。检测模型使用 `det_model_name='naive_det'` 。"""
    if verbose:
        logger = set_logger(log_level=logging.DEBUG)
    else:
        logger = set_logger(log_level=logging.INFO)

    ocr = CnOcr(
        rec_model_name=rec_model_name,
        rec_model_backend=rec_model_backend,
        rec_vocab_fp=rec_vocab_fp,
        rec_model_fp=pretrained_model_fp,
        det_model_name='naive_det',
        context=context,
    )

    fn_labels_list = read_input_file(eval_index_fp)

    metrics_config = {"complete_match": {}, "cer": {}}
    metrics = Metrics.from_config(metrics_config)
    cer = torchmetrics.text.CharErrorRate()
    miss_cnt, redundant_cnt = Counter(), Counter()
    total_time_cost = 0.0
    bad_cnt = 0
    badcases = []

    start_idx = 0
    while start_idx < len(fn_labels_list):
        logger.info('start_idx: %d', start_idx)
        batch = fn_labels_list[start_idx : start_idx + batch_size]
        img_fps = [os.path.join(image_folder, fn) for fn, _ in batch]
        reals = [''.join(labels) for _, labels in batch]

        imgs = [read_img(img) for img in img_fps]
        start_time = time.time()
        outs = ocr.ocr_for_single_lines(imgs, batch_size=1)
        total_time_cost += time.time() - start_time

        preds = [out['text'] for out in outs]
        metrics.add_batch(reals, preds)
        for bad_info in compare_preds_to_reals(preds, reals, img_fps):
            if verbose:
                logger.info('\t'.join(bad_info))
            _cer = float(cer(preds=bad_info[2], target=bad_info[1]))
            bad_info.insert(0, _cer)
            badcases.append(bad_info)
            miss_cnt.update(list(bad_info[-2]))
            redundant_cnt.update(list(bad_info[-1]))
            bad_cnt += 1

        start_idx += batch_size

    results = metrics.compute()
    badcases.sort(key=itemgetter(0), reverse=True)

    output_dir = Path(output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir)
    with open(output_dir / 'badcases.txt', 'w') as f:
        f.write(
            '\t'.join(
                [
                    'distance',
                    'image_fp',
                    'real_words',
                    'pred_words',
                    'miss_words',
                    'redundant_words',
                ]
            )
            + '\n'
        )
        for bad_info in badcases:
            f.write('\t'.join(map(str, bad_info)) + '\n')
    with open(output_dir / 'miss_words_stat.txt', 'w') as f:
        for word, num in miss_cnt.most_common():
            f.write('\t'.join([word, str(num)]) + '\n')
    with open(output_dir / 'redundant_words_stat.txt', 'w') as f:
        for word, num in redundant_cnt.most_common():
            f.write('\t'.join([word, str(num)]) + '\n')

    logger.info(f'metrics: {results}')
    logger.info(
        "number of total cases: %d, number of bad cases: %d, acc: %.4f, time cost per image: %f"
        % (
            len(fn_labels_list),
            bad_cnt,
            1.0 - bad_cnt / len(fn_labels_list),
            total_time_cost / len(fn_labels_list),
        )
    )
    logger.info(f'more detailed infos can be found in the dir: {output_dir}')


def read_input_file(in_fp):
    fn_labels_list = []
    with open(in_fp) as f:
        for line in f:
            fields = line.strip().split('\t')
            labels = fields[1].split(' ')
            labels = [l if l != '<space>' else ' ' for l in labels]
            fn_labels_list.append((fields[0], labels))
    return fn_labels_list


def compare_preds_to_reals(batch_preds, batch_reals, batch_img_fns):
    for preds, reals, img_fn in zip(batch_preds, batch_reals, batch_img_fns):
        if preds == reals:
            continue
        preds_set, reals_set = set(preds), set(reals)

        miss_words = reals_set.difference(preds_set)
        redundant_words = preds_set.difference(reals_set)
        yield [
            img_fn,
            ''.join(reals),
            ''.join(preds),
            ''.join(miss_words),
            ''.join(redundant_words),
        ]


@cli.command('resave')
@click.option('-i', '--input-model-fp', type=str, required=True, help='输入的识别模型文件路径')
@click.option('-o', '--output-model-fp', type=str, required=True, help='输出的识别模型文件路径')
def resave_model_file(
    input_model_fp, output_model_fp,
):
    """训练好的识别模型会存储训练状态，使用此命令去掉预测时无关的数据，降低模型大小"""
    resave_model(input_model_fp, output_model_fp, map_location='cpu')


def export_to_onnx(model_name, vocab_fp, output_model_fp, input_model_fp=None):
    import onnx

    ocr = Recognizer(
        model_name, model_backend='pytorch', model_fp=input_model_fp, vocab_fp=vocab_fp,
    )
    model = ocr._model

    x = torch.randn(1, 1, 32, 280)
    input_lengths = torch.tensor([280])

    model.postprocessor = None  # 这个无法ONNX化
    symbolic_names = {0: 'batch_size', 3: 'width'}
    with torch.no_grad():
        model.eval()
        torch.onnx.export(
            model,
            args=(x, input_lengths),
            f=output_model_fp,
            export_params=True,
            # opset_version=10,
            do_constant_folding=True,
            input_names=['x', 'input_lengths'],
            output_names=['logits', 'output_lengths'],
            dynamic_axes={
                'x': symbolic_names,  # variable length axes
                'input_lengths': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
            },
        )

    onnx_model = onnx.load(output_model_fp)
    onnx.checker.check_model(onnx_model)
    logger.info('model is exported to %s' % output_model_fp)


@cli.command('export-onnx')
@click.option(
    '-m',
    '--rec-model-name',
    type=str,
    default=DEFAULT_MODEL_NAME,
    help='识别模型名称。默认值为 `%s`' % DEFAULT_MODEL_NAME,
)
@click.option(
    '-v',
    '--rec-vocab-fp',
    type=str,
    default=None,
    help='识别模型使用的词表。默认取值为 `None` 表示使用系统设定的词表',
)
@click.option(
    '-i',
    '--input-model-fp',
    type=str,
    default=None,
    help='输入的识别模型文件路径。 默认为 `None`，表示使用系统自带的预训练模型',
)
@click.option(
    '-o', '--output-model-fp', type=str, required=True, help='输出的识别模型文件路径（.onnx）'
)
def export_onnx_model(
    rec_model_name, rec_vocab_fp, input_model_fp, output_model_fp,
):
    """把训练好的识别模型导出为 ONNX 格式。
    """
    export_to_onnx(rec_model_name, rec_vocab_fp, output_model_fp, input_model_fp)


@cli.command('serve')
@click.option(
    '-H', '--host', type=str, default='0.0.0.0', help='server host. Default: "0.0.0.0"',
)
@click.option(
    '-p', '--port', type=int, default=8501, help='server port. Default: 8501',
)
@click.option(
    '--reload',
    is_flag=True,
    help='whether to reload the server when the codes have been changed',
)
def serve(host, port, reload):
    """开启HTTP服务。"""

    path = os.path.realpath(os.path.dirname(__file__))
    api = Process(
        target=start_server,
        kwargs={'path': path, 'host': host, 'port': port, 'reload': reload},
    )
    api.start()
    api.join()


def start_server(path, host, port, reload):
    cmd = ['uvicorn', 'serve:app', '--host', host, '--port', str(port)]
    if reload:
        cmd.append('--reload')
    subprocess.call(cmd, cwd=path)


if __name__ == "__main__":
    cli()
