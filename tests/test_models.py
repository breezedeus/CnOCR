# coding: utf-8
import os
import sys
from copy import deepcopy
import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

from cnocr.utils import set_logger, get_model_size
from cnocr.consts import IMG_STANDARD_HEIGHT, ENCODER_CONFIGS, DECODER_CONFIGS
from cnocr.models.densenet import DenseNet, DenseNetLite


logger = set_logger('info')


def test_conv():
    conv = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=False)
    input = torch.rand(1, 32, 10, 4)
    res = conv(input)
    logger.info(res.shape)


def test_densenet():
    width = 280
    img = torch.rand(4, 1, IMG_STANDARD_HEIGHT, width)
    net = DenseNet(32, [2, 2, 2, 2], 64)
    net.eval()
    logger.info(net)
    logger.info(f'model size: {get_model_size(net)}')  # 406464
    logger.info(img.shape)
    res = net(img)
    logger.info(res.shape)
    assert tuple(res.shape) == (4, 128, 4, 35)

    net = DenseNet(32, [2, 2, 1, 1], 64)
    net.eval()
    logger.info(net)
    logger.info(f'model size: {get_model_size(net)}')  # 301440
    logger.info(img.shape)
    res = net(img)
    logger.info(res.shape)
    assert tuple(res.shape) == (4, 80, 4, 35)

    net = DenseNet(32, [2, 1, 1, 1], 64)
    net.eval()
    logger.info(net)
    logger.info(f'model size: {get_model_size(net)}')  # 243616
    logger.info(img.shape)
    res = net(img)
    logger.info(res.shape)
    assert tuple(res.shape) == (4, 72, 4, 35)

    net = DenseNet(32, [1, 1, 1, 2], 64)
    net.eval()
    logger.info(net)
    logger.info(f'model size: {get_model_size(net)}')  # 230680
    logger.info(img.shape)
    res = net(img)
    logger.info(res.shape)
    assert tuple(res.shape) == (4, 100, 4, 35)


def test_densenet_lite():
    width = 280
    img = torch.rand(4, 1, IMG_STANDARD_HEIGHT, width)
    net = DenseNetLite(32, [2, 2, 2], 64)
    net.eval()
    logger.info(net)
    logger.info(f'model size: {get_model_size(net)}')  # 302976
    logger.info(img.shape)
    res = net(img)
    logger.info(res.shape)
    assert tuple(res.shape) == (4, 128, 2, 35)

    # net = DenseNetLite(32, [2, 1, 1], 64)
    # net.eval()
    # logger.info(net)
    # logger.info(f'model size: {get_model_size(net)}')  # 197952
    # logger.info(img.shape)
    # res = net(img)
    # logger.info(res.shape)
    # assert tuple(res.shape) == (4, 80, 2, 35)

    net = DenseNetLite(32, [1, 1, 2], 64)
    net.eval()
    logger.info(net)
    logger.info(f'model size: {get_model_size(net)}')  # 186672
    logger.info(img.shape)
    res = net(img)
    logger.info(res.shape)
    assert tuple(res.shape) == (4, 104, 2, 35)

    net = DenseNetLite(32, [1, 2, 2], 64)
    net.eval()
    logger.info(net)
    logger.info(f'model size: {get_model_size(net)}')  #
    logger.info(img.shape)
    res = net(img)
    logger.info(res.shape)
    assert tuple(res.shape) == (4, 120, 2, 35)


def test_crnn():
    _hp = deepcopy(HP)
    _hp.set_seq_length(_hp.img_width // 4)
    x = nd.random.randn(128, 64, 32, 280)
    layer_channels_list = [(64, 128, 256, 512), (32, 64, 128, 256)]
    for layer_channels in layer_channels_list:
        densenet = DenseNet(layer_channels)
        crnn = CRnn(_hp, densenet)
        crnn.initialize()
        y = crnn(x)
        logger.info(
            'output shape: %s', y.shape
        )  # res: `(sequence_length, batch_size, 2*num_hidden)`
        assert y.shape == (_hp.seq_length, _hp.batch_size, 2 * _hp.num_hidden)
        logger.info('number of parameters: %d', cal_num_params(crnn))


def test_crnn_lstm():
    hp = deepcopy(HP)
    hp.set_seq_length(hp.img_width // 8)
    data = mx.sym.Variable('data', shape=(128, 1, 32, 280))
    pred = crnn_lstm(HP, data)
    pred_shape = pred.infer_shape()[1][0]
    logger.info('shape of pred: %s', pred_shape)
    assert pred_shape == (hp.seq_length, hp.batch_size, 2 * hp.num_hidden)


def test_crnn_lstm_lite():
    hp = deepcopy(HP)
    width = hp.img_width  # 280
    data = mx.sym.Variable('data', shape=(128, 1, 32, width))
    for shorter in (False, True):
        pred = crnn_lstm_lite(HP, data, shorter=shorter)
        pred_shape = pred.infer_shape()[1][0]
        logger.info('shape of pred: %s', pred_shape)
        seq_len = hp.img_width // 8 if shorter else hp.img_width // 4 - 1
        assert pred_shape == (seq_len, hp.batch_size, 2 * hp.num_hidden)


def test_pipline():
    hp = deepcopy(HP)
    hp.set_seq_length(hp.img_width // 4)
    hp._loss_type = None  # infer mode
    layer_channels_list = [(64, 128, 256, 512), (32, 64, 128, 256)]
    for layer_channels in layer_channels_list:
        densenet = DenseNet(layer_channels)
        crnn = CRnn(hp, densenet)
        data = mx.sym.Variable('data', shape=(128, 1, 32, 280))
        pred = pipline(crnn, hp, data)
        pred_shape = pred.infer_shape()[1][0]
        logger.info('shape of pred: %s', pred_shape)
        assert pred_shape == (hp.batch_size * hp.seq_length, hp.num_classes)


MODEL_NAMES = []
for emb_model in ENCODER_CONFIGS:
    for seq_model in DECODER_CONFIGS:
        MODEL_NAMES.append('%s-%s' % (emb_model, seq_model))


@pytest.mark.parametrize(
    'model_name', MODEL_NAMES
)
def test_gen_networks(model_name):
    logger.info('model_name: %s', model_name)
    network, hp = gen_network(model_name, HP)
    shape_dict = get_infer_shape(network, HP)
    logger.info('shape_dict: %s', shape_dict)
    assert shape_dict['pred_fc_output'] == (
        hp.batch_size * hp.seq_length,
        hp.num_classes,
    )
