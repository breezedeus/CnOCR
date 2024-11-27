<div align="center">
  <img src="./docs/figs/cnocr-logo.jpg" width="250px"/>
  <div>&nbsp;</div>

[![Discord](https://img.shields.io/discord/1200765964434821260?label=Discord)](https://discord.gg/GgD87WM8Tf)
[![Downloads](https://static.pepy.tech/personalized-badge/cnocr?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/cnocr)
[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fbreezedeus%2FCnOCR&label=Visitors&countColor=%23f5c791&style=flat&labelStyle=none)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2Fbreezedeus%2FCnOCR)
[![license](https://img.shields.io/github/license/breezedeus/cnocr)](./LICENSE)
[![Docs](https://readthedocs.org/projects/cnocr/badge/?version=latest)](https://cnocr.readthedocs.io/zh-cn/stable/?badge=latest)
[![PyPI version](https://badge.fury.io/py/cnocr.svg)](https://badge.fury.io/py/cnocr)
[![forks](https://img.shields.io/github/forks/breezedeus/cnocr)](https://github.com/breezedeus/cnocr)
[![stars](https://img.shields.io/github/stars/breezedeus/cnocr)](https://github.com/breezedeus/cnocr)
![last-releast](https://img.shields.io/github/release-date/breezedeus/cnocr)
![last-commit](https://img.shields.io/github/last-commit/breezedeus/cnocr)
[![Twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2Fbreezedeus)](https://twitter.com/breezedeus)

[📖 文档](https://cnocr.readthedocs.io/zh-cn/stable/) |
[🛠️ 安装](https://cnocr.readthedocs.io/zh-cn/stable/install/) |
[🧳 可用模型](https://cnocr.readthedocs.io/zh-cn/stable/models/) |
[🕹 模型训练](https://cnocr.readthedocs.io/zh-cn/stable/train/) |
[🛀🏻 在线Demo](https://huggingface.co/spaces/breezedeus/CnOCR-Demo) |
[💬 交流群](https://www.breezedeus.com/article/join-group)

</div>

<div align="center">

[English](./README_en.md) | 中文

</div>

# CnOCR

<div align="center">
<strong>Tech should serve the people, not enslave them!</strong>
<br>
<strong>请勿将此项目用于文字审查！</strong>
<br>
---
</div>

### [Update 2024.11.28]：发布 V2.3.1

主要变更：

* 基于 RapidOCR 集成 PPOCRv4 最新版 OCR 模型，提供更多的模型选择
  * 新增支持 PP-OCRv4  识别模型，包括标准版和服务器版
* 修改读文件实现方式，支持 Windows 的中文路径
* 修复Bug：当使用多个进程时，transform_func 无法序列化
* 修复Bug：与 albumentations=1.4.* 兼容

### [Update 2023.12.24]：发布 V2.3

主要变更：

* 重新训练了所有的模型，比上一版精度更高。
* 按使用场景把模型分为几大类场景（见 [识别模型列表](#可使用的识别模型)）：
  * `scene`：场景图片，适合识别一般拍照图片中的文字。此类模型以 `scene-` 开头，如模型 `scene-densenet_lite_136-gru`。
  * `doc`：文档图片，适合识别规则文档的截图图片，如书籍扫描件等。此类模型以 `doc-` 开头，如模型 `doc-densenet_lite_136-gru`。
  * `number`：仅识别**纯数字**（只能识别 `0~9` 十个数字）图片，适合银行卡号、身份证号等场景。此类模型以 `number-` 开头，如模型 `number-densenet_lite_136-gru`。
  * `general`: 通用场景，适合图片无明显倾向的一般图片。此类模型无特定开头，与旧版模型名称保持一致，如模型 `densenet_lite_136-gru`。
  > 注意 ⚠️：以上说明仅为参考，具体选择模型时建议以实际效果为准。
* 加入了两个更大的系列模型：
  * `*-densenet_lite_246-gru_base`：优先供 **知识星球** [**CnOCR/CnSTD私享群**](https://t.zsxq.com/FEYZRJQ) 会员使用，一个月后会免费开源。
  * `*-densenet_lite_666-gru_large`：Pro 模型，购买后可使用。
  
更多细节请参考：[CnOCR V2.3 新版发布：模型更好、更多、更大 | Breezedeus.com](https://www.breezedeus.com/article/cnocr-v2.3-better-more)。



[**CnOCR**](https://github.com/breezedeus/cnocr) 是 **Python 3** 下的**文字识别**（**Optical Character Recognition**，简称**OCR**）工具包，支持**简体中文**、**繁体中文**（部分模型）、**英文**和**数字**的常见字符识别，支持竖排文字的识别。自带了**20+个** [训练好的模型](https://cnocr.readthedocs.io/zh-cn/stable/models/)，适用于不同应用场景，安装后即可直接使用。同时，CnOCR也提供简单的[训练命令](https://cnocr.readthedocs.io/zh-cn/stable/train/)供使用者训练自己的模型。欢迎扫码加小助手为好友，备注 `ocr`，小助手会定期统一邀请大家入群：

<div align="center">
  <img src="https://huggingface.co/datasets/breezedeus/cnocr-wx-qr-code/resolve/main/wx-qr-code.JPG" alt="微信群二维码" width="300px"/>
</div>


作者也维护 **知识星球** [**CnOCR/CnSTD私享群**](https://t.zsxq.com/FEYZRJQ) ，这里面的提问会较快得到作者的回复，欢迎加入。**知识星球会员** 可享受以下福利：

- 可免费下载部分**未开源的付费模型**；
- 购买其他所有的付费模型一律八折优化；
- 作者快速回复使用过程中遇到的各种困难；
- 作者每月提供两次免费特有数据的训练服务。
- 星球会陆续发布一些CnOCR/CnSTD相关的私有资料；
- 星球会持续发布 OCR/STD/CV 等相关的最新研究资料。



## 详细文档

见 [CnOCR在线文档](https://cnocr.readthedocs.io/) 。

## 使用说明

**CnOCR** 从 **V2.2** 开始，内部自动调用文字检测引擎 **[CnSTD](https://github.com/breezedeus/cnstd)** 进行文字检测和定位。所以 **CnOCR** V2.2 不仅能识别排版简单的印刷体文字图片，如截图图片，扫描件等，也能识别**一般图片中的场景文字**。

以下是一些不同场景的调用示例。



## 不同场景的调用示例

### 常见的图片识别

所有参数都使用默认值即可。如果发现效果不够好，多调整下各个参数看效果，最终往往能获得比较理想的精度。

```python
from cnocr import CnOcr

img_fp = './docs/examples/huochepiao.jpeg'
ocr = CnOcr()  # 所有参数都使用默认值
out = ocr.ocr(img_fp)

print(out)
```

识别结果：

<div align="center">
  <img src="./docs/predict-outputs/huochepiao.jpeg-result.jpg" alt="火车票识别" width="800px"/>
</div>


### 排版简单的印刷体截图图片识别

针对 **排版简单的印刷体文字图片**，如截图图片，扫描件图片等，可使用 `det_model_name='naive_det'`，相当于不使用文本检测模型，而使用简单的规则进行分行。

> **Note**
>
>  `det_model_name='naive_det'` 的效果相当于 `V2.2` 之前（`V2.0.*`, `V2.1.*`）的 CnOCR 版本。

使用 `det_model_name='naive_det'` 的最大优势是**速度快**，劣势是对图片比较挑剔。如何判断是否该使用此检测模型呢？最简单的方式就是拿应用图片试试效果，效果好就用，不好就不用。

```python
from cnocr import CnOcr

img_fp = './docs/examples/multi-line_cn1.png'
ocr = CnOcr(det_model_name='naive_det') 
out = ocr.ocr(img_fp)

print(out)
```

识别结果：

<div align="center">

| 图片                                                                      | OCR结果                                                                                                                         |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| ![docs/examples/multi-line_cn1.png](./docs/examples/multi-line_cn1.png) | 网络支付并无本质的区别，因为<br />每一个手机号码和邮件地址背后<br />都会对应着一个账户--这个账<br />户可以是信用卡账户、借记卡账<br />户，也包括邮局汇款、手机代<br />收、电话代收、预付费卡和点卡<br />等多种形式。 |

</div>


### 竖排文字识别

采用来自 [**PaddleOCR**](https://github.com/PaddlePaddle/PaddleOCR)（之后简称 **ppocr**）的中文识别模型 `rec_model_name='ch_PP-OCRv3'` 进行识别。

```python
from cnocr import CnOcr

img_fp = './docs/examples/shupai.png'
ocr = CnOcr(rec_model_name='ch_PP-OCRv3')
out = ocr.ocr(img_fp)

print(out)
```

识别结果：
<div align="center">
  <img src="./docs/predict-outputs/shupai.png-result.jpg" alt="竖排文字识别" width="800px"/>
</div>


### 英文识别

虽然中文检测和识别模型也能识别英文，但**专为英文文字训练的检测器和识别器往往精度更高**。如果是纯英文的应用场景，建议使用来自 **ppocr** 的英文检测模型 `det_model_name='en_PP-OCRv3_det'`， 和英文识别模型 `rec_model_name='en_PP-OCRv3'` 。

```python
from cnocr import CnOcr

img_fp = './docs/examples/en_book1.jpeg'
ocr = CnOcr(det_model_name='en_PP-OCRv3_det', rec_model_name='en_PP-OCRv3')
out = ocr.ocr(img_fp)

print(out)
```

识别结果：

<div align="center">
  <img src="./docs/predict-outputs/en_book1.jpeg-result.jpg" alt="英文识别" width="600px"/>
</div>


### 繁体中文识别

采用来自ppocr的繁体识别模型 `rec_model_name='chinese_cht_PP-OCRv3'` 进行识别。

```python
from cnocr import CnOcr

img_fp = './docs/examples/fanti.jpg'
ocr = CnOcr(rec_model_name='chinese_cht_PP-OCRv3')  # 识别模型使用繁体识别模型
out = ocr.ocr(img_fp)

print(out)
```

使用此模型时请注意以下问题：

* 识别精度一般，不是很好；

* 除了繁体字，对标点、英文、数字的识别都不好；

* 此模型不支持竖排文字的识别。

识别结果：
<div align="center">
  <img src="./docs/predict-outputs/fanti.jpg-result.jpg" alt="繁体中文识别" width="700px"/>
</div>


### 单行文字的图片识别

如果明确知道待识别的图片是单行文字图片（如下图），可以使用类函数 `CnOcr.ocr_for_single_line()` 进行识别。这样就省掉了文字检测的时间，速度会快一倍以上。

<div align="center">
  <img src="./docs/examples/helloworld.jpg" alt="单行文本识别" width="300px"/>
</div>
调用代码如下：

```python
from cnocr import CnOcr

img_fp = './docs/examples/helloworld.jpg'
ocr = CnOcr()
out = ocr.ocr_for_single_line(img_fp)
print(out)
```



### 更多应用示例

* **核酸疫苗截图识别**
<div align="center">
  <img src="./docs/predict-outputs/jiankangbao.jpeg-result.jpg" alt="核酸疫苗截图识别" width="500px"/>
</div>

* **身份证识别**
<div align="center">
  <img src="./docs/predict-outputs/aobama.webp-result.jpg" alt="身份证识别" width="700px"/>
</div>

* **饭店小票识别**
<div align="center">
  <img src="./docs/predict-outputs/fapiao.jpeg-result.jpg" alt="饭店小票识别" width="500px"/>
</div>
  

  

## 安装

嗯，顺利的话一行命令即可。

```bash
$ pip install cnocr[ort-cpu]
```

如果是 **GPU** 环境使用 ONNX 模型，请使用以下命令进行安装：

```bash
$ pip install cnocr[ort-gpu]
```



如果要训练自己的模型，，可以使用以下命令安装：

```bash
$ pip install cnocr[dev]
```



安装速度慢的话，可以指定国内的安装源，如使用阿里云的安装源：

```bash
$ pip install cnocr[ort-cpu] -i https://mirrors.aliyun.com/pypi/simple
```

> **Note** 
>
> 请使用 **Python3**（3.7.\*~3.10.\*之间的版本应该都行），没测过Python2下是否ok。

更多说明可见 [安装文档](https://cnocr.readthedocs.io/zh-cn/stable/install/)。

> **Warning** 
>
> 如果电脑中从未安装过 `PyTorch`，`OpenCV` python包，初次安装可能会遇到问题，但一般都是常见问题，可以自行百度/Google解决。



### Docker Image

可以从 [Docker Hub](https://hub.docker.com/u/breezedeus) 直接拉取已安装好 CnOCR 的镜像使用。

```bash
$ docker pull breezedeus/cnocr:latest
```

更多说明可见 [安装文档](https://cnocr.readthedocs.io/zh-cn/stable/install/)。



## HTTP服务

CnOCR **V2.2.1** 加入了基于 FastAPI 的HTTP服务。开启服务需要安装几个额外的包，可以使用以下命令安装：

```bash
pip install cnocr[serve]
```



安装完成后，可以通过以下命令启动HTTP服务（**`-p`** 后面的数字是**端口**，可以根据需要自行调整）：

```bash
cnocr serve -p 8501
```



服务开启后，可以使用以下方式调用服务。



### 命令行

比如待识别文件为 `docs/examples/huochepiao.jpeg`，如下使用 curl 调用服务：

```bash
> curl -F image=@docs/examples/huochepiao.jpeg http://0.0.0.0:8501/ocr
```



### Python

使用如下方式调用服务：

```python
import requests

image_fp = 'docs/examples/huochepiao.jpeg'
r = requests.post(
    'http://0.0.0.0:8501/ocr', files={'image': (image_fp, open(image_fp, 'rb'), 'image/png')},
)
ocr_out = r.json()['results']
print(ocr_out)
```



具体也可参考文件 [scripts/screenshot_daemon_with_server.py](scripts/screenshot_daemon_with_server.py) 。 



### 其他语言

请参照 curl 的调用方式自行实现。





## 可使用的模型

### 可使用的检测模型

具体参考 [CnSTD的下载说明](https://github.com/breezedeus/CnSTD?tab=readme-ov-file#%E5%B7%B2%E6%9C%89std%E6%A8%A1%E5%9E%8B)。

| `det_model_name`                                             | PyTorch 版本 | ONNX 版本 | 模型原始来源 | 模型文件大小 | 支持语言                       | 是否支持竖排文字识别 |
| ------------------------------------------------------------ | ------------ | --------- | ------------ | ------------ | ------------------------------ | -------------------- |
| db_shufflenet_v2                                             | √            | X         | cnocr        | 18 M         | 简体中文、繁体中文、英文、数字 | √                    |
| **db_shufflenet_v2_small**                                   | √            | X         | cnocr        | 12 M         | 简体中文、繁体中文、英文、数字 | √                    |
| db_mobilenet_v3                                              | √            | X         | cnocr        | 16 M         | 简体中文、繁体中文、英文、数字 | √                    |
| db_mobilenet_v3_small                                        | √            | X         | cnocr        | 7.9 M        | 简体中文、繁体中文、英文、数字 | √                    |
| db_resnet34                                                  | √            | X         | cnocr        | 86 M         | 简体中文、繁体中文、英文、数字 | √                    |
| db_resnet18                                                  | √            | X         | cnocr        | 47 M         | 简体中文、繁体中文、英文、数字 | √                    |
| ch_PP-OCRv4_det                                              | X            | √         | ppocr        | 4.5 M        | 简体中文、繁体中文、英文、数字 | √                    |
| ch_PP-OCRv4_det_server                                       | X            | √         | ppocr        | 108 M        | 简体中文、繁体中文、英文、数字 | √                    |
| ch_PP-OCRv3_det                                              | X            | √         | ppocr        | 2.3 M        | 简体中文、繁体中文、英文、数字 | √                    |
| **en_PP-OCRv3_det**                                          | X            | √         | ppocr        | 2.3 M        | **英文**、数字                 | √                    |



### 可使用的识别模型

相比于 CnOCR V2.2.* 版本，**V2.3** 中的大部分模型都经过了重新训练和精调，精度比旧版模型更高。同时，加入了两个参数量更多的模型系列：

  * `*-densenet_lite_246-gru_base`：优先供 **知识星球** [**CnOCR/CnSTD私享群**](https://t.zsxq.com/FEYZRJQ) 会员使用，后续会免费开源。
  * `*-densenet_lite_666-gru_large`：**Pro 模型**，购买后可使用。购买链接见文档：

**V2.3** 中的模型按使用场景可以分为以下几大类：

* `scene`：场景图片，适合识别一般拍照图片中的文字。此类模型以 `scene-` 开头，如模型 `scene-densenet_lite_136-gru`。
* `doc`：文档图片，适合识别规则文档的截图图片，如书籍扫描件等。此类模型以 `doc-` 开头，如模型 `doc-densenet_lite_136-gru`。
* `number`：仅识别**纯数字**（只能识别 `0~9` 十个数字）图片，适合银行卡号、身份证号等场景。此类模型以 `number-` 开头，如模型 `number-densenet_lite_136-gru`。
* `general`: 通用场景，适合图片无明显倾向的一般图片。此类模型无特定开头，与旧版模型名称保持一致，如模型 `densenet_lite_136-gru`。

> 注意 ⚠️：以上说明仅供参考，具体选择模型时建议以实际效果为准。

更多说明见：[可用模型](https://cnocr.readthedocs.io/zh-cn/stable/models/)。

| `rec_model_name`                                             | PyTorch 版本 | ONNX 版本 | 模型原始来源 | 模型文件大小 | 支持语言                            | 是否支持竖排文字识别 |
| ------------------------------------------------------------ | ------------ | --------- | ------------ | ------------ | ----------------------------------- | -------------------- |
| **densenet_lite_136-gru** 🆕                                  | √            | √         | cnocr        | 12 M         | 简体中文、英文、数字                | X                    |
| **scene-densenet_lite_136-gru** 🆕                            | √            | √         | cnocr        | 12 M         | 简体中文、英文、数字                | X                    |
| **doc-densenet_lite_136-gru** 🆕                              | √            | √         | cnocr        | 12 M         | 简体中文、英文、数字                | X                    |
| **densenet_lite_246-gru_base** 🆕 <br /> ([星球会员](https://t.zsxq.com/FEYZRJQ)专享) | √            | √         | cnocr        | 25 M         | 简体中文、英文、数字                | X                    |
| **scene-densenet_lite_246-gru_base** 🆕 <br /> ([星球会员](https://t.zsxq.com/FEYZRJQ)专享) | √            | √         | cnocr        | 25 M         | 简体中文、英文、数字                | X                    |
| **doc-densenet_lite_246-gru_base** 🆕 <br /> ([星球会员](https://t.zsxq.com/FEYZRJQ)专享) | √            | √         | cnocr        | 25 M         | 简体中文、英文、数字                | X                    |
| **densenet_lite_666-gru_large** 🆕 <br />（购买链接：[B站](https://gf.bilibili.com/item/detail/1104812055)、[Lemon Squeezy](https://ocr.lemonsqueezy.com/)） | √            | √         | cnocr        | 82 M         | 简体中文、英文、数字                | X                    |
| **scene-densenet_lite_666-gru_large** 🆕 <br />（购买链接：[B站](https://gf.bilibili.com/item/detail/1104815055)、[Lemon Squeezy](https://ocr.lemonsqueezy.com/)） | √            | √         | cnocr        | 82 M         | 简体中文、英文、数字                | X                    |
| **doc-densenet_lite_666-gru_large** 🆕 <br />（购买链接：[B站](https://gf.bilibili.com/item/detail/1104820055)、[Lemon Squeezy](https://ocr.lemonsqueezy.com/)） | √            | √         | cnocr        | 82 M         | 简体中文、英文、数字                | X                    |
| **number-densenet_lite_136-fc** 🆕                            | √            | √         | cnocr        | 2.7 M        | **纯数字**（仅包含 `0~9` 十个数字） | X                    |
| **number-densenet_lite_136-gru**  🆕 <br /> ([星球会员](https://t.zsxq.com/FEYZRJQ)专享) | √            | √         | cnocr        | 5.5 M        | **纯数字**（仅包含 `0~9` 十个数字） | X                    |
| **number-densenet_lite_666-gru_large** 🆕 <br />（购买链接：[B站](https://gf.bilibili.com/item/detail/1104055055)、[Lemon Squeezy](https://ocr.lemonsqueezy.com/)） | √            | √         | cnocr        | 55 M         | **纯数字**（仅包含 `0~9` 十个数字） | X                    |
| ch_PP-OCRv4                                                  | X            | √         | ppocr        | 10 M         | 简体中文、英文、数字                | √                    |
| ch_PP-OCRv4_server                                           | X            | √         | ppocr        | 86 M         | 简体中文、英文、数字                | √                    |
| ch_PP-OCRv3                                                  | X            | √         | ppocr        | 10 M         | 简体中文、英文、数字                | √                    |
| ch_ppocr_mobile_v2.0                                         | X            | √         | ppocr        | 4.2 M        | 简体中文、英文、数字                | √                    |
| en_PP-OCRv3                                                  | X            | √         | ppocr        | 8.5 M        | **英文**、数字                      | √                    |
| en_number_mobile_v2.0                                        | X            | √         | ppocr        | 1.8 M        | **英文**、数字                      | √                    |
| chinese_cht_PP-OCRv3                                         | X            | √         | ppocr        | 11 M         | **繁体中文**、英文、数字            | X                    |



## 未来工作

* [x] 支持图片包含多行文字 (`Done`)
* [x] crnn模型支持可变长预测，提升灵活性 (since `V1.0.0`)
* [x] 完善测试用例 (`Doing`)
* [x] 修bugs（目前代码还比较凌乱。。） (`Doing`)
* [x] 支持`空格`识别（since `V1.1.0`）
* [x] 尝试新模型，如 DenseNet，进一步提升识别准确率（since `V1.1.0`）
* [x] 优化训练集，去掉不合理的样本；在此基础上，重新训练各个模型
* [x] 由 MXNet 改为 PyTorch 架构（since `V2.0.0`）
* [x] 基于 PyTorch 训练更高效的模型
* [x] 支持列格式的文字识别
* [x] 打通与 [CnSTD](https://github.com/breezedeus/cnstd) 的无缝衔接（since `V2.2`）
* [ ] 模型精度进一步优化
* [ ] 支持更多的应用场景



## 给作者来杯咖啡

开源不易，如果此项目对您有帮助，可以考虑 [给作者加点油🥤，鼓鼓气💪🏻](https://cnocr.readthedocs.io/zh-cn/stable/buymeacoffee/) 。

---

官方代码库：[https://github.com/breezedeus/cnocr](https://github.com/breezedeus/cnocr)。
