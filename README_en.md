<div align="center">
  <img src="./docs/figs/cnocr-logo.jpg" width="250px"/>
  <div>&nbsp;</div>

[![Discord](https://img.shields.io/discord/1200765964434821260?label=Discord)](https://discord.gg/GgD87WM8Tf)
[![Downloads](https://static.pepy.tech/personalized-badge/cnocr?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://cnocr.readthedocs.io/zh-cn/stable/)
[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fbreezedeus%2FCnOCR&label=Visitors&countColor=%23f5c791&style=flat&labelStyle=none)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2Fbreezedeus%2FCnOCR)
[![license](https://img.shields.io/github/license/breezedeus/cnocr)](./LICENSE)
[![Docs](https://readthedocs.org/projects/cnocr/badge/?version=latest)](https://cnocr.readthedocs.io/zh-cn/stable/?badge=latest)
[![PyPI version](https://badge.fury.io/py/cnocr.svg)](https://badge.fury.io/py/cnocr)
[![forks](https://img.shields.io/github/forks/breezedeus/cnocr)](https://github.com/breezedeus/cnocr)
[![stars](https://img.shields.io/github/stars/breezedeus/cnocr)](https://github.com/breezedeus/cnocr)
![last-release](https://img.shields.io/github/release-date/breezedeus/cnocr)
![last-commit](https://img.shields.io/github/last-commit/breezedeus/cnocr)
[![Twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2Fbreezedeus)](https://twitter.com/breezedeus)

[📖 Doc](https://cnocr.readthedocs.io/zh-cn/stable/) |
[🛠️ Install](https://cnocr.readthedocs.io/zh-cn/stable/install/) |
[🧳 Models](https://cnocr.readthedocs.io/zh-cn/stable/models/) |
[🕹 Training](https://cnocr.readthedocs.io/zh-cn/stable/train/) |
[🛀🏻 Online Demo](https://share.streamlit.io/breezedeus/cnstd/st-deploy/cnstd/app.py) |
[💬 Contact](https://www.breezedeus.com/article/join-group)

</div>

<div align="center">

 [中文](./README.md) | English

</div>

# CnOCR
<div align="center">
<strong>Tech should serve the people, not enslave them!</strong>
<br>
<strong>Please do NOT use this project for text censorship!</strong>
<br>
---
</div>

### [Update 2026.07.04]: Release of V2.3.3

Major Changes:

* Added RapidOCR-based PP-OCRv6 multilingual OCR models
  * New PP-OCRv6 recognition models: `multi_PP-OCRv6_tiny`, `multi_PP-OCRv6`, `multi_PP-OCRv6_small`, and `multi_PP-OCRv6_medium`
  * New PP-OCRv6 detection models through CnSTD: `multi_PP-OCRv6_det_tiny`, `multi_PP-OCRv6_det_small`, and `multi_PP-OCRv6_det_medium`
  * Added `rec_lang_type` to `CnOcr` for setting the language type used by RapidOCR v6 recognition models
  * Added the CLI options `--rec-lang-type` and `--det-lang-type` for setting the language type used by RapidOCR v6 models


### [Update 2024.11.30]: Release of V2.3.1

Major Changes:

* Based on RapidOCR, integrate the latest version of PPOCRv4 OCR models, providing more model options
  * Add support for PP-OCRv4 recognition models, including standard and server versions
* Modify the implementation of reading files to support Chinese paths on Windows
* Fix bug: When using multiple processes, the transform_func cannot be serialized
* Fix bug: Compatible with albumentations=1.4.*

### [Update 2023.12.24]: Release of V2.3

Major Changes:

* All models have been retrained, offering higher accuracy than the previous version.
* Models are categorized into several types based on usage scenarios (see [Pre-trained Recognition Models](#pre-trained-recognition-models)):
  * `scene`: For scene images, suitable for recognizing text in general photography. Models in this category start with `scene-`, such as the `scene-densenet_lite_136-gru` model.
  * `doc`: For document images, suitable for recognizing text in regular document screenshots, like scanned book pages. Models in this category start with `doc-`, such as the `doc-densenet_lite_136-gru` model.
  * `number`: Specifically for recognizing **only numbers** (able to recognize only the ten digits `0~9`), suitable for scenarios like bank card numbers, ID numbers, etc. Models in this category start with `number-`, such as the `number-densenet_lite_136-gru` model.
  * `general`: For general scenarios, suitable for images without a clear preference. Models in this category do not have a specific prefix and maintain the same naming convention as older versions, such as the `densenet_lite_136-gru` model.
  > Note ⚠️: The above descriptions are for reference only. It is recommended to choose models based on actual performance.
* Two larger series of models have been added:
  * `*-densenet_lite_246-gru_base`: Initially available for **Planet of Knowledge** [**CnOCR/CnSTD Private Group**](https://t.zsxq.com/FEYZRJQ) members, to be open-sourced for free after one month.
  * `*-densenet_lite_666-gru_large`: Pro models, available for use after purchase. Purchase link can be found at [https://ocr.lemonsqueezy.com](https://ocr.lemonsqueezy.com/).

For more details, please refer to: [CnOCR V2.3 New Release: Better, More, and Larger Models | Breezedeus.com](https://www.breezedeus.com/article/cnocr-v2.3-better-more).

[**CnOCR**](https://github.com/breezedeus/cnocr)  is an **Optical Character Recognition (OCR)** toolkit for **Python 3**. It supports recognition of common characters in **English and numbers**, **Simplified Chinese**, **Traditional Chinese** (some models), and **vertical text** recognition. It comes with [**20+ well-trained models**](https://cnocr.readthedocs.io/zh-cn/stable/models/) for different application scenarios and can be used directly after installation. Also, CnOCR provides simple training [commands](https://cnocr.readthedocs.io/zh-cn/stable/train/) for users to train their own models. Welcome to join the WeChat contact group.

<div align="center">
  <img src="https://huggingface.co/datasets/breezedeus/cnocr-wx-qr-code/resolve/main/wx-qr-code.JPG" alt="WeChat Group" width="300px"/>
</div>

The author also maintains **Planet of Knowledge** [**CnOCR/CnSTD Private Group**](https://t.zsxq.com/FEYZRJQ), 
where questions are more likely to receive prompt responses from the author. You are welcome to join. Knowledge Planet Members can enjoy the following benefits:

* Access to download some non-open source paid models for free.
* A 20% discount on the purchase of all other paid models.
* Rapid responses from the author to various difficulties encountered during usage.
* The author offers two free training services per month using unique data.
* The group will continuously release exclusive materials related to CnOCR/CnSTD.
* The group will regularly publish the latest research materials related to OCR, STD, CV, and more.


## Documentation

See [CnOCR online documentation](https://cnocr.readthedocs.io/) , in Chinese.

## Usage

Starting from **V2.2**, **CnOCR** internally uses the text detection engine **[CnSTD](https://github.com/breezedeus/cnstd)** for text detection and positioning. So **CnOCR** V2.2 can recognize not only typographically simple printed text images, such as screenshot images, scanned copies, etc., but also **scene text in general images**.

Here are some examples of usages for different scenarios.


## Usages for Different Scenarios

### Common image recognition

Just use default values for all parameters. If you find that the result is not good enough, try different parameters more to see the effect, and usually you will end up with a more desirable accuracy.

```python
from cnocr import CnOcr

img_fp = './docs/examples/huochepiao.jpeg'
ocr = CnOcr()  # Use default values for all parameters
out = ocr.ocr(img_fp)

print(out)
```

Recognition results:

<div align="center">
  <img src="./docs/predict-outputs/huochepiao.jpeg-result.jpg" alt="Train ticket recognition" width="800px"/>
</div>



### English Recognition

Although Chinese detection and recognition models can also recognize English, **detectors and recognizers trained specifically for English texts tend to be more accurate**. For English-only application scenarios, it is recommended to use the English detection model `det_model_name='en_PP-OCRv3_det'` and the English recognition model `rec_model_name='en_PP-OCRv3'` from  [**PaddleOCR**](https://github.com/PaddlePaddle/PaddleOCR) (also called **ppocr**).

```python
from cnocr import CnOcr

img_fp = './docs/examples/en_book1.jpeg'
ocr = CnOcr(det_model_name='en_PP-OCRv3_det', rec_model_name='en_PP-OCRv3')
out = ocr.ocr(img_fp)

print(out)
```

Recognition results:

<div align="center">
  <img src="./docs/predict-outputs/en_book1.jpeg-result.jpg" alt="Recognition of English texts" width="600px"/>
</div>


### Recognition of typographically screenshot images

For **typographically simple typographic text images**, such as screenshot images, scanned images, etc., you can use `det_model_name='naive_det'`, which is equivalent to not using a text detection model, but using simple rules for branching.

> **Note**
>
> `det_model_name='naive_det'` is equivalent to CnOCR versions before `V2.2` (`V2.0.*`, `V2.1.*`).

The biggest advantage of using `det_model_name='naive_det'` is that the speech is **fast** and the disadvantage is that it is picky about images. How do you determine if you should use the detection model `'naive_det'`? The easiest way is to take your application image and try the effect, if it works well, use it, if not, don't.

```python
from cnocr import CnOcr

img_fp = './docs/examples/multi-line_cn1.png'
ocr = CnOcr(det_model_name='naive_det') 
out = ocr.ocr(img_fp)

print(out)
```

Recognition results:

<div align="center">

| 图片                                                                      | OCR结果                                                                                                                         |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| ![docs/examples/multi-line_cn1.png](./docs/examples/multi-line_cn1.png) | 网络支付并无本质的区别，因为<br />每一个手机号码和邮件地址背后<br />都会对应着一个账户--这个账<br />户可以是信用卡账户、借记卡账<br />户，也包括邮局汇款、手机代<br />收、电话代收、预付费卡和点卡<br />等多种形式。 |

</div>

### Vertical text recognition

The PP-OCRv6 multilingual recognition model `rec_model_name='multi_PP-OCRv6'` from **ppocr** is used for recognition.

```python
from cnocr import CnOcr

img_fp = './docs/examples/shupai.png'
ocr = CnOcr(rec_model_name='multi_PP-OCRv6')
out = ocr.ocr(img_fp)

print(out)
```

Recognition results:

<div align="center">
  <img src="./docs/predict-outputs/shupai.png-result.jpg" alt="vertical text recognition" width="800px"/>
</div>

### Traditional Chinese Recognition

Use the PP-OCRv6 multilingual recognition model from ppocr, and set `rec_lang_type='chinese_cht'` for Traditional Chinese.

```python
from cnocr import CnOcr

img_fp = './docs/examples/fanti.jpg'
ocr = CnOcr(rec_model_name='multi_PP-OCRv6', rec_lang_type='chinese_cht')
out = ocr.ocr(img_fp)

print(out)
```

`multi_PP-OCRv6` is an alias of `multi_PP-OCRv6_small`; `chinese_cht` is the `lang_type` for Traditional Chinese.

<div align="center">
  <img src="./docs/predict-outputs/fanti.jpg-result.jpg" alt="traditional Chinese recognition" width="700px"/>
</div>

Note: The recognition result shown above was produced by a V3 model. The V6 model has significantly improved recognition quality.

### Single line text image recognition

If it is clear that the image to be recognized is a single line text image (as shown below), you can use the class function `CnOcr.ocr_for_single_line()` for recognition. This saves the time of text detection and will be more than twice as fast.

<div align="center">
  <img src="./docs/examples/helloworld.jpg" alt="single line text image recognition" width="300px"/>
</div>

The code is as follows:

```python
from cnocr import CnOcr

img_fp = './docs/examples/helloworld.jpg'
ocr = CnOcr()
out = ocr.ocr_for_single_line(img_fp)
print(out)
```

### More Applications

* **Recognition of Vaccine App Screenshot**
<div align="center">
  <img src="./docs/predict-outputs/jiankangbao.jpeg-result.jpg" alt="Recognition of Vaccine App Screenshot" width="500px"/>
</div>

* **Recognition of ID Card**
<div align="center">
  <img src="./docs/predict-outputs/aobama.webp-result.jpg" alt="Recognition of ID Card" width="700px"/>
</div>

* **Recognition of Restaurant Ticket**
<div align="center">
  <img src="./docs/predict-outputs/fapiao.jpeg-result.jpg" alt="Recognition of Restaurant Ticket" width="500px"/>
</div>
  

## Install

Well, one line of command is enough if it goes well.

```bash
$ pip install cnocr[ort-cpu]
```

If you are using a **GPU** environment with an ONNX model, please install using the following command:

```bash
$ pip install cnocr[ort-gpu]
```

If you want to train new models with your own data, please install using the following command:

```bash
$ pip install cnocr[dev]
```

If the installation is slow, you can specify a domestic installation source, such as using the Aliyun source: 

```bash
$ pip install cnocr[ort-cpu] -i https://mirrors.aliyun.com/pypi/simple
```

> **Note** 
>
> Please use **Python 3.8 or later**.

More instructions can be found in the [installation documentation](https://cnocr.readthedocs.io/zh-cn/stable/install/) (in Chinese).

> **Warning** 
>
> If you have never installed `PyTorch`, `OpenCV` python packages on your computer, you may encounter problems with the first installation, but they are usually common problems that can be solved by Baidu/Google.


### Docker Image

You can directly pull the image with CnOCR installed from Docker Hub: [Docker Hub](https://hub.docker.com/u/breezedeus) . 

```bash
$ docker pull breezedeus/cnocr:latest
```

More instructions can be found in the [installation documentation](https://cnocr.readthedocs.io/zh-cn/stable/install/) (in Chinese).


## Pre-trained Models

### Pre-trained Detection Models

Refer to [CnSTD](https://github.com/breezedeus/CnSTD?tab=readme-ov-file#%E5%B7%B2%E6%9C%89std%E6%A8%A1%E5%9E%8B) for details.

| `det_model_name`                                             | PyTorch Version | ONNX Version | Model original source | Model File Size | Supported Language                       | Whether to support vertical text detection |
| ------------------------------------------------------------ | ------------ | --------- | ------------ | ------------ | ------------------------------ | -------------------- |
| **en_PP-OCRv3_det**                                          | X            | √         | ppocr        | 2.3 M        | **English**、Numbers  | √                    |
| db_shufflenet_v2                                             | √            | X         | cnocr        | 18 M         | Simplified Chinese, Traditional Chinese, English, Numbers | √                    |
| db_shufflenet_v2_small                                   | √            | X         | cnocr        | 12 M         | Simplified Chinese, Traditional Chinese, English, Numbers | √                    |
| db_mobilenet_v3                                              | √            | X         | cnocr        | 16 M         | Simplified Chinese, Traditional Chinese, English, Numbers | √                    |
| db_mobilenet_v3_small                                        | √            | X         | cnocr        | 7.9 M        | Simplified Chinese, Traditional Chinese, English, Numbers | √                    |
| db_resnet34                                                  | √            | X         | cnocr        | 86 M         | Simplified Chinese, Traditional Chinese, English, Numbers | √                    |
| db_resnet18                                                  | √            | X         | cnocr        | 47 M         | Simplified Chinese, Traditional Chinese, English, Numbers | √                    |
| multi_PP-OCRv6_det_tiny                                      | X            | √         | ppocr        | 1.7 M        | Multilingual, except Japanese | √                    |
| multi_PP-OCRv6_det_small                                     | X            | √         | ppocr        | 9.5 M        | Multilingual | √                    |
| multi_PP-OCRv6_det_medium                                    | X            | √         | ppocr        | 59 M         | Multilingual | √                    |
| ch_PP-OCRv5_det                                              | X            | √         | ppocr        | 4.6 M        | Simplified Chinese, Traditional Chinese, English, Numbers | √                    |
| ch_PP-OCRv5_det_server                                       | X            | √         | ppocr        | 84 M         | Simplified Chinese, Traditional Chinese, English, Numbers | √                    |
| ch_PP-OCRv4_det                                              | X            | √         | ppocr        | 4.5 M        | Simplified Chinese, Traditional Chinese, English, Numbers | √                    |
| ch_PP-OCRv4_det_server                                       | X            | √         | ppocr        | 108 M        | Simplified Chinese, Traditional Chinese, English, Numbers | √                    |
| ch_PP-OCRv3_det                                              | X            | √         | ppocr        | 2.3 M        | Simplified Chinese, Traditional Chinese, English, Numbers | √                    |

For PP-OCRv6, `multi_PP-OCRv6_det_small` and `multi_PP-OCRv6_det_medium` support these `lang_type` values: `ch`, `chinese_cht`, `en`, `japan`, `af`, `az`, `bs`, `ca`, `cs`, `cy`, `da`, `de`, `es`, `et`, `eu`, `fi`, `fr`, `ga`, `gl`, `hr`, `hu`, `id`, `is`, `it`, `ku`, `la`, `lb`, `lt`, `lv`, `mi`, `ms`, `mt`, `nl`, `no`, `oc`, `pl`, `pt`, `qu`, `rm`, `ro`, `rs_latin`, `sk`, `sl`, `sq`, `sv`, `sw`, `tl`, `tr`, `uz`, `vi`, `french`, and `german`. `multi_PP-OCRv6_det_tiny` does not support `japan`. `multi` is the model family name, not a valid `lang_type`.



### Pre-trained Recognition Models

Compared to the CnOCR V2.2.* versions, most models in **V2.3** have been retrained and fine-tuned, offering higher accuracy than the older versions. Additionally, two series of models with larger parameter volumes have been added:

  * `*-densenet_lite_246-gru_base`: Currently available for **Knowledge Planet** [**CnOCR/CnSTD Private Group**](https://t.zsxq.com/FEYZRJQ) members, it will be open-sourced afterward.
  * `*-densenet_lite_666-gru_large`: **Pro models**, available for use after purchase. The purchase link: [https://ocr.lemonsqueezy.com](https://ocr.lemonsqueezy.com).

Models in **V2.3** are categorized into the following types based on usage scenarios:

* `scene`: For scene images, suitable for recognizing text in general photography. Models in this category start with `scene-`, such as the `scene-densenet_lite_136-gru` model.
* `doc`: For document images, suitable for recognizing text in regular document screenshots, like scanned book pages. Models in this category start with `doc-`, such as the `doc-densenet_lite_136-gru` model.
* `number`: Specifically for recognizing **only numbers** (able to recognize only the ten digits `0~9`), suitable for scenarios like bank card numbers, ID numbers, etc. Models in this category start with `number-`, such as the `number-densenet_lite_136-gru` model.
* `general`: For general scenarios, suitable for images without a clear preference. Models in this category do not have a specific prefix and maintain the same naming convention as older versions, such as the `densenet_lite_136-gru` model.

> Note ⚠️: The above descriptions are for reference only. It is recommended to choose models based on actual performance.

For more details, see: [Available Models](https://cnocr.readthedocs.io/zh-cn/stable/models/).

| `rec_model_name`                                                                                                      | PyTorch Version | ONNX Version | Model original source | Model File Size | Supported Language                       | Whether to support vertical text recognition |
|-----------------------------------------------------------------------------------------------------------------------|-----------------| --------- | ------------ | ------------ | ------------------------ | -------------------- |
| densenet_lite_136-gru 🆕                                                                                          | √               | √         | cnocr        | 12 M         | Simplified Chinese, English, Numbers | X                    |
| **scene-densenet_lite_136-gru** 🆕                                                                                    | √               | √         | cnocr        | 12 M         | Simplified Chinese, English, Numbers | X                    |
| **doc-densenet_lite_136-gru** 🆕                                                                                      | √               | √         | cnocr        | 12 M         | Simplified Chinese, English, Numbers | X                    |
| **densenet_lite_246-gru_base** 🆕 <br /> ([Planet Members](https://t.zsxq.com/FEYZRJQ) Only)       | √               | √         | cnocr        | 25 M         | Simplified Chinese, English, Numbers | X                    |
| **scene-densenet_lite_246-gru_base** 🆕 <br /> ([Planet Members](https://t.zsxq.com/FEYZRJQ) Only) | √               | √         | cnocr        | 25 M         | Simplified Chinese, English, Numbers | X                    |
| **doc-densenet_lite_246-gru_base** 🆕 <br /> ([Planet Members](https://t.zsxq.com/FEYZRJQ) Only)   | √               | √         | cnocr        | 25 M         | Simplified Chinese, English, Numbers | X                    |
| **densenet_lite_666-gru_large** 🆕 <br /> ([Purchase Link](https://ocr.lemonsqueezy.com))                             | √              | √         | cnocr        | 82 M         | Simplified Chinese, English, Numbers | X                    |
| **scene-densenet_lite_666-gru_large** 🆕 <br /> ([Purchase Link](https://ocr.lemonsqueezy.com))                       | √              | √         | cnocr        | 82 M         | Simplified Chinese, English, Numbers | X                    |
| **doc-densenet_lite_666-gru_large** 🆕 <br /> ([Purchase Link](https://ocr.lemonsqueezy.com))                         | √              | √         | cnocr        | 82 M         | Simplified Chinese, English, Numbers | X                    |
| **number-densenet_lite_136-fc** 🆕                                                                                    | √               | √  | cnocr        | 2.7 M        | **Pure Numeric** (contains only the ten digits `0~9`) | X                    |
| **number-densenet_lite_136-gru**  🆕 <br /> ([Planet Members](https://t.zsxq.com/FEYZRJQ) Only)                       | √               | √         | cnocr        | 5.5 M       | **Pure Numeric** (contains only the ten digits `0~9`)  | X                    |
| **number-densenet_lite_666-gru_large** 🆕 <br /> ([Purchase Link](https://ocr.lemonsqueezy.com))                      | √               | √         | cnocr        | 56 M      | **Pure Numeric** (contains only the ten digits `0~9`)  | X                    |
| multi_PP-OCRv6_tiny                                          | X            | √         | ppocr        | 4.3 M        | Multilingual, except Japanese | √                    |
| multi_PP-OCRv6 / multi_PP-OCRv6_small                        | X            | √         | ppocr        | 20 M         | Multilingual | √                    |
| multi_PP-OCRv6_medium                                        | X            | √         | ppocr        | 73 M         | Multilingual | √                    |
| ch_PP-OCRv5                                                  | X            | √         | ppocr        | 16 M         | Simplified Chinese, English, Numbers  | √                    |
| ch_PP-OCRv5_server                                           | X            | √         | ppocr        | 81 M         | Simplified Chinese, English, Numbers | √                    |
| ch_PP-OCRv4                                                  | X            | √         | ppocr        | 10 M         | Simplified Chinese, English, Numbers  | √                    |
| ch_PP-OCRv4_server                                           | X            | √         | ppocr        | 86 M         | Simplified Chinese, English, Numbers | √                    |
| ch_PP-OCRv3                                                 | X               | √         | ppocr        | 10 M         | Simplified Chinese, English, Numbers | √                    |
| ch_ppocr_mobile_v2.0                                        | X               | √         | ppocr        | 4.2 M        | Simplified Chinese, English, Numbers | √                    |
| **en_PP-OCRv3**                                             | X               | √         | ppocr        | 8.5 M        | **English**、Numbers | √                    |
| **en_PP-OCRv4**                                             | X            | √         | ppocr        | 8.6 M        | **English**、Numbers | √                    |
| **en_number_mobile_v2.0**                                   | X               | √         | ppocr        | 1.8 M        | **English**、Numbers | √                    |
| **chinese_cht_PP-OCRv3**                                    | X               | √         | ppocr        | 11 M         | **Traditional Chinese**, English, Numbers | X     |
| **japan_PP-OCRv3**                                                                                                    | X               | √         | ppocr        | 9.6 M         | **Japanese**, English, Numbers | √     |
| **korean_PP-OCRv3**                                                                                                   | X               | √         | ppocr        | 9.4 M         | **Korean**, English, Numbers | √     |
| **latin_PP-OCRv3**                                                                                                    | X               | √         | ppocr        | 8.6 M         | **Latin**, English, Numbers | √     |
| **arabic_PP-OCRv3**                                                                                                   | X               | √         | ppocr        | 8.6 M         | **Arabic**, English, Numbers | √     |

For PP-OCRv6, `multi_PP-OCRv6_small` and `multi_PP-OCRv6_medium` support these `lang_type` values: `ch`, `chinese_cht`, `en`, `japan`, `af`, `az`, `bs`, `ca`, `cs`, `cy`, `da`, `de`, `es`, `et`, `eu`, `fi`, `fr`, `ga`, `gl`, `hr`, `hu`, `id`, `is`, `it`, `ku`, `la`, `lb`, `lt`, `lv`, `mi`, `ms`, `mt`, `nl`, `no`, `oc`, `pl`, `pt`, `qu`, `rm`, `ro`, `rs_latin`, `sk`, `sl`, `sq`, `sv`, `sw`, `tl`, `tr`, `uz`, `vi`, `french`, and `german`. `multi_PP-OCRv6_tiny` does not support `japan`. `multi_PP-OCRv6` is an alias of `multi_PP-OCRv6_small`; `multi` is the model family name, not a valid `lang_type`.


## Future work

* [x] Support for images containing multiple lines of text (`Done`)
* [x] crnn model support for variable length prediction, improving flexibility (since `V1.0.0`)
* [x] Refine test cases (`Doing`)
* [x] Fix bugs (The code is still messy.) (`Doing`)
* [x] Support `space` recognition (since `V1.1.0`)
* [x] Try new models like DenseNet to further improve recognition accuracy (since `V1.1.0`)
* [x] Optimize the training set to remove unreasonable samples; based on this, retrain each model
* [x] Change from MXNet to PyTorch architecture (since `V2.0.0`)
* [x] Train more efficient models based on PyTorch
* [x] Support text recognition in column format  (since `V2.1.2`)
* [x] Integration with [CnSTD](https://github.com/breezedeus/cnstd) (since `V2.2`)
* [ ] Further optimization of model accuracy
* [ ] Support more application scenarios, such as formula recognition, table recognition, layout analysis, etc.


## A cup of coffee for the author

It is not easy to maintain and evolve the project, so if it is helpful to you, please consider [offering the author a cup of coffee 🥤](https://cnocr.readthedocs.io/zh-cn/stable/buymeacoffee/).

---

Official code base: [https://github.com/breezedeus/cnocr](https://github.com/breezedeus/cnocr). Please cite it properly.
