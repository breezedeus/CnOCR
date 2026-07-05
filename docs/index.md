<figure markdown>
![CnOCR](figs/cnocr-logo.jpg){: style="width:180px"}
</figure>

# CnOCR
[![Discord](https://img.shields.io/discord/1200765964434821260?label=Discord)](https://discord.gg/GgD87WM8Tf)
[![Downloads](https://static.pepy.tech/personalized-badge/cnocr?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/cnocr)
[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fcnocr.readthedocs.io%2Fzh-cn%2Fstable%2F&label=Visitors&countColor=%23f5c791&style=flat&labelStyle=none)](https://visitorbadge.io/status?path=https%3A%2F%2Fcnocr.readthedocs.io%2Fzh-cn%2Fstable%2F)
[![license](https://img.shields.io/github/license/breezedeus/cnocr)](./LICENSE)
[![PyPI version](https://badge.fury.io/py/cnocr.svg)](https://badge.fury.io/py/cnocr)
[![forks](https://img.shields.io/github/forks/breezedeus/cnocr)](https://github.com/breezedeus/cnocr)
[![stars](https://img.shields.io/github/stars/breezedeus/cnocr)](https://github.com/breezedeus/cnocr)
![last-releast](https://img.shields.io/github/release-date/breezedeus/cnocr)
![last-commit](https://img.shields.io/github/last-commit/breezedeus/cnocr)
[![Twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2Fbreezedeus)](https://twitter.com/breezedeus)

<figure markdown>
[📖 使用](usage.md) |
[🛠️ 安装](install.md) |
[🧳 可用模型](models.md) |
[🕹 模型训练](train.md) |
[🛀🏻 在线Demo](demo.md) |
[💬 交流群](contact.md)

[English](https://github.com/breezedeus/cnocr/blob/master/README_en.md) | 中文
</figure>

[**CnOCR**](https://github.com/breezedeus/cnocr) 是 **Python 3** 下的**文字识别**（**Optical Character Recognition**，简称**OCR**）工具包，支持**简体中文**、**繁体中文**（部分模型）、**英文**和**数字**的常见字符识别，支持竖排文字的识别。自带了**20+个**[训练好的识别模型](models.md)，适用于不同应用场景，安装后即可直接使用。同时，CnOCR也提供简单的[训练命令](train.md)供使用者训练自己的模型。欢迎加入 [交流群](contact.md)。

作者也维护 **知识星球** [**CnOCR/CnSTD私享群**](https://t.zsxq.com/FEYZRJQ) ，欢迎加入。**知识星球私享群**会陆续发布一些CnOCR/CnSTD相关的私有资料，包括[**更详细的训练教程**](https://articles.zsxq.com/id_u6b4u0wrf46e.html)，**未公开的模型**，使用过程中遇到的难题解答等。本群也会发布OCR/STD相关的最新研究资料。此外，**私享群中作者每月提供两次免费特有数据的训练服务**。

可以使用 [**在线 Demo**](demo.md) 查看效果。

CnOCR的目标是**使用简单**。

## 最新更新

**V2.3.3** 支持 PP-OCRv6 多语种 OCR 模型，新增 `multi_PP-OCRv6_tiny`、`multi_PP-OCRv6`、`multi_PP-OCRv6_small`、`multi_PP-OCRv6_medium` 等识别模型，并通过 CnSTD 支持 `multi_PP-OCRv6_det_tiny`、`multi_PP-OCRv6_det_small`、`multi_PP-OCRv6_det_medium` 等检测模型。使用 PP-OCRv6 多语种模型时，可通过 `rec_lang_type` 指定识别语言，通过 `det_more_configs={'lang_type': ...}` 指定检测语言；更多说明见 [使用方法](usage.md) 和 [可用模型](models.md)。

## 安装简单

嗯，顺利的话一行命令即可完成安装。

```bash
$ pip install cnocr[ort-cpu]
```

更多说明可见 [安装文档](install.md)。

> **注**：如果电脑中从未安装过 `PyTorch`，`OpenCV` python包，初次安装可能会遇到问题，但一般都是常见问题，可以自行百度/Google解决。



### Docker Image

可以从 [Docker Hub](https://hub.docker.com/u/breezedeus) 直接拉取已安装好 CnOCR 的镜像使用。

```bash
$ docker pull breezedeus/cnocr:latest
```

更多说明可见 [安装文档](install.md)。



## 各种场景的调用示例

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

<figure markdown>
![火车票识别](predict-outputs/huochepiao.jpeg-result.jpg){: style="width:700px"}
</figure>




### 排版简单的印刷体截图图片识别

针对 **排版简单的印刷体文字图片**，如截图图片，扫描件图片等，可使用 `det_model_name='naive_det'`，相当于不使用文本检测模型，而使用简单的规则进行分行。

使用 `det_model_name='naive_det'` 的最大优势是**速度快**，劣势是对图片比较挑剔。如何判断是否该使用此检测模型呢？最简单的方式就是拿应用图片试试效果，效果好就用，不好就不用。

```python
from cnocr import CnOcr

img_fp = './docs/examples/multi-line_cn1.png'
ocr = CnOcr(det_model_name='naive_det') 
out = ocr.ocr(img_fp)

print(out)
```

识别结果：

<figure markdown>


| 图片                                                         | OCR结果                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![examples/multi-line_cn1.png](./examples/multi-line_cn1.png) | 网络支付并无本质的区别，因为<br />每一个手机号码和邮件地址背后<br />都会对应着一个账户--这个账<br />户可以是信用卡账户、借记卡账<br />户，也包括邮局汇款、手机代<br />收、电话代收、预付费卡和点卡<br />等多种形式。 |

</figure>


### 竖排文字识别

采用来自 [**PaddleOCR**](https://github.com/PaddlePaddle/PaddleOCR)（之后简称 **ppocr**）的 PP-OCRv6 多语种识别模型 `rec_model_name='multi_PP-OCRv6'` 进行识别。

```python
from cnocr import CnOcr

img_fp = './docs/examples/shupai.png'
ocr = CnOcr(rec_model_name='multi_PP-OCRv6')
out = ocr.ocr(img_fp)

print(out)
```

识别结果：

<figure markdown>
![竖排文字识别](./predict-outputs/shupai.png-result.jpg){: style="width:750px"}
</figure>



### 英文识别

虽然中文检测和识别模型也能识别英文，但**专为英文文字训练的检测器和识别器往往精度更高**。如果是纯英文的应用场景，建议使用来自 **ppocr** 的英文检测模型 `det_model_name='en_PP-OCRv3_det'`， 和英文识别模型 `rec_model_name='en_PP-OCRv3'` 。

也可以使用 PP-OCRv6 多语种模型，并显式指定英文：

```python
from cnocr import CnOcr

img_fp = './docs/examples/en_book1.jpeg'
ocr = CnOcr(
    rec_model_name='multi_PP-OCRv6',
    det_model_name='multi_PP-OCRv6_det_small',
    rec_lang_type='en',
    det_more_configs={'lang_type': 'en'},
)
out = ocr.ocr(img_fp)

print(out)
```

```python
from cnocr import CnOcr

img_fp = './docs/examples/en_book1.jpeg'
ocr = CnOcr(det_model_name='en_PP-OCRv3_det', rec_model_name='en_PP-OCRv3')
out = ocr.ocr(img_fp)

print(out)
```

识别结果：

<figure markdown>
![英文识别](./predict-outputs/en_book1.jpeg-result.jpg){: style="width:670px"}
</figure>



### 繁体中文识别

采用来自 ppocr 的 PP-OCRv6 多语种识别模型，并通过 `rec_lang_type='chinese_cht'` 指定繁体中文进行识别。

```python
from cnocr import CnOcr

img_fp = './docs/examples/fanti.jpg'
ocr = CnOcr(rec_model_name='multi_PP-OCRv6', rec_lang_type='chinese_cht')
out = ocr.ocr(img_fp)

print(out)
```

`multi_PP-OCRv6` 是 `multi_PP-OCRv6_small` 的别名；`chinese_cht` 是繁体中文对应的 `lang_type`。

识别结果：

<figure markdown>
![繁体中文识别](./predict-outputs/fanti.jpg-result.jpg){: style="width:700px"}
</figure>

注：上图中的识别结果来自 V3 模型；V6 模型的识别效果已经有显著增强。



### 单行文字的图片识别

如果明确知道待识别的图片是单行文字图片（如下图），可以使用类函数 `CnOcr.ocr_for_single_line()` 进行识别。这样就省掉了文字检测的时间，速度会快一倍以上。

<figure markdown>
![单行文本识别](./examples/helloworld.jpg){: style="width:270px"}
</figure>


调用代码如下：

```python
from cnocr import CnOcr

img_fp = './docs/examples/helloworld.jpg'
ocr = CnOcr()
out = ocr.ocr_for_single_line(img_fp)
print(out)
```

### 更多应用示例
- **核酸疫苗截图识别**
	<figure markdown>

 	![核酸疫苗截图识别](./predict-outputs/jiankangbao.jpeg-result.jpg){: style="width:600px"}
 	</figure>

- **身份证识别**
	<figure markdown>

 	![身份证识别](./predict-outputs/aobama.webp-result.jpg){: style="width:700px"}
 	</figure>

- **饭店小票识别**
	<figure markdown>
	![饭店小票识别](./predict-outputs/fapiao.jpeg-result.jpg){: style="width:550px"}
	</figure>



## HTTP服务

CnOCR 自 **V2.2.1** 开始加入了基于 **FastAPI** 的HTTP服务。开启服务需要安装几个额外的包，可以使用以下命令安装：

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



具体也可参考文件 [scripts/screenshot_daemon_with_server.py](https://github.com/breezedeus/CnOCR/tree/master/scripts/screenshot_daemon_with_server.py) 。 



### 其他语言

请参照 curl 的调用方式自行实现。



### Flask 服务

我们也提供了 **Flask** Server 的实现，见 [scripts/flask-serve.py](https://github.com/breezedeus/CnOCR/blob/master/scripts/flask-serve.py) 。下载此文件，然后安装 flask 后即可启动。



安装 Flask：

```bash
> pip install flask
```



启动服务：

```bash
> FLASK_APP=scripts/flask-serve.py flask run
```





## 其他文档

* [自己训练模型](train.md)
* [OCR技术介绍（PPT+视频）](std_ocr.md)
* [给作者加油](buymeacoffee.md)
* [FAQ](faq.md)
* [RELEASE文档](RELEASE.md)


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
* [ ] 支持更多的应用场景，如公式识别、表格识别、版面分析等

## 给作者加油鼓气

开源不易，如果此项目对您有帮助，可以考虑[给作者加点油🥤，鼓鼓气💪🏻](buymeacoffee.md) 。

---

官方代码库：[https://github.com/breezedeus/cnocr](https://github.com/breezedeus/cnocr)。
