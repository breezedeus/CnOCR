# Release Notes

### Update 2026.07.04：发布 V2.3.3

Major Changes:

* Added RapidOCR-based PP-OCRv6 multilingual OCR models.
  * Recognition models: `multi_PP-OCRv6_tiny`, `multi_PP-OCRv6`, `multi_PP-OCRv6_small`, and `multi_PP-OCRv6_medium`.
  * Detection models through CnSTD: `multi_PP-OCRv6_det_tiny`, `multi_PP-OCRv6_det_small`, and `multi_PP-OCRv6_det_medium`.
  * Python API: `CnOcr(rec_lang_type=...)` for RapidOCR v6 recognition language selection.
  * CLI options: `--rec-lang-type` and `--det-lang-type` for RapidOCR v6 language selection.
* Bump dependency: `cnstd>=1.2.8`.

主要变更：

* 基于 RapidOCR 支持 PP-OCRv6 多语种 OCR 模型。
  * 识别模型：`multi_PP-OCRv6_tiny`、`multi_PP-OCRv6`、`multi_PP-OCRv6_small` 和 `multi_PP-OCRv6_medium`。
  * 通过 CnSTD 支持检测模型：`multi_PP-OCRv6_det_tiny`、`multi_PP-OCRv6_det_small` 和 `multi_PP-OCRv6_det_medium`。
  * Python API 新增 `CnOcr(rec_lang_type=...)`，可为 RapidOCR v6 识别模型指定语言类型。
  * CLI 新增 `--rec-lang-type` 和 `--det-lang-type`，可为 RapidOCR v6 模型指定语言类型。
* 依赖升级：`cnstd>=1.2.8`。

### Update 2026.02.07：发布 V2.3.2.3

Major Changes:

* Bump dependency: `cnstd>=1.2.7.1`.

主要变更：

* 依赖升级：`cnstd>=1.2.7.1`。

### Update 2025.09.21：发布 V2.3.2.2

Major Changes:

* Fix bug: https://github.com/breezedeus/CnOCR/pull/365. Thanks to [@wangsrGit119](https://github.com/wangsrGit119).

主要变更：

* 修复bug: https://github.com/breezedeus/CnOCR/pull/365 。感谢 [@wangsrGit119](https://github.com/wangsrGit119) 。

### Update 2025.06.28：发布 V2.3.2.1

Major Changes:

* Bug fixes.

主要变更：

* 修复bug。

### Update 2025.06.26：发布 V2.3.2

Major Changes:

* Integrated the latest PPOCRv5 text recognition
  * Added support for PP-OCRv5 recognition models: `ch_PP-OCRv5` and `ch_PP-OCRv5_server`

主要变更：

* 集成 PPOCRv5 最新版 OCR 模型
  * 新增支持 PP-OCRv5 识别模型：`ch_PP-OCRv5` 和 `ch_PP-OCRv5_server`

### Update 2024.11.30：发布 V2.3.1

主要变更：

* 基于 RapidOCR 集成 PPOCRv4 最新版 OCR 模型，提供更多的模型选择
  * 新增支持 PP-OCRv4 检测和识别模型，包括标准版和服务器版
  * 新增多语言OCR模型支持：
    * chinese_cht_PP-OCRv3：繁体中文识别
    * japan_PP-OCRv3：日文识别
    * korean_PP-OCRv3：韩文识别
    * latin_PP-OCRv3：拉丁文识别
    * arabic_PP-OCRv3：阿拉伯文识别
    * en_PP-OCRv4：英文识别（v4版本）
* 修改读文件实现方式，支持 Windows 的中文路径
* 修复Bug：当使用多个进程时，transform_func 无法序列化
* 修复Bug：与 albumentations=1.4.* 兼容

Major Changes:

* Based on RapidOCR, integrate the latest version of PPOCRv4 OCR models, providing more model options
  * Add support for PP-OCRv4 detection and recognition models, including standard and server versions
  * Add multilingual OCR model support:
    * chinese_cht_PP-OCRv3: Traditional Chinese recognition
    * japan_PP-OCRv3: Japanese recognition
    * korean_PP-OCRv3: Korean recognition
    * latin_PP-OCRv3: Latin recognition
    * arabic_PP-OCRv3: Arabic recognition
    * en_PP-OCRv4: English recognition (v4 version)
* Modify the implementation of reading files to support Chinese paths on Windows
* Fix bug: When using multiple processes, the transform_func cannot be serialized
* Fix bug: Compatible with albumentations=1.4.*

### Update 2024.06.22：发布 V2.3.0.3

主要变更：

* 修复文档中 broken 的链接。

### Update 2024.04.10：发布 V2.3.0.2

主要变更：

* CN OSS 不可用了，默认下载模型地址由 `CN` 改为 `HF`。

### Update 2023.12.26：发布 V2.3.0.1

主要变更：

* 修复使用 ppocr ONNX 模型时传入的 providers 参数的逻辑问题。

### Update 2023.12.24：发布 V2.3

主要变更：

* 重新训练了所有的模型，比上一版精度更高。
* 模型按使用场景分为 3 大类场景：
  * `scene`：场景图片，适合识别一般拍照图片中的文字。
  * `doc`：文档图片，适合识别规则文档的截图图片，如书籍扫描件等。
  * `general`: 通用场景，适合图片无明显倾向的一般图片。
  > 注意 ⚠️：以上说明仅供参考，具体选择模型时建议以实际效果为准。
* 加入了两个更大的系列模型：
  * `*-densenet_lite_246-gru_base`：优先供 **知识星球** [**CnOCR/CnSTD私享群**](https://t.zsxq.com/FEYZRJQ) 会员使用，一个月后会免费开源。
  * `*-densenet_lite_666-gru_large`：Pro 模型，购买后可使用。

更多细节请参考：[CnOCR V2.3 新版发布：模型更好、更多、更大 | Breezedeus.com](https://www.breezedeus.com/article/cnocr-v2.3-better-more)。

### Update 2023.10.09：发布 V2.2.4.2

主要变更：

* 支持基于环境变量 `CNOCR_DOWNLOAD_SOURCE` 的取值，来决定不同的模型下载路径。

### Update 2023.10.01：发布 V2.2.4.1

主要变更：

* 加入了纯数字识别系列模型 `number-*` 中的大模型 `number-densenet_lite_666-gru_large`，购买后可使用。具体说明见：[CnOCR 纯数字识别新模型 | Breezedeus.com](https://www.breezedeus.com/article/cnocr-number-model-20231001) 。

### Update 2023.09.27：发布 V2.2.4

主要变更：

* 加入了纯数字识别系列模型 `number-*`，可用于纯数字识别场景，如银行卡识别、身份证识别、硬币年份识别等；
* 对各个包的新版做了接口适配，如 `pytorch_lightning`、`onnxruntime`、`pillow`等；
* 优化了训练过程使用的数据增强方式，并借鉴了**Nougat** 中的数据增强方法；
* 增加了对更大模型的支持，如 `densenet-lite-666`、`gru_large` 等；
* 以前的 `*-gru` 系列模型，现在也有 ONNX 版了；
* 修复了一堆的bugs，如 `val-complete_match-epoch` 训练过程一直为 `0` 等。

### Update 2023.06.30：发布 V2.2.3

主要变更：

* 修复了模型文件自动下载的功能。HuggingFace似乎对下载文件的逻辑做了调整，导致之前版本的自动下载失败，当前版本已修复。但由于HuggingFace国内被墙，国内下载仍需 **梯子（VPN）**。
* 更新了各个依赖包的版本号。


### Update 2023.02.11: 发布 cnocr V2.2.2.2

主要变更：

* 修复了识别很窄图片时异常的问题。
* 修复了对 torchvision 0.14 的兼容问题。


### Update 2022.10.30: 发布 cnocr V2.2.2.1

主要变更：

* 修复了与新版 torch 和 torchvision 不兼容的问题。


### Update 2022.09.09: 发布 cnocr V2.2.2

主要变更：

* 修复HTTP服务存在的问题，感谢 [@Sugobet](https://github.com/Sugobet) 。

* 增加图片分类模型，以及配套的训练和预测脚本，具体见 [图片分类工具](clf_command.md)。

* 适配了新版的pytorch_lightning接口，训练中引入`torchmetrics`计算各种指标。

  

### Update 2022.08.21: 发布 cnocr V2.2.1

主要变更：

* 修复了一些bug。
* 加入了基于 FastAPI 的HTTP服务，使用命令 `cnocr serve` 启动HTTP服务，具体见 [安装说明](install.md)。
* 加入了一些工具脚本，如对截屏图片进行OCR，具体见[cnocr/scripts](https://github.com/breezedeus/CnOCR/tree/master/scripts)。



### Update 2022.07.25: 发布 cnocr V2.2

主要变更：

* CnOCR 内部集成 [CnSTD](https://github.com/breezedeus/cnstd) 进行文本检测，降低使用门槛，提升适用场景的范围。
* 对诸多代码做了重构，同时也对文档进行了大幅度的优化。
* 更新了测试用例，清理了过期的用例。


### Update 2022.05.27: 发布 cnocr V2.1.2.1

主要变更：

* 修复 V2.1.2 bug：打包时忘记把 ppocr 模型相关的字符集文件打包进来了 😭。

### Update 2022.05.25: 发布 cnocr V2.1.2

主要变更：

- 引入了对外部模型的支持，此版加入了对 PaddleOCR 模型的 **ONNX** 版本的支持，具体参见 [可用模型](models.md)；
- 新引入的模型支持识别竖排文字、繁体中文（部分模型），具体参见 [可用模型](models.md)。
- 模型输出结果的格式略有调整，具体参见 [使用方法](usage.md)。

### Update 2022.05.15: 发布 cnocr V2.1.1.1

主要变更：

- 增加了对 **ONNX** 模型的支持，支持 **`*-fc`** 模型，提升预测速度；
- 类 `CnOcr` 的初始化中增加了参数 `model_backend` 和 `vocab_fp`，具体参见 [使用方法](usage.md) ；
- 增加了 `cnocr export-onnx` 命令，把训练好的PyTorch模型导出为ONNX模型；
- 去掉了对包 `python-Levenshtein` 的依赖。

### Update 2021.11.06: 发布 cnocr V2.1.0

主要变更：

* 使用了更精简的模型架构：`densenet_lite_*`；
* 使用了更丰富的数据重新训练了所有模型，精度相较于之前版本更高；
* 提供了更多预训练好的模型；
* 加入了 `cnocr evaluate` 命令以评估效果。

### Update 2021.09.21: 发布 cnocr V2.0.1

主要变更：

* 重新训练了模型，模型识别精度略有提升；
* 函数 `CnOcr.ocr_for_single_lines(img_list, batch_size=1)` 中加入了 `batch_size` 参数。

### Update 2021.08.26: 发布 cnocr V2.0.0

主要变更：

* MXNet 越来越小众化，故从基于 MXNet 的实现转为基于 **PyTorch** 的实现；
* 重新实现了识别模型，优化了训练数据，重新训练模型；
* 优化了能识别的字符集合；
* 优化了对英文的识别效果；
* 优化了对场景文字的识别效果；
* 使用接口略有调整，请谨慎更新。

### Update 2021.08.24: 发布 cnocr V1.2.3

主要变更：

* 更改了模型的默认下载urls；
* 依赖中去掉了对numpy的约束。

### Update 2020.05.29: 发布 cnocr V1.2.2

主要变更：

* `CnOcr`加入类函数 `CnOcr.set_cand_alphabet(cand_alphabet) `。可通过此类函数设置`cand_alphabet`。这样同一个实例也可以指定不同的`cand_alphabet`进行识别。
* bugfix:
  * 修复同时初始化多个实例时会报错的问题。

### Update 2020.05.25: 发布 cnocr V1.2.1

主要变更：

* bugfix:
  * 修复了zip文件名的typo。

### Update 2020.05.25: 发布 cnocr V1.2.0

主要变更：

* 优化了对数字识别的准确度。
* 优化了模型结构，进一步降低了模型的大小，提升了预测速度；最小模型从原来的`6.8M`降为`4.7M`。
* 使用了[爱因互动 Ein+](https://einplus.cn)自己的CDN存储模型文件，下载速度超快。
* 提供了预测速度更快的 `shorter (-s)`版预训练模型：`densenet-lite-s-gru`和`densenet-lite-s-fc`。
* 默认模型由之前的`conv-lite-fc`改为`densenet-lite-fc`。
* 预测支持使用GPU。
* bugfixs:
  * Web 调用时的内存泄露。感谢 [@myuanz](https://github.com/myuanz)；
  * 输入图片宽度很小时导致异常；
  * 去掉  `f-print`。

### Update 2020.04.21: 发布 cnocr V1.1.0

V1.1.0对代码做了很大改动，重写了大部分训练的代码，也生成了更多更难的训练和测试数据。训练好的模型相较于之前版本的模型精度有显著提升，尤其是针对英文单词的识别。

以下列出了主要的变更：

* 更新了训练代码，使用mxnet的`recordio`首先把数据转换成二进制格式，提升后续的训练效率。训练时支持对图片做实时数据增强。也加入了更多可传入的参数。

* **允许训练集中的文字数量不同，目前是中文10个字，英文20个字母。**

* 提供了更多的模型选择，允许大家按需训练多种不同大小的识别模型。

* 内置了各种训练好的模型，最小的模型只有之前模型的`1/5`大小。所有模型都可免费使用。

* 相较于之前版本的模型，新的模型精度有显著提升，尤其是针对英文单词的识别。**新模型已经可以识别英文单词间的空格。**

* **支持文字识别只在给定字符集中进行。** 对于一些纯数字或者纯英文字母的应用场景可以带来识别率提升。

* 优化了对黑底白字多行文字图片的支持。

* mxnet依赖升级到更新的版本了。很多人反馈mxnet `1.4.1`经常找不到没法装，现在升级到`>=1.5.0,<1.7.0`。

### Update 2019.07.25: 发布 cnocr V1.0.0

`cnocr`发布了预测效率更高的新版本v1.0.0。**新版本的模型跟以前版本的模型不兼容**。所以如果大家是升级的话，需要重新下载最新的模型文件。具体说明见下面（流程和原来相同）。

主要改动如下：

- **crnn模型支持可变长预测，提升预测效率**
- 支持利用特定数据对现有模型进行精调（继续训练）
- 修复bugs，如训练时`accuracy`一直为`0`
- 依赖的 `mxnet` 版本从`1.3.1`更新至 `1.4.1`
