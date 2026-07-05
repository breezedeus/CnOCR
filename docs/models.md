# 可用的模型

直接使用的模型都放在 [**cnstd-cnocr-models**](https://huggingface.co/breezedeus/cnstd-cnocr-models) 或对应的 `breezedeus/cnocr-ppocr-*`、`breezedeus/cnstd-ppocr-*` HuggingFace 模型仓库中，可免费下载使用。如果下载太慢，也可以从 [百度云盘](https://pan.baidu.com/s/1RhLBf8DcLnLuGLPrp89hUg?pwd=nocr) 下载， 提取码为 ` nocr`。具体方法可参考 [使用方法](usage.md) 。

模型分为两大类，1）来自 **[CnSTD](https://github.com/breezedeus/cnstd)** 的**检测模型**；2）来自 CnOCR 的**识别模型**。



## 检测模型

具体说明请参考 **[CnSTD 文档](https://github.com/breezedeus/CnSTD/tree/master#%E4%BD%BF%E7%94%A8%E6%96%B9%E6%B3%95)**，以下仅罗列出可用模型：

| `det_model_name`                                             | PyTorch 版本 | ONNX 版本 | 模型原始来源 | 模型文件大小 | 支持语言                       | 是否支持竖排文字识别 |
| ------------------------------------------------------------ | ------------ | --------- | ------------ | ------------ | ------------------------------ | -------------------- |
| db_shufflenet_v2                                             | √            | X         | cnocr        | 18 M         | 简体中文、繁体中文、英文、数字 | √                    |
| db_shufflenet_v2_small                                   | √            | X         | cnocr        | 12 M         | 简体中文、繁体中文、英文、数字 | √                    |
| db_mobilenet_v3                                              | √            | X         | cnocr        | 16 M         | 简体中文、繁体中文、英文、数字 | √                    |
| db_mobilenet_v3_small                                        | √            | X         | cnocr        | 7.9 M        | 简体中文、繁体中文、英文、数字 | √                    |
| db_resnet34                                                  | √            | X         | cnocr        | 86 M         | 简体中文、繁体中文、英文、数字 | √                    |
| db_resnet18                                                  | √            | X         | cnocr        | 47 M         | 简体中文、繁体中文、英文、数字 | √                    |
| multi_PP-OCRv6_det_tiny                                      | X            | √         | ppocr        | 1.7 M        | 多语种（不含日文）             | √                    |
| multi_PP-OCRv6_det_small                                     | X            | √         | ppocr        | 9.5 M        | 多语种                         | √                    |
| multi_PP-OCRv6_det_medium                                    | X            | √         | ppocr        | 59 M         | 多语种                         | √                    |
| ch_PP-OCRv5_det                                              | X            | √         | ppocr        | 4.6 M        | 简体中文、繁体中文、英文、数字 | √                    |
| ch_PP-OCRv5_det_server                                       | X            | √         | ppocr        | 84 M         | 简体中文、繁体中文、英文、数字 | √                    |
| ch_PP-OCRv4_det                                              | X            | √         | ppocr        | 4.5 M        | 简体中文、繁体中文、英文、数字 | √                    |
| ch_PP-OCRv4_det_server                                       | X            | √         | ppocr        | 108 M        | 简体中文、繁体中文、英文、数字 | √                    |
| ch_PP-OCRv3_det                                              | X            | √         | ppocr        | 2.3 M        | 简体中文、繁体中文、英文、数字 | √                    |
| **en_PP-OCRv3_det**                                          | X            | √         | ppocr        | 2.3 M        | **英文**、数字                 | √                    |


PP-OCRv6 的 `multi_PP-OCRv6_det_small` 和 `multi_PP-OCRv6_det_medium` 支持的 `lang_type` 包括：`ch`, `chinese_cht`, `en`, `japan`, `af`, `az`, `bs`, `ca`, `cs`, `cy`, `da`, `de`, `es`, `et`, `eu`, `fi`, `fr`, `ga`, `gl`, `hr`, `hu`, `id`, `is`, `it`, `ku`, `la`, `lb`, `lt`, `lv`, `mi`, `ms`, `mt`, `nl`, `no`, `oc`, `pl`, `pt`, `qu`, `rm`, `ro`, `rs_latin`, `sk`, `sl`, `sq`, `sv`, `sw`, `tl`, `tr`, `uz`, `vi`, `french`, `german`；`multi_PP-OCRv6_det_tiny` 不支持 `japan`。`multi` 是模型族名称，不是可传入的 `lang_type`。

> **Note**
>
> 列 **`PyTorch 版本`** 为 `√` 表示此模型支持 `det_model_backend=='pytorch'`；列 **`ONNX 版本`** 为 `√` 表示此模型支持 `det_model_backend=='onnx'`；取值为 `X` 则表示不支持对应的取值。

## 识别模型

CnOCR 自 **V2.1.2** 之后，可直接使用的识别模型包含两类：1）CnOCR 自己训练的模型，通常会包含 PyTorch 和 ONNX 版本；2）从其他ocr引擎搬运过来的训练好的外部模型，ONNX化后用于 CnOCR 中。

### 1) CnOCR 自己训练的模型

CnOCR **V2.3** 重新训练了所有的模型，模型较 V2.2.* 精度更高。V2.3 按使用场景把模型分为几大类场景：

* `scene`：场景图片，适合识别一般拍照图片中的文字。此类模型以 `scene-` 开头，如模型 `scene-densenet_lite_136-gru`。
* `doc`：文档图片，适合识别规则文档的截图图片，如书籍扫描件等。此类模型以 `doc-` 开头，如模型 `doc-densenet_lite_136-gru`。
* `number`：仅识别**纯数字**（只能识别 `0~9` 十个数字）图片，适合银行卡号、身份证号等场景。此类模型以 `number-` 开头，如模型 `number-densenet_lite_136-gru`。
* `general`: 通用场景，适合图片无明显倾向的一般图片。此类模型无特定开头，与旧版模型名称保持一致，如模型 `densenet_lite_136-gru`。

> 注意 ⚠️：以上说明仅为参考，具体选择模型时建议以实际效果为准。

同时，加入了两个参数量更多的模型系列：

  * `*-densenet_lite_246-gru_base`：优先供 **知识星球** [**CnOCR/CnSTD私享群**](https://t.zsxq.com/FEYZRJQ) 会员使用，后续会免费开源。
  * `*-densenet_lite_666-gru_large`：**Pro 模型**，购买后可使用。购买链接见文档：

CnOCR 自己训练的模型都支持**常见简体中文、英文和数字**的识别，大家也可以基于这些模型在自己的领域数据上继续精调模型。模型列表如下：

| `rec_model_name`                                             | PyTorch 版本 | ONNX 版本 | 模型原始来源 | 模型文件大小 | 支持语言                            | 是否支持竖排文字识别 |
| ------------------------------------------------------------ | ------------ | --------- | ------------ | ------------ | ----------------------------------- | -------------------- |
| **densenet_lite_136-gru** 🆕                                  | √            | √         | cnocr        | 12 M         | 简体中文、英文、数字                | X                    |
| **scene-densenet_lite_136-gru** 🆕                            | √            | √         | cnocr        | 12 M         | 简体中文、英文、数字                | X                    |
| **doc-densenet_lite_136-gru** 🆕                              | √            | √         | cnocr        | 12 M         | 简体中文、英文、数字                | X                    |
| **densenet_lite_246-gru_base** 🆕 <br /> ([星球会员](https://t.zsxq.com/FEYZRJQ)专享) | √            | √         | cnocr        | 25 M         | 简体中文、英文、数字                | X                    |
| **scene-densenet_lite_246-gru_base** 🆕 <br /> ([星球会员](https://t.zsxq.com/FEYZRJQ)专享) | √            | √         | cnocr        | 25 M         | 简体中文、英文、数字                | X                    |
| **doc-densenet_lite_246-gru_base** 🆕 <br /> ([星球会员](https://t.zsxq.com/FEYZRJQ)专享) | √            | √         | cnocr        | 25 M         | 简体中文、英文、数字                | X                    |
| **densenet_lite_666-gru_large** 🆕 <br />（购买链接：[B站](https://mall.bilibili.com/neul-next/detailuniversal/detail.html?isMerchant=1&page=detailuniversal_detail&saleType=10&itemsId=11884138&loadingShow=1&noTitleBar=1&msource=merchant_share)、[Lemon Squeezy](https://ocr.lemonsqueezy.com/)） | √            | √         | cnocr        | 82 M         | 简体中文、英文、数字                | X                    |
| **scene-densenet_lite_666-gru_large** 🆕 <br />（购买链接：[B站](https://mall.bilibili.com/neul-next/detailuniversal/detail.html?isMerchant=1&page=detailuniversal_detail&saleType=10&itemsId=11883935&loadingShow=1&noTitleBar=1&msource=merchant_share)、[Lemon Squeezy](https://ocr.lemonsqueezy.com/)） | √            | √         | cnocr        | 82 M         | 简体中文、英文、数字                | X                    |
| **doc-densenet_lite_666-gru_large** 🆕 <br />（购买链接：[B站](https://mall.bilibili.com/neul-next/detailuniversal/detail.html?isMerchant=1&page=detailuniversal_detail&saleType=10&itemsId=11883965&loadingShow=1&noTitleBar=1&msource=merchant_share)、[Lemon Squeezy](https://ocr.lemonsqueezy.com/)） | √            | √         | cnocr        | 82 M         | 简体中文、英文、数字                | X                    |
| **number-densenet_lite_136-fc** 🆕                            | √            | √         | cnocr        | 2.7 M        | **纯数字**（仅包含 `0~9` 十个数字） | X                    |
| **number-densenet_lite_136-gru**  🆕 <br /> ([星球会员](https://t.zsxq.com/FEYZRJQ)专享) | √            | √         | cnocr        | 5.5 M        | **纯数字**（仅包含 `0~9` 十个数字） | X                    |
| **number-densenet_lite_666-gru_large** 🆕 <br />（购买链接：[B站](https://mall.bilibili.com/neul-next/detailuniversal/detail.html?isMerchant=1&page=detailuniversal_detail&saleType=10&itemsId=11884155&loadingShow=1&noTitleBar=1&msource=merchant_share)、[Lemon Squeezy](https://ocr.lemonsqueezy.com/)） | √            | √         | cnocr        | 55 M         | **纯数字**（仅包含 `0~9` 十个数字） | X                    |


一些说明：

1. 模型名称是由**局部编码**模型和**序列编码**模型名称拼接而成，以符合"-"分割，如 `densenet_lite_136-gru`。如果是特定应用场景，则前面还增加了场景名称，如 `scene-densenet_lite_136-gru`。
2. 列 **`PyTorch 版本`** 为 `√` 表示此模型支持 `model_backend=='pytorch'`；列 **`ONNX 版本`** 为 `√` 表示此模型支持 `model_backend=='onnx'`；取值为 `X` 则表示不支持对应的取值。

CnOCR 的自有模型从结构上可以分为两阶段：第一阶段是获得ocr图片的局部编码向量，第二部分是对局部编码向量进行序列学习，获得序列编码向量。目前的PyTorch版本的两个阶段分别包含以下模型：

1. 局部编码模型（emb model）
   - **`densenet_lite_<numbers>`**：一个微型的`densenet`网络；其中的`<number>`表示模型中每个block包含的层数。
   - **`densenet`**：一个小型的`densenet`网络；
2. 序列编码模型（seq model）
   - **`fc`**：两层的全连接网络；
   - **`gru`**：一层的GRU网络；
   - **`lstm`**：一层的LSTM网络。

### 2) 外部模型

以下模型是 [**PaddleOCR**](https://github.com/PaddlePaddle/PaddleOCR) 中模型的 **ONNX** 版本，所以不会依赖 **PaddlePaddle** 相关工具包，故而也不支持基于这些模型在自己的领域数据上继续精调模型。这些模型应该都支持**竖排文字**。

| `model_name`          | PyTorch 版本 | ONNX 版本 | 支持语言                 | 是否支持竖排文字识别 | 模型文件大小 |
| --------------------- | ------------ | --------- | ------------------------ | -------------------- | ------------ |
| multi_PP-OCRv6_tiny   | X            | √         | 多语种（不含日文）       | √                    | 4.3 M        |
| multi_PP-OCRv6 / multi_PP-OCRv6_small | X | √       | 多语种                   | √                    | 20 M         |
| multi_PP-OCRv6_medium | X            | √         | 多语种                   | √                    | 73 M         |
| ch_PP-OCRv5           | X            | √         | 简体中文、英文、数字     | √                    | 16 M         | 
| ch_PP-OCRv5_server    | X            | √         | 简体中文、英文、数字     | √                    | 81 M         | 
| ch_PP-OCRv4           | X            | √         | 简体中文、英文、数字     | √                    | 10 M         | 
| ch_PP-OCRv4_server    | X            | √         | 简体中文、英文、数字     | √                    | 86 M         | 
| ch_PP-OCRv3           | X            | √         | 简体中文、英文、数字     | √                    | 10 M         |
| ch_ppocr_mobile_v2.0  | X            | √         | 简体中文、英文、数字     | √                    | 4.2 M        |
| en_PP-OCRv3           | X            | √         | **英文**、数字           | √                    | 8.5 M        |
| en_PP-OCRv4           | X            | √         | **英文**、数字           | √                    | 8.6 M        |
| en_number_mobile_v2.0 | X            | √         | **英文**、数字           | √                    | 1.8 M        |
| chinese_cht_PP-OCRv3  | X            | √         | **繁体中文**、英文、数字 | X                    | 11 M         |
| japan_PP-OCRv3        | X            | √         | **日文**、英文、数字     | √                    | 9.6 M         |
| korean_PP-OCRv3       | X            | √         | **韩文**、英文、数字     | √                    | 9.4 M         |
| latin_PP-OCRv3        | X            | √         | **拉丁文**、英文、数字   | √                    | 8.6 M         |
| arabic_PP-OCRv3       | X            | √         | **阿拉伯文**、英文、数字 | √                    | 8.6 M         |

PP-OCRv6 的 `multi_PP-OCRv6_small`、`multi_PP-OCRv6_medium` 支持的 `lang_type` 包括：`ch`, `chinese_cht`, `en`, `japan`, `af`, `az`, `bs`, `ca`, `cs`, `cy`, `da`, `de`, `es`, `et`, `eu`, `fi`, `fr`, `ga`, `gl`, `hr`, `hu`, `id`, `is`, `it`, `ku`, `la`, `lb`, `lt`, `lv`, `mi`, `ms`, `mt`, `nl`, `no`, `oc`, `pl`, `pt`, `qu`, `rm`, `ro`, `rs_latin`, `sk`, `sl`, `sq`, `sv`, `sw`, `tl`, `tr`, `uz`, `vi`, `french`, `german`；`multi_PP-OCRv6_tiny` 不支持 `japan`。`multi_PP-OCRv6` 是 `multi_PP-OCRv6_small` 的别名；`multi` 是模型族名称，不是可传入的 `lang_type`。

更多模型可参考 [PaddleOCR/models_list.md](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/doc/doc_ch/models_list.md) 。如有其他外语识别需求，可在 **知识星球** [**CnOCR/CnSTD私享群**](https://t.zsxq.com/FEYZRJQ) 中向作者提出建议。
