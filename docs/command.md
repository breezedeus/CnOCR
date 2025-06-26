# 脚本工具

**cnocr** 包含了几个命令行工具，安装 **cnocr** 后即可使用。

## 图片预测

使用命令 **`cnocr predict`** 预测单个文件或文件夹中所有图片，以下是使用说明：

```bash
$ cnocr predict -h
Usage: cnocr predict [OPTIONS]

Options:
  -m, --rec-model-name TEXT       识别模型名称。默认值为 densenet_lite_136-gru
  -b, --rec-model-backend [pytorch|onnx]
                                  识别模型类型。默认值为 `onnx`
  -v, --rec-vocab-fp TEXT         识别模型使用的词表。默认取值为 `None` 表示使用系统设定的词表
  -d, --det-model-name TEXT       检测模型名称。默认值为 ch_PP-OCRv5_det
  --det-model-backend [pytorch|onnx]
                                  检测模型类型。默认值为 `onnx`
  -p, --pretrained-model-fp TEXT  识别模型使用训练好的模型。默认为 `None`，表示使用系统自带的预训练模型
  -c, --context TEXT              使用cpu还是 `gpu` 运行代码，也可指定为特定gpu，如`cuda:0`。默认为
                                  `cpu`
  -i, --img-file-or-dir TEXT      输入图片的文件路径或者指定的文件夹  [required]
  -s, --single-line               是否输入图片只包含单行文字。对包含单行文字的图片，不做按行切分；否则会先对图片按行分割后
                                  再进行识别
  --draw-results-dir TEXT         画出的检测与识别效果图所存放的目录；取值为 `None` 表示不画图
  --draw-font-path TEXT           画出检测与识别效果图时使用的字体文件
  --verbose                       是否打印详细日志信息。默认值为 `False`
  -h, --help                      Show this message and exit.
```

例如可以使用以下命令对图片 `docs/examples/rand_cn1.png` 进行文字识别：

```bash
$ cnocr predict -i docs/examples/rand_cn1.png -s
```

具体使用也可参考文件 [Makefile](https://github.com/breezedeus/cnocr/blob/master/Makefile) 。

## 模型评估

使用命令 **`cnocr evaluate`** 在指定的数据集上评估模型效果，以下是使用说明：

```bash
$ cnocr evaluate -h
Usage: cnocr evaluate [OPTIONS]

  评估模型效果。检测模型使用 `det_model_name='naive_det'` 。

Options:
  -m, --rec-model-name TEXT       识别模型名称。默认值为 densenet_lite_136-gru
  -b, --rec-model-backend [pytorch|onnx]
                                  识别模型类型。默认值为 `onnx`
  -v, --rec-vocab-fp TEXT         识别模型使用的词表。默认取值为 `None` 表示使用系统设定的词表
  -p, --pretrained-model-fp TEXT  识别模型使用训练好的模型。默认为 `None`，表示使用系统自带的预训练模型
  -c, --context TEXT              使用cpu还是 `gpu` 运行代码，也可指定为特定gpu，如`cuda:0`。默认为
                                  `cpu`
  -i, --eval-index-fp TEXT        待评估文件所在的索引文件，格式与训练时训练集索引文件相同，每行格式为 `<图片路径>
                                  <以空格分割的labels>`
  --image-folder TEXT             图片所在文件夹，相对于索引文件中记录的图片位置  [required]
  --batch-size INTEGER            batch size. 默认值：128
  -o, --output-dir TEXT           存放评估结果的文件夹。默认值：`eval_results`
  --verbose                       whether to print details to screen
  -h, --help                      Show this message and exit.
```

例如可以使用以下命令评估 `data/test/dev.tsv` 中指定的所有样本：

```bash
$ cnocr evaluate -i data/test/dev.tsv --img-folder data/images 
```

具体使用也可参考文件 [Makefile](https://github.com/breezedeus/cnocr/blob/master/Makefile) 。

## 模型训练

使用命令 **`cnocr train`**  训练文本检测模型，以下是使用说明：

```bash
$ cnocr train -h
Usage: cnocr train [OPTIONS]

  训练识别模型

Options:
  -m, --rec-model-name TEXT       识别模型名称。默认值为 `densenet_lite_136-gru`
  -i, --index-dir TEXT            索引文件所在的文件夹，会读取文件夹中的 train.tsv 和 dev.tsv 文件
                                  [required]
  --train-config-fp TEXT          识别模型训练使用的json配置文件，参考
                                  `docs/examples/train_config.json`
                                  [required]
  --finetuning                    是否为精调模式（精调模式使用更温柔的transform）。默认为 `False`
  -r, --resume-from-checkpoint TEXT
                                  恢复此前中断的训练状态，继续训练识别模型。所以文件中应该包含训练状态。默认为
                                  `None`
  -p, --pretrained-model-fp TEXT  导入的训练好的识别模型，作为模型初始值。优先级低于"--resume-from-
                                  checkpoint"，当传入"--resume-from-
                                  checkpoint"时，此传入失效。默认为 `None`
  -h, --help                      Show this message and exit.
```

例如可以使用以下命令进行训练：

```bash
$ cnocr train -m densenet_lite_136-gru --index-dir data/test --train-config-fp docs/examples/train_config.json
```

训练数据的格式见文件夹 [data/test](https://github.com/breezedeus/cnocr/blob/master/data/test) 中的 [train.tsv](https://github.com/breezedeus/cnocr/blob/master/data/test/train.tsv) 和 [dev.tsv](https://github.com/breezedeus/cnocr/blob/master/data/test/dev.tsv) 文件。

具体使用也可参考文件 [Makefile](https://github.com/breezedeus/cnocr/blob/master/Makefile) 。

## 模型API服务

CnOCR 自 **V2.2.1** 开始加入了基于 FastAPI 的HTTP服务。开启服务需要安装几个额外的包，可以使用以下命令安装：

```bash
> pip install cnocr[serve]
```



使用命令 **`cnocr serve`**  启动API服务，以下是使用说明：

```bash
$ cnocr serve -h
Usage: cnocr serve [OPTIONS]

  开启HTTP服务。

Options:
  -H, --host TEXT     server host. Default: "0.0.0.0"
  -p, --port INTEGER  server port. Default: 8501
  --reload            whether to reload the server when the codes have been
                      changed
  -h, --help          Show this message and exit.
```



例如使用以下命令启动服务：

```bash
$ cnocr serve -p 8501
```



服务调用方式参考 [HTTP服务](index.md) 。



## 模型转存

训练好的模型会存储训练状态，使用命令 **`cnocr resave`**  去掉与预测无关的数据，降低模型大小。

```bash
$ cnocr resave -h
Usage: cnocr resave [OPTIONS]

  训练好的识别模型会存储训练状态，使用此命令去掉预测时无关的数据，降低模型大小

Options:
  -i, --input-model-fp TEXT   输入的识别模型文件路径  [required]
  -o, --output-model-fp TEXT  输出的识别模型文件路径  [required]
  -h, --help                  Show this message and exit.
```

示例：

```bash
$ cnocr resave -i cnocr-v2.3-densenet_lite_136-gru-epoch=005.ckpt -o cnocr-v2.3-densenet_lite_136-gru-epoch=005-model.ckpt
```

## PyTorch 模型导出为 ONNX 模型

把训练好的模型导出为 ONNX 格式。

```bash
$ cnocr export-onnx -h
Usage: cnocr export-onnx [OPTIONS]

  把训练好的识别模型导出为 ONNX 格式。

Options:
  -m, --rec-model-name TEXT   识别模型名称。默认值为 `densenet_lite_136-gru`
  -v, --rec-vocab-fp TEXT     识别模型使用的词表。默认取值为 `None` 表示使用系统设定的词表
  -i, --input-model-fp TEXT   输入的识别模型文件路径。 默认为 `None`，表示使用系统自带的预训练模型
  -o, --output-model-fp TEXT  输出的识别模型文件路径（.onnx）  [required]
  -h, --help                  Show this message and exit.
```

示例：

```bash
$ cnocr export-onnx -m densenet_lite_136-gru -i cnocr-v2.3-densenet_lite_136-gru-epoch=005-model.ckpt -o cnocr-v2.3-densenet_lite_136-gru-epoch=005-model.onnx
```
