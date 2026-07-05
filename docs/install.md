# 安装

## pip 安装

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


安装速度慢的话，可以指定国内的安装源，如使用豆瓣源：

```bash
$ pip install cnocr[ort-cpu] -i https://mirrors.aliyun.com/pypi/simple
```

> **Note**
>
> 请使用 **Python 3.8 或更高版本**。



> **Warning** 
>
> 如果电脑中从未安装过 `PyTorch`，`OpenCV` python包，初次安装可能会遇到问题，但一般都是常见问题，可以自行百度/Google解决。



### 包含API接口安装

CnOCR 自 **V2.2.1** 开始加入了基于 FastAPI 的HTTP服务。开启服务需要安装几个额外的包，可以使用以下命令安装：

```bash
$ pip install cnocr[serve] onnxruntime
```



安装完成后，可以通过以下命令启动HTTP服务（**`-p`** 后面的数字是**端口**，可以根据需要自行调整）：

```bash
$ cnocr serve -p 8501
```



服务的调用方式请参考 [首页/HTTP服务](index.md) 。



如遇到安装问题，欢迎在 [Github](https://github.com/breezedeus/cnocr)、 知识星球[**CnOCR/CnSTD私享群**](https://t.zsxq.com/FEYZRJQ) 或者 微信交流群反馈给作者 [breezedeus](https://github.com/breezedeus) 。



## Docker Image

可以从 [Docker Hub](https://hub.docker.com/u/breezedeus) 直接拉取已安装好 CnOCR 的镜像使用。

```bash
$ docker pull breezedeus/cnocr:latest
```

**注意** ⚠️ ：如果你的系统架构是 ARM（如 Mac with Apple Silicon），需要拉取 ARM 版本的镜像：

```bash
$ docker pull breezedeus/cnocr-arm64:latest
```


利用以下命令启动容器：

```bash
$ docker run -it -p 8501:8501 breezedeus/cnocr:latest bash
```



容器启动后会**自动启动HTTP服务**，在容器外可以直接调用相应的服务：

```bash
$ curl -F image=@docs/examples/huochepiao.jpeg http://0.0.0.0:8501/ocr
```

**注意** ⚠️ ：有些情况下容器启动后不会自动启动HTTP服务（原因未知），需要进入容器手动启动。
可以通过以下命令进入容器：

```bash
$ docker exec -it <container_id> bash
```

使用以下命令查看服务是否正常启动：

```bash
$ ps afx | grep uvicorn
```

如果返回结果中存在 `/usr/local/bin/python /usr/local/bin/uvicorn serve:app --host 0.0.0.0 --port 8501` 这种行，则说明服务已启动；
否则说明服务未自动启动。此时可以通过以下命令手动启动服务：

```bash
$ nohup cnocr serve -H 0.0.0.0 -p 8501 &
```


更详细的调用方式见 [HTTP服务](index.md) 。



## GPU 环境使用 ONNX 模型

默认情况下安装的 **ONNX** 包是 **`onnxruntime`**，它只能在 `CPU` 上运行。如果需要在 `GPU` 环境使用 **ONNX** 模型，需要卸载此包，然后安装包 **`onnxruntime-gpu`** 。

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```
