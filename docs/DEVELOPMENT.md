# 开发与发布

## 本地开发

建议使用 Python 3.10。CPU 测试不依赖原始数据、模型或 CUDA；GPU 集成测试需要 Jetson、数据和对应模型。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e '.[test,calibration,detector]'
.venv/bin/python -m pytest -q
python3 scripts/check_release.py
```

GPU 测试入口见 [Jetson 部署](JETSON.md)。测试覆盖数据有效性、几何变换、场景 BA、推理后端、缓冲区生命周期和并发队列行为。

性能更改需要保持数据版本、模型输入约定和标定来源可追溯，同时比较逐帧预测、几何结果、无效组/丢帧数和完整视频 wall time。具体指标定义见 [性能说明](BENCHMARKS.md)。

## 仓库内容

公开仓库包含源代码、运行配置、通用测试、部署文档、完整演示/动态预览及两个发布配置的性能证据。

原始数据、上游模型仓库、checkpoint/ONNX/engine、Python 环境和运行生成物保存在 `data/`、`models/weights/`、`results/` 等忽略目录。新的报告应保存到 `reports/local/`，只将经过审查的最终结果放进 `reports/published/`。

## 发布检查

```bash
python3 scripts/check_release.py
python3 scripts/package_release.py
```

检查包括公开文件集合、私有地址/绝对路径/凭据模式、相对文档链接、媒体哈希和循环元数据、6 份计时 CSV 的分位数与吞吐汇总。

打包输出 `results/releases/jetson-stereo-perception-source.zip` 及 SHA-256/文件清单，包含公开源码、文档、演示媒体和性能证据。视频与 GIF 为普通仓库文件，无需 Git LFS。

原创代码采用 [MIT](../LICENSE)。数据派生媒体需保留 [第三方声明](../THIRD_PARTY.md)；上游模型源码和权重不随公开包分发。
