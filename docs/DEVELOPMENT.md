# 开发与公开发布

## CPU 开发

建议 Python 3.10；其他版本需确认依赖支持。CPU 测试不需要下载视频数据、模型或 CUDA，GPU 测试会明确跳过：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e '.[test,calibration,detector]'
.venv/bin/python -m pytest -q
python3 scripts/check_release.py
```

CUDA/TensorRT 的真实测试需要 Jetson、数据和对应模型，见 [部署说明](JETSON.md)。CPU 测试通过不能证明 GPU 性能或相机精度。

保持固定数据版本、模型输入约定、标定哈希和报告计时定义。性能更改需要同时比较逐帧预测、几何结果、丢帧/无效组数和完整视频 wall time，不能只给模型推理耗时。

本次发布整理已在目标 Jetson 的新建运行 venv 中验证：CPU 53 通过 / 9 跳过，GPU 62 通过；完整 296 帧 H.264 回放与原 FP32 参考逐帧一致。模型从新的上游 Git 检出重新导出，ONNX 哈希与原模型相同。数据和目标 engine 复用已有文件，未重新安装 JetPack 或重建 engine。记录见 [发布验证](../reports/published/release_validation.json)。

## 文件边界

源代码、配置、文档、两段演示和精选计时证据随仓库发布。数据、上游模型仓库、checkpoint/ONNX/engine、虚拟环境和生成视频保存在 `data/`、`models/weights/`、`results/`，由 Git 忽略。机器环境转储和新 runtime 报告同样默认忽略。

本次整理前的原始文件保存在维护者本地 `results/local-history/pre-github-20261126/`，有 SHA-256 清单；历史交接包也保留。这些本地备份不影响新克隆部署，不包含在公开包。

## 发布前检查和打包

```bash
python3 scripts/check_release.py
python3 scripts/package_release.py
```

检查包括：候选文件白名单、凭据/私有地址/个人绝对路径模式、相对 Markdown 链接、视频大小和 SHA-256、公开证据来源哈希、9 份 CSV 的分位数和吞吐汇总。扫描是针对公开工作树的辅助检查，不是对过去 Git 历史的全面审计。

打包产生 `results/releases/jetson-stereo-perception-source.zip` 和旁边的 SHA-256/文件清单；不包含 `.git`、本地数据、模型和历史备份。解压后仍可直接执行发布检查。可把解压内容放进新仓库，或检查现有仓库的 `git diff` 后自行提交。如果继续使用旧 Git 历史，旧提交中的文件仍会保留；本工具不重写历史。

视频已经是普通仓库文件，不需要 Git LFS。若网页上传保留目录结构不方便，可用 Git 客户端提交。自动化工具没有替你创建远程仓库、commit 或 push。

[MIT](../LICENSE) 仅适用于原创代码。公开视频必须连同 [第三方说明](../THIRD_PARTY.md) 和数据许可一起保留；LCAFNet 的源码/权重不包含在公开包。
