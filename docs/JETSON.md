# 部署到 Jetson AGX Orin

本指南对应 [README](../README.md) 的 TartanRGBT RGB + thermal 主线。所有命令在 Jetson 的仓库根目录执行。使用已有 JetPack 的 CUDA/TensorRT，不在 Python 环境中安装另一套 TensorRT。

## 1. 硬件和系统检查

已验证：AGX Orin 64GB、Ubuntu 22.04.5、L4T 36.4.7、Python 3.10.12、CUDA 12.6、TensorRT 10.3。该组合是测试条件，不是所有 JetPack 版本的兼容承诺。需要网络下载数据和上游模型，以及足够空间容纳解压数据、两个 Python 环境和模型构建临时文件；建议预留至少 15 GB。

```bash
cat /proc/device-tree/model
cat /etc/nv_tegra_release
python3 --version
python3 -c "import tensorrt as trt; print(trt.__version__)"
/usr/src/tensorrt/bin/trtexec --help
gst-inspect-1.0 nvv4l2h264enc
gst-inspect-1.0 nvvidconv
ffprobe -version
```

缺少基础工具时安装 `git python3-venv python3-dev ffmpeg gstreamer1.0-tools`。TensorRT Python 绑定、`trtexec`、CUDA/NVRTC 和 NVIDIA GStreamer 插件应来自匹配的 JetPack 安装。如果插件不可用，先修复系统环境；不能用软件编码的速度冒充本文硬件编码结果。

```bash
git clone https://github.com/RyanAnvie/jetson-stereo-perception.git
cd jetson-stereo-perception
python3 -m venv --system-site-packages .venv-jetson-trt
.venv-jetson-trt/bin/python -m pip install --upgrade pip setuptools wheel
.venv-jetson-trt/bin/python -m pip install -r requirements/jetson-runtime.txt
.venv-jetson-trt/bin/python -m pip install --no-deps -e .
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1
.venv-jetson-trt/bin/python -c "import cv2, numpy, tensorrt; from cuda import cuda, nvrtc; print(cv2.__version__, numpy.__version__, tensorrt.__version__)"
```

使用 venv 内的 pip/setuptools 安装项目，保留系统 Python 与 JetPack 组件。

运行依赖固定在 [jetson-runtime.txt](../requirements/jetson-runtime.txt)。这是直接依赖版本清单，不是包含系统包和所有传递依赖的完整锁文件。CUDA 配准运行时由 NVRTC 编译内核；模型预热与内核初始化不计入稳态基准。

## 2. 下载指定序列

```bash
.venv-jetson-trt/bin/python scripts/download_sample.py \
  --revision 23475c0fa9190154e6f1fd78f6b82b29f9d508a0 \
  --sequence day2/undistorted_images_all_cameras_20250824_094534 \
  --inspect-only

.venv-jetson-trt/bin/python scripts/download_sample.py \
  --revision 23475c0fa9190154e6f1fd78f6b82b29f9d508a0 \
  --sequence day2/undistorted_images_all_cameras_20250824_094534
```

默认保存到 `data/tartan/`，约 0.9 GB 压缩下载量，只选一个序列以及共享标定/配置。下载器记录文件 SHA-256，解压 RGB、8 位热像、热像 FFC 标志和已发布深度，生成 `sample_manifest.json`。不要只拷贝 PNG 而遗漏时间戳、FFC、manifest 和标定文件。

当前固定修正严格绑定此数据版本、序列和标定哈希。更换序列要重新验证，不能关闭校验来强行复用。

## 3. 获取并导出双模态模型

上游为 [LCAFNet](https://github.com/WenCongWu/LCAFNet/tree/491f21019f15abb5b94c3dc832307f567747a461)，使用官方 M3FD checkpoint。上游代码与权重没有被本项目重新授权为 MIT；下载和使用前参见 [第三方说明](../THIRD_PARTY.md)。本仓库不分发它们。

在 Jetson ARM64 上建立独立 CPU 导出环境；它不需要 Jetson GPU 版 PyTorch，不修改运行环境：

```bash
python3 -m venv .venv-rgbt-export
.venv-rgbt-export/bin/python -m pip install --upgrade pip setuptools wheel
.venv-rgbt-export/bin/python -m pip install -r requirements/export-cpu.txt
mkdir -p results/thermal_research models/weights
git clone https://github.com/WenCongWu/LCAFNet.git results/thermal_research/LCAFNet
git -C results/thermal_research/LCAFNet checkout --detach 491f21019f15abb5b94c3dc832307f567747a461

.venv-rgbt-export/bin/gdown 14HxymFlpwtq4eVr8-gn5QoivRs7AYieJ \
  -O models/weights/LCAFNet_M3FD.pt
sha256sum models/weights/LCAFNet_M3FD.pt
```

checkpoint 预期 SHA-256：

```text
40c041aa51d974f7c7b7f398b45060d420785312e39c33879fd00f1fba49469b
```

```bash
.venv-rgbt-export/bin/python scripts/export_rgbt_model.py \
  --repository results/thermal_research/LCAFNet \
  --checkpoint models/weights/LCAFNet_M3FD.pt \
  --output models/weights/lcafnet_m3fd.onnx
```

导出器在反序列化 checkpoint 前检查上游源代码版本和 checkpoint 哈希；仅使用上述可信上游文件。输出包含 ONNX 和同名 `.json` 元数据，输入为两个 `[1,3,640,640]` 张量，预测为 `[1,102000,11]`。不要遗漏 JSON。官方链接不可用时从上游 README 获取同一文件，保持哈希检查。

## 4. 在目标设备构建 TensorRT

```bash
.venv-jetson-trt/bin/python scripts/build_jetson_engines.py \
  --model models/weights/lcafnet_m3fd.onnx \
  --precision fp32 \
  --output-dir models/weights/jetson \
  --report-dir reports/local/build_rgbt
```

构建使用 `--noTF32`，生成 `lcafnet_m3fd_fp32.engine` 及同名 `.json`，记录模型/engine 哈希、TensorRT 和构建参数。脚本拒绝覆盖已有 engine。需要重建时先把旧 engine 和 JSON 移到自己的备份目录，或选择新输出目录并更新配置。

engine 依赖目标硬件和 TensorRT 版本，不应直接跨设备拷贝。发布配置使用 FP32，保持与公开参考结果一致的数值计算路径。

## 5. 运行和检查结果

```bash
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1
.venv-jetson-trt/bin/python scripts/run_tartan_joint.py \
  --frames 5 --report reports/local/smoke.json

.venv-jetson-trt/bin/python scripts/run_tartan_joint.py \
  --video results/demo/balanced.mp4 --report reports/local/balanced.json

.venv-jetson-trt/bin/python scripts/run_tartan_joint.py \
  --config configs/tartan_joint_responsive.yaml \
  --video results/demo/responsive.mp4 --report reports/local/responsive.json
```

完整序列预期：296 组有效输入、6 组 FFC 无效输入、0 丢弃有效帧、0 保持上一帧的视频帧。输出为报告 `.json`、计时 `.csv`、逐帧预测 `.outputs.json` 和指定的视频。新的 engine 可能因构建策略产生微小数值变化，需要重新比较，不能直接宣称逐位等价。

```bash
.venv-jetson-trt/bin/python -m pip install -r requirements/test.txt
JETSON_TEST_JOINT_MODEL=1 JETSON_TEST_CUDA_REGISTRATION=1 \
  .venv-jetson-trt/bin/python -m pytest -q
```

上述 GPU 测试使用实际模型和 CUDA。`JETSON_TEST_ENGINE_DIR` 属于额外 YOLOX FP32/FP16 测试，只有按 [KITTI 指南](KITTI.md) 构建这两个 engine 后才设置。

CPU 参考运行需要 ONNX Runtime 和 `configs/tartan_joint_fixed.yaml`：

```bash
.venv-jetson-trt/bin/python scripts/run_tartan_joint.py \
  --config configs/tartan_joint_fixed.yaml --backend onnxruntime \
  --frames 5 --report reports/local/cpu_reference.json
```

## 6. 功耗、自动风扇和性能复测

默认使用系统自动风扇服务：

```bash
sudo systemctl enable --now nvfancontrol
systemctl is-active nvfancontrol
sudo nvpmodel -q
sudo jetson_clocks --show
```

风扇由 `nvfancontrol` 管理。根据散热条件选择功耗模式，并记录实际运行环境。

需要性能对照时，先查本机 `/etc/nvpmodel.conf`，确认 MAXN 对应的模式编号，再选择模式；不要把其他型号的编号直接照搬。公开性能的具体硬件设置见 [性能说明](BENCHMARKS.md)。`sudo jetson_clocks` 可固定时钟，**不要添加 `--fan`** 以保持自动风扇策略。记录实际温度、功耗、时钟和风扇条件，与 [基准条件](BENCHMARKS.md) 分开报告。

```bash
# 交错运行两个发布配置各三次，含完整视频编码。
.venv-jetson-trt/bin/python scripts/benchmark_tartan_latency.py --repeats 3
```

这个脚本比较公开 FP32 参考输出并记录性能。新构建 engine 的微小数值差异可能触发一致性检查。先用上面的普通运行命令生成完整报告并检查差异；不要用新结果静默覆盖已发布参考。无需 Jetson 即可复核发布数字：`python3 scripts/check_release.py`。
