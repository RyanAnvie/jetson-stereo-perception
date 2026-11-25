# 从另一台电脑接手并部署到 Jetson AGX

更新时间：2026-11-25。项目：AXG_project。本文可以直接交给另一台电脑上的 Codex，或按步骤手动执行。

**接下来先在 Jetson 上复现现有 CPU 流程，再实现和验证 TensorRT 后端。当前仓库还不能直接读取 `.engine` 做完整推理。** 本文的 Jetson 命令是部署步骤，尚未在实机执行；已执行的 PC 结果见 [阶段报告](reports/kitti_status.md)。

## 1. 接手时需要知道什么

项目主线是：公开 KITTI 左右 RGB → YOLOX Nano 检测 → OpenCV StereoSGBM 深度 → 相机坐标系中的三维位置 → 离线 LiDAR 参考检查。无需先采集自己的数据，也无需先训练模型。

| 项目 | 当前状态 |
| --- | --- |
| 原电脑项目路径 | `D:\Ryan\GitHub\AXG_project` |
| Git 状态 | 写本文时所有项目文件尚未提交，没有配置远端；不能靠 `git clone` 或 `git archive HEAD` 取回当前实现 |
| 当前数据 | KITTI raw `2011_09_26_drive_0005_sync`，154 组图像 |
| 模型 | `models/weights/yolox_nano.onnx`，输入 `[1,3,416,416]`，输出 `[1,3549,85]` |
| 已验证后端 | Windows / ONNX Runtime CPU / FP32 |
| PC 证据 | 21 项测试通过；14.14 FPS；受控 10 Hz 回放零丢帧，输出延迟 p95 74.18 ms |
| Jetson 状态 | 尚未接入设备；型号、JetPack、Python、TensorRT 版本未知 |
| 下一阶段 | Jetson CPU 基线 → 目标机编译 TensorRT 引擎 → 接入推理 → 数值与性能对比 |

三维位置是检测框内有效双目点的逐分量中位数，代表可见表面，**不是物体中心**。输出坐标 `kitti_cam2_rect`：X 右、Y 下、Z 前，单位米。LiDAR 和标签只在离线评估中使用，不参与推理。热像属于原来的独立扩展，当前 Jetson 主线不依赖热像标定。

建议顺序：**准备迁移文件 → 新电脑连接设备 → 确认型号和软件版本 → 跑通 CPU → 接入 TensorRT → 保存比较报告**。

## 2. 在原电脑准备迁移文件

最简单的方法是复制整个项目文件夹，但跳过 Windows `.venv`、缓存和 `.git`。如果只复制 Git 能看到的文件，会漏掉被忽略的 KITTI 数据和模型权重。

推荐拆成两个包，方便在新电脑离线接手。以下命令在**原电脑 PowerShell** 中执行，包含尚未提交的源码，不依赖 Git 提交记录：

```powershell
Set-Location 'D:\Ryan\GitHub\AXG_project'
New-Item -ItemType Directory -Force -Path 'results/handoff' | Out-Null

tar -czf results/handoff/axg_code.tar.gz --exclude=__pycache__ --exclude=*.pyc --exclude=*.egg-info .gitignore README.md JETSON_AGX_HANDOFF.md THIRD_PARTY.md pyproject.toml requirements-lock-pc.txt configs docs reports scripts src tests
if ($LASTEXITCODE -ne 0) { throw 'Code archive failed' }

tar -czf results/handoff/axg_kitti_runtime.tar.gz data/kitti/2011_09_26 data/kitti/sample_manifest.json models/weights/yolox_nano.onnx
if ($LASTEXITCODE -ne 0) { throw 'Data archive failed' }

$archiveNames = @('axg_code.tar.gz', 'axg_kitti_runtime.tar.gz')
$checksumLines = foreach ($name in $archiveNames) {
    $digest = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path 'results/handoff' $name)).Hash.ToLowerInvariant()
    "$digest  $name"
}
$checksumLines | Set-Content -Encoding ascii -LiteralPath 'results/handoff/SHA256SUMS'
```

把 `results/handoff` 中的两个压缩包和 `SHA256SUMS` 拷到新电脑。数据包包含解压后的校准文件、图像、时间戳、LiDAR、tracklets 和模型；不需要再次携带原始数据 ZIP。Windows 虚拟环境不能用于 Jetson ARM64，必须在目标机重新建环境。

如果希望在新电脑看演示，另外复制 `results/kitti/axg_kitti_3d_demo.mp4`。视频不是运行依赖。数据及派生媒体许可见 [第三方说明](THIRD_PARTY.md)。

不方便携带数据包时，只带代码包也能接手；后续运行 `scripts/download_kitti.py` 和 `scripts/download_model.py` 重新下载。KITTI 压缩包总计约 646 MB，解压及环境需要额外空间，先用 `df -h .` 检查磁盘。无需下载整个 KITTI 数据集。

## 3. 新电脑负责控制，Jetson 负责运行

新电脑可作为 Windows、Linux 或 macOS 的 SSH 控制端。此路线假设 Jetson 已能正常启动并连网；无需为了接手代码先刷机。如果设备尚未安装系统，应先按实际型号选择 NVIDIA SDK Manager 支持的系统和主机组合，而不是套用另一种 AGX 的刷机步骤。

先从 Jetson 本机终端或路由器确认用户名和 IP。下面 `jetson-user`、`192.168.1.50` 都是**占位示例，必须替换**。在新电脑终端执行：

```bash
ssh jetson-user@192.168.1.50
```

首次连接核对目标设备的主机指纹，然后退出该 SSH 会话。在新电脑存放三个迁移文件的目录执行：

```bash
scp axg_code.tar.gz axg_kitti_runtime.tar.gz SHA256SUMS jetson-user@192.168.1.50:~/
ssh jetson-user@192.168.1.50
```

随后这些命令都在 **Jetson 的 Bash 终端** 执行：

```bash
cd ~
sha256sum --check SHA256SUMS
mkdir AXG_project
cd AXG_project
tar -xzf ../axg_code.tar.gz
tar -xzf ../axg_kitti_runtime.tar.gz
pwd
```

两个校验结果都应为 `OK`。`mkdir AXG_project` 故意不覆盖已有项目；如果目录已存在，先确认其内容，或选一个新的空目录，不要把两版代码混在一起。之后所有相对路径都以项目根目录为起点。

如果只带了代码包，跳过不存在的数据包的传输、校验和解压，完成环境准备后再下载数据。

## 4. 先确认 AGX 型号和 JetPack，不猜版本

运行只读检查，保存一份独立的设备记录：

```bash
cd ~/AXG_project
mkdir -p reports/jetson
python3 scripts/inspect_jetson.py
cp reports/jetson_environment.json reports/jetson/environment_initial.json
tr -d '\0' < /proc/device-tree/model
printf '\n'
cat /etc/nv_tegra_release
cat /etc/os-release
uname -m
python3 --version
dpkg-query -W nvidia-jetpack nvidia-l4t-core
dpkg-query -W 'libnvinfer*' 'python3-libnvinfer*'
df -h .
```

应看到 `aarch64` 和真实 Jetson 型号。`nvidia-jetpack` 元包可能没安装，因此其查询失败不等于没有 JetPack；结合 L4T 版本、已安装组件和 NVIDIA 版本表判断。原仓库中 `reports/jetson_environment.json` 是旧 PC 的探测结果，不能当成新设备状态。

根据实际情况分支，**不要自动升级 CUDA、TensorRT 或替换系统 Python**：

| 实测设备环境 | 接下来怎么做 |
| --- | --- |
| AGX Orin，JetPack 6.x，系统 Python ≥ 3.10 | 可按第 5 节尝试隔离的 CPU 环境，再检查本机 TensorRT |
| AGX Orin，已有其他受支持的 JetPack（例如 7.x） | 保留现有系统，核对 Python/ARM64 wheel 和 TensorRT 主版本；第 8 节旧精度参数不一定适用 |
| AGX Orin 或 AGX Xavier，JetPack 5.x，系统 Python 3.8 | 当前项目声明 Python ≥ 3.10，不能直接照抄通用安装命令；先执行下方兼容性任务 |
| 型号未知、系统不完整或没有 Jetson Linux | 先解决设备识别/官方系统安装，保留检查输出 |

NVIDIA 的 [JetPack 版本与设备表](https://developer.nvidia.com/embedded/jetpack-archive) 区分 Orin、Xavier 和 Thor，AGX 名称本身不足以选版本。例如 [JetPack 6.2](https://developer.nvidia.com/embedded/jetpack-sdk-62) 的系统基于 Ubuntu 22.04，包含 TensorRT 10.3；[JetPack 5.1.6](https://developer.nvidia.com/embedded/jetpack-sdk-516) 基于 Ubuntu 20.04，包含 TensorRT 8.5.2，并支持 Xavier。这里举例说明差异，不要求设备升级到这些版本。

**若是 Xavier / Python 3.8：** 把探测输出交给后文的接手提示词。优先评估对系统 Python 3.8 的最小兼容修改，包括项目 Python 版本声明、依赖可用版本及 `Path.is_relative_to` 等较新 API，再实际执行测试；不能只降低 `requires-python` 就宣称兼容。另建 Python ≥ 3.10 环境可以作为 CPU 路线，但不会自动获得系统 Python 3.8 的 TensorRT 二进制绑定。也不要把不同 Python ABI 的 `tensorrt` 目录手工拷进新环境。本文没有声称已完成这条适配路线。

## 5. 在 Jetson 建立 CPU 基线环境

本节前提：选定的 Python ≥ 3.10。使用隔离环境先运行现有 ONNX Runtime CPU 后端，不改系统 JetPack 库。

如果 `python3 -m venv` 报缺少 `venv/ensurepip`，先在该设备上安装与系统 Python 对应的 venv 包；Ubuntu 系统 Python 通常使用 `sudo apt install python3-venv`。已经具备时不要重复修改系统。

```bash
cd ~/AXG_project
python3 -m venv .venv-jetson-cpu
source .venv-jetson-cpu/bin/activate
python --version
python -m pip install --only-binary=:all: -e '.[test,detector]'
python -c 'import cv2, numpy, onnxruntime, yaml; print("OpenCV", cv2.__version__); print("NumPy", numpy.__version__); print("ORT", onnxruntime.__version__, onnxruntime.get_available_providers())'
python -m pip freeze > reports/jetson/requirements-cpu.txt
```

这里要求依赖有目标架构的二进制 wheel，避免不知情地开始编译 OpenCV。若提示 `No matching distribution`，保存 Python、架构及完整错误，再依据 [ONNX Runtime 安装说明](https://onnxruntime.ai/docs/install/) 选择兼容版本或官方构建路线；不要安装 x86 wheel 或随意找第三方 GPU wheel。当前代码明确选择 `CPUExecutionProvider`，即使机器有 GPU，也不会自动使用 TensorRT。

**不要在 Jetson 安装 `requirements-lock-pc.txt`。** 它记录的是原 Windows 环境。首次安装后记录目标机解析出的版本，后续比较固定这些版本。

## 6. 检查数据，然后做小规模运行

如果已经带了数据包，不必重复下载。只带代码包时，在项目根目录和已激活的 CPU 环境执行：

```bash
python scripts/download_kitti.py
python scripts/download_model.py
```

关键目录应至少包含：

```text
AXG_project/
  configs/kitti.yaml
  models/weights/yolox_nano.onnx
  data/kitti/sample_manifest.json
  data/kitti/2011_09_26/calib_cam_to_cam.txt
  data/kitti/2011_09_26/calib_velo_to_cam.txt
  data/kitti/2011_09_26/2011_09_26_drive_0005_sync/
    image_02/data/ ... 与 timestamps.txt
    image_03/data/ ... 与 timestamps.txt
    velodyne_points/data/ ... 与 timestamps*.txt
    tracklet_labels.xml
```

当前数据加载器即使在 benchmark 模式也检查 LiDAR 文件索引并读取 tracklets 元数据，所以不要只迁移左右图片。其逐帧推理仍然不使用 LiDAR 或标签。

核对模型，然后测试与运行：

```bash
sha256sum models/weights/yolox_nano.onnx
set -o pipefail
python -m pytest -q 2>&1 | tee reports/jetson/tests_cpu.log
python scripts/run_kitti.py --frames 3 --report reports/jetson/smoke_cpu.json 2>&1 | tee reports/jetson/smoke_cpu.log
```

模型 SHA-256 应为：

```text
c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d
```

测试预期是当前的 21 项通过。3 帧检查应生成 JSON、CSV 和 `_outputs.jsonl`，退出码为 0；失败时先定位异常，不要为了通过测试改变标定、深度单位或误差阈值。

## 7. 在同一台 Jetson 留下可比较的 CPU 结果

先记录电源/时钟配置。下面是查询，不是切换功耗模式：

```bash
sudo nvpmodel -q
sudo jetson_clocks --show
```

保存输出到 `reports/jetson`，并记录设备型号、内存、散热方式及运行日期。基线和 TensorRT 对比必须使用相同配置，不在测量中途改变电源模式或锁频。

在第二个 Jetson 终端运行监控；若系统要求权限，再使用 `sudo`：

```bash
cd ~/AXG_project
tegrastats --interval 1000 --logfile reports/jetson/tegrastats_cpu.log
```

保持前台监控，主终端完成下列测试后回到监控终端按 Ctrl+C。记录 RAM、温度、GPU/CPU 使用及设备实际提供的功耗字段，不把某一路电源数值直接当作整机功耗。[NVIDIA tegrastats 文档](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)

主终端执行：

```bash
cd ~/AXG_project
source .venv-jetson-cpu/bin/activate
set -o pipefail
python scripts/run_kitti.py --report reports/jetson/cpu_benchmark.json 2>&1 | tee reports/jetson/cpu_benchmark.log
python scripts/run_kitti.py --paced --rate-hz 10 --drop-late --report reports/jetson/cpu_10hz.json 2>&1 | tee reports/jetson/cpu_10hz.log
python scripts/run_kitti.py --mode evaluate --report reports/jetson/cpu_evaluation.json 2>&1 | tee reports/jetson/cpu_evaluation.log
```

这些命令当前已有实现。独立报告路径保留原 PC 基线；注意公共的 `reports/kitti_dataset_audit.json` 和 `results/kitti` 中的派生文件仍会更新，所以应使用迁移后的独立副本运行。

核对 154 帧是否完成、检测和有效位置计数、深度参考误差、各阶段 p50/p95、端到端吞吐、10 Hz 延迟和丢帧数量。PC 的 992 次检测/976 次有效位置是比较参考，跨平台浮点和 OpenCV 版本可能导致差异，需要解释，不能强行改到相同数量。不同架构也不要求 JSONL 摘要逐字相同。

若 Jetson CPU 低于 10 FPS，照实记录；它是同机 TensorRT 对照组。原 PC 结果不能作为“Jetson 加速前”的分母。

## 8. TensorRT：先编译引擎，再接入项目

### 8.1 检查设备上的实际工具与 Python 绑定

先退出 CPU 虚拟环境，检查系统环境：

```bash
deactivate
command -v trtexec
ls /usr/src/tensorrt/bin/trtexec
python3 -c 'import tensorrt as trt; print(trt.__version__)'
```

前两个工具路径检查只需有一个成功。`trtexec` 存在不代表 Python 绑定已安装，反之亦然。缺失时按当前 JetPack 的组件安装说明补齐；不要执行通用的 `pip install --upgrade tensorrt` 替换系统栈。

把实际找到的路径填入下面变量。此处路径是 JetPack 常见位置，必须以设备检查为准：

```bash
TRTEXEC=/usr/src/tensorrt/bin/trtexec
"$TRTEXEC" --help > reports/jetson/trtexec_help.txt 2>&1
mkdir -p models/weights/jetson
```

### 8.2 在目标 Jetson 上构建

复制 ONNX 到目标机并在目标机编译，不复用原 Windows PC 编译的引擎。引擎兼容性受平台、硬件和 TensorRT 版本影响。[NVIDIA TensorRT 兼容性说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html)

**以下两条仅用于本机 `--help` 明确支持 `--noTF32` 和 `--fp16` 的 TensorRT 8/10 环境。** 第一条约束为 FP32 对照；第二条允许 FP16，实际各层精度仍应检查构建日志。模型本来就是固定 batch 1，不要凭空添加动态 shape 或猜输入名称。

```bash
set -o pipefail
"$TRTEXEC" --onnx=models/weights/yolox_nano.onnx --saveEngine=models/weights/jetson/yolox_nano_fp32.engine --noTF32 2>&1 | tee reports/jetson/build_fp32.log
"$TRTEXEC" --onnx=models/weights/yolox_nano.onnx --saveEngine=models/weights/jetson/yolox_nano_fp16.engine --fp16 2>&1 | tee reports/jetson/build_fp16.log
sha256sum models/weights/jetson/*.engine > reports/jetson/engine_sha256.txt
```

**TensorRT 11 分支：** 官方已移除 `--fp16` 等精度开关，精度由强类型网络表达。不要照抄上面的 8/10 命令。先保留原 FP32 ONNX，按设备对应版本确认构建参数；混合精度模型需采用该版本支持的转换流程，另存 ONNX、记录新哈希并重新做数值对照。详见 [NVIDIA trtexec 10 → 11 迁移说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html)。

构建成功只证明 ONNX 能生成引擎；`trtexec` 的模型吞吐不包含图像读取、双目匹配和位置计算，也不构成本项目的精度验证。

### 8.3 当前真正需要继续开发的部分

当前 `src/multimodal_perception/detection.py` 的 `load_detector` 和 `detect` 直接调用 ONNX Runtime。`scripts/run_kitti.py` **没有** `--backend`、`--engine` 参数。不能把配置里的 `.onnx` 换成 `.engine` 就运行，也不要把尚未实现的命令写成“已经可用”。

交给下一台电脑上的 Codex 完成以下最小修改：

1. 根据实际 TensorRT 主版本实现普通函数形式的加载与推理，保留 CPU 后端。只在需要时增加一个小的 `tensorrt_inference.py`，不创建框架、类层级或装饰器。
2. 选择与系统 Python ABI 兼容的运行环境。需要系统 TensorRT 时，可从系统 Python 创建 `python3 -m venv --system-site-packages .venv-jetson-trt`，再实际检查 `numpy/cv2/tensorrt` 导入。不要将 CPU 虚拟环境的所有包直接覆盖进去。系统 OpenCV 未必以 `opencv-python` 的 pip 包名登记，可先用 `PYTHONPATH=src` 运行源码，记录实际版本并测试，而不是盲目重装 OpenCV。
3. 保持完全相同的预处理：BGR、0–255、float32、NCHW、保持比例缩放、左上角放图、补边 114、416×416。不要加入 RGB 转换、除 255、居中 letterbox 或第二次 NMS。
4. 运行时读取真实的 tensor 名称、dtype 和 shape。原模型输出 `[1,3549,85]`，复用现有 strides 8/16/32 解码、类别映射及 NMS；FP16 引擎的 I/O dtype 不可凭文件名推断。
5. 按该版本 API 管理 CUDA stream、buffer 和同步。GPU 推理计时必须等待相应工作完成；分别报告传输、模型推理和原有完整处理时间。
6. 在配置和运行报告中明确 backend、模型/引擎哈希、TensorRT/CUDA 版本、精度和 I/O 信息。保留原 CPU 命令的行为及现有 JSONL 坐标/单位。
7. 在这台 Jetson 上比较相同输入的 ORT CPU、TRT FP32、TRT FP16。CPU 与 TRT 若使用不同虚拟环境，保存预处理 tensor 或核对其内容，避免环境差异混入精度比较。

首轮不增加 INT8、DLA、追踪器、ROS 或训练流程；先把已有感知链在 GPU 上正确跑通。

## 9. 如何判断部署完成

| 检查 | 应保存的证据 |
| --- | --- |
| 环境可复现 | 型号、JetPack/L4T、Python、依赖版本、代码快照/后续提交号、模型与引擎哈希 |
| 基础正确性 | 现有 21 项测试和新增后端数值测试真实执行；合成几何测试继续通过 |
| 原始网络输出 | 同一 tensor 的 FP32/FP16 最大及分位绝对误差、相对误差；说明零附近如何处理 |
| 检测变化 | 同类框匹配后的 IoU/置信度差异、新增/丢失检测、阈值附近案例 |
| 三维输出 | 有效点比例、XYZ 差异、无深度情况；仍使用 `kitti_cam2_rect` 和米 |
| 深度参考 | 同一 154 帧的 LiDAR 覆盖率、MAE/RMSE 和大误差案例，不删异常值 |
| 性能 | 相同输入、线程、分辨率、预热及功耗配置下，模型延迟和端到端 FPS 分别比较 |
| 连续输入 | 受控 10 Hz 的完成数、丢帧数及调度至输出 p95；超过预算时照实记录 |
| 视频 | 同时展示检测、三维输出、实测设备和限制；编码可在新电脑完成 |

目前没有约定 TensorRT 数值误差容差，不要临时编造一个“全部通过”的门槛。先测量差异，再依据任务用途确定接受标准。原设计方向是 10 FPS / 100 ms 级处理；是否满足还需以 Jetson 的完整流水线数据判断。

StereoSGBM 在当前 PC 基线中比 YOLO 更耗时，所以检测器加速不等于整个系统获得同样倍数的提升。先分阶段测量，再决定是否优化双目算法；更换深度算法属于新的实验，须重新评估误差。

## 10. 给另一台电脑上 Codex 的接手提示词

将项目放到新电脑可访问的目录，在该目录打开 Codex 后粘贴以下文字。不要把密码、SSH 私钥或访问令牌写进文档。

```text
请接手 AXG_project，并先阅读根目录 JETSON_AGX_HANDOFF.md、docs/KITTI.md、
reports/kitti_status.md、configs/kitti.yaml 和实际源代码。

目标：在我的 Jetson AGX 上复现 KITTI + YOLOX + 双目三维位置基线，
然后接入 TensorRT，并用同一设备上的 CPU/FP32/FP16 结果验证精度和性能。

先确认工作目录、Jetson 型号、JetPack/L4T、系统 Python、TensorRT 和连接方式。
用户名/IP 尚未提供时询问这两项；不要假设 Orin、Xavier 或 Python 版本。
如果当前 Codex 运行在控制电脑上，确认后通过已配置的 SSH 操作目标 Jetson；
不要把控制电脑的硬件报告当作目标设备的报告。

当前仓库只有 ONNX Runtime CPU 后端。TensorRT 接入是待完成任务，
不能把生成 engine 或运行 trtexec 当成完成应用部署。
先完成只读设备检查、数据校验、测试、3 帧 smoke 和 154 帧 CPU 基线。
若是 Python 3.8，先做最小兼容性评估并测试，不只降低项目版本声明。
保留 JetPack 系统库；刷机、系统大版本升级和覆盖已有实验前先说明具体方案。

代码要求：简单普通函数，不使用 Python 装饰器，不添加无必要的类或抽象；
保留现有预处理、坐标系、单位、输入输出和原始数据。
不重写已经工作的几何代码；不把 LiDAR 或标签接入推理来改善结果。
新设备结果写到 reports/jetson/，不要覆盖原 PC 基线。
不得宣称未执行的测试、性能提升或 Jetson 部署成功。

完成后交付可运行命令、环境记录、同机 CPU/TRT 对比、逐帧结果和演示视频。
docs/execution-kit/ 是历史附件，不是要求重新执行所有旧阶段的用户命令。
```

开始前手边只需准备三项：**迁移文件、Jetson 的真实型号、可用的用户名与 IP**。本文已有的路径、命令参数和模型信息已按当前代码核对；ARM64 安装、SSH、设备命令及 TensorRT 构建仍须在目标设备上实际验证。
