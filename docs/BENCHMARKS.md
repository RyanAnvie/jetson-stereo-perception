# 性能、延迟与验证边界

这里是实机测量的公开摘要。原始逐帧 CSV 与处理参数在 [tartan_latency](../reports/published/tartan_latency/benchmark_summary.json)，发布整理只移除了机器环境转储等信息，没有改写计时样本。各文件的原始/公开 SHA-256 见 [provenance.json](../reports/published/provenance.json)。

## 条件和样本

| 项目 | 条件 |
| --- | --- |
| 设备 | NVIDIA Jetson AGX Orin 64GB，12 CPU 核、8 GPU TPC |
| 系统 | Ubuntu 22.04.5，L4T 36.4.7，Python 3.10.12 |
| 计算环境 | CUDA 12.6，TensorRT 10.3，NumPy 1.26.4，OpenCV 4.11 |
| 性能设置 | MAXN；CPU 2201.6、GPU 1300.5、EMC 3199 MHz；最大风扇 PWM 255 |
| 检测 | LCAFNet M3FD，FP32 / no-TF32，RGB+thermal 各 640×640 |
| 数据 | TartanRGBT 固定版本和单个序列；发布的 FoundationStereo 深度 |
| 采样 | 三种配置交错各 3 次，每次 296 有效组；每配置合并 888 个逐帧样本 |
| 输出 | 1280×800 H.264，Jetson 硬件编码器，完整写入 MP4 |

系统后来恢复自动风扇，**这里的性能没有在自动风扇下重测**。重复同一序列的 2,664 组记录并不等于 2,664 个独立场景样本，也不是长时间连续相机采集测试。

## FPS 和延迟

| 配置 | FPS（总帧数/总耗时） | FPS 最小–最大 | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| previous | 33.31 | 32.96–34.04 | 201.32 ms | 231.91 ms | 299.21 ms |
| balanced，默认 | 34.71 | 34.42–35.04 | 133.40 ms | 146.15 ms | 258.75 ms |
| responsive | 24.11 | 23.69–24.52 | 95.13 ms | 108.91 ms | 141.58 ms |

表中延迟是 `input_to_output_ms`。分位数按全部逐帧样本做线性插值，不是三次分位数的平均值。每次 p95：balanced 143.72–148.41 ms，responsive 107.00–110.34 ms。

| 计时字段 | 起止点 | balanced p95 | responsive p95 |
| --- | --- | ---: | ---: |
| `input_to_detection_ms` | 开始准备输入 → 检测结果完成 | 112.31 ms | 84.82 ms |
| `input_to_delivery_ms` | 开始准备输入 → 主循环收到检测结果 | 118.65 ms | 85.54 ms |
| `input_to_encoder_ms` | 开始准备输入 → 提交画面到编码器返回 | 144.36 ms | 108.91 ms |
| `input_to_output_ms` | 开始准备输入 → 提交画面及可选截图保存结束 | 146.15 ms | 108.91 ms |
| `admission_to_encoder_ms` | 任务入队 → 提交编码器返回，包含准备线程排队 | 144.58 ms | 108.98 ms |

完整流程 FPS 包含读盘/解码、RGB 运动估计、GPU 配准/预处理、TensorRT、筛选/NMS/XYZ、绘图、视频写入、队列排空和编码器最终完成。不包含模型载入、CUDA 内核编译及每上下文 3 帧预热。帧级延迟截止到向编码器提交返回，不包含该帧压缩完成、网络传输或显示器呈现。

这些起点不是物理曝光时间。算法可能使用下一 RGB 帧估计运动，离线文件已经可读，**实时等待下一帧的时间没有计入**。在线深度网络也不在范围内。10 FPS 视频播放速度和处理 FPS 分开。

默认配置较高的 p99 保留了冷启动编码器带来的尾部影响：早期画面提交可阻塞约 118–152 ms，并使后续排队。未删除这些慢帧来改善统计。更多调度细节见 [延迟优化](TARTAN_LATENCY.md)。

## 精度与数值等价

9 次完整运行逐帧输出均与 [原 FP32 参考](../reports/jetson/tartan_fixed/fixed_corrected.outputs.json) 相同，检查覆盖类别、框、分数、位置和状态，见 [等价记录](../reports/published/tartan_latency/output_equivalence.json)。每遍总计 21 个检测、8 个有效位置。这证明本次调度优化保留了参考计算行为，**不证明这些预测正确，也没有检测 mAP 或召回率测量**。

[CUDA 几何对照](../reports/published/tartan_max/registration_equivalence.json) 保留 CPU/GPU 数值比较。[FP16 对照](../reports/published/tartan_max/fp16_prediction_comparison.json) 中，单次速度约 36.74 FPS，21 个预测的数量/类别相同，最小框 IoU 约 0.9517，最大框坐标差约 0.516 px、分数差约 0.01718，但最大位置差约 0.93484 m，因此保留 FP32。这里的位置差是相对 FP32 参考，并非相对物理真值。

固定外参和时间补偿的 [整段验证](../reports/published/tartan_fixed/fixed_sequence_validation.json) 包含 296 帧，其中 279 帧接受 RGB 运动补偿，17 帧回退。跨模态结构相关性是诊断指标，不能转换为像素精度。固定修正下的 3 帧、6 个人工水平边缘观察从平均 11.33 px 降至 4.79 px，人工标注不确定度约 ±4 px；原始观测在 [固定修正配置](../configs/tartan_fixed_refinement.json)。这不是全视频的平均配准误差。

## 持续运行与测试

上一轮吞吐配置的 [持续运行汇总](../reports/published/tartan_max/sustained_summary.json) 包含 31 次 H.264 和 3 次无视频处理，共 10,064 组有效输入；实际计时约 301.4 秒。它重复同一短片段，不是数小时热稳定性测试；也不是新 balanced/responsive 配置的持续测试。

历史目标设备测试：[GPU 59 passed](../reports/published/tartan_latency/tests_trt.log)，[CPU 50 passed / 9 skipped](../reports/published/tartan_latency/tests_cpu.log)。GPU 结果包含另行准备的 YOLOX engine 测试；仅部署 LCAFNet 时该项会跳过。发布整理后的检查结果另见 [发布验证](../reports/published/release_validation.json)。

## 复核与重测

`python3 scripts/check_release.py` 用 Python 标准库重新读取 9 份计时 CSV，核对所有汇总分位数、逐次 FPS、来源哈希、视频哈希和相对文档链接。不需要 GPU，也不重新执行推理。

要在 Jetson 复测，按 [部署指南](JETSON.md) 准备数据/模型，运行 `scripts/benchmark_tartan_latency.py --repeats 3`。原始参考来自特定 FP32 engine；新构建的 engine 未必逐位一致，普通运行脚本可先生成结果供比较。重测报告保存到本地生成目录，不覆盖公开证据。
