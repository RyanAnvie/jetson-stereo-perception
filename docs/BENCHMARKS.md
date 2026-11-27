# 性能与验证范围

公开结果对应两个 FP32 配置：balanced 用于均衡吞吐与延迟，responsive 优先降低单帧响应延迟。[汇总与逐次记录](../reports/published/tartan_latency/benchmark_summary.json) 包含每种配置 3 次完整运行；逐帧 CSV 和 [来源校验和](../reports/published/provenance.json) 可用于复核。

## 运行条件

| 项目 | 条件 |
| --- | --- |
| 设备 | NVIDIA Jetson AGX Orin 64GB，12 CPU 核、8 GPU TPC |
| 系统 | Ubuntu 22.04.5，L4T 36.4.7，Python 3.10.12 |
| 计算环境 | CUDA 12.6，TensorRT 10.3，NumPy 1.26.4，OpenCV 4.11 |
| 功耗与散热 | MAXN；CPU 2201.6、GPU 1300.5、EMC 3199 MHz；风扇 PWM 255 |
| 检测 | LCAFNet M3FD，FP32 / no-TF32，RGB+thermal 各 640×640 |
| 数据 | TartanRGBT 固定版本的单个序列，使用发布的 FoundationStereo 深度 |
| 样本 | 每配置 3 次，每次 296 有效组；每配置合并 888 个逐帧样本 |
| 输出 | 1280×800 H.264，Jetson 硬件编码器，完整写入 MP4 |

结果描述上述硬件、模型和工作负载的表现；不同功耗、散热、数据或软件环境需要重新测量。两个配置共 1,776 组处理记录，来自同一序列的重复运行，不代表独立场景样本或长时间相机采集。

## 吞吐与延迟

| 配置 | 完整流程 FPS | 每次 FPS 范围 | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced，默认 | 34.71 | 34.42–35.04 | 133.40 ms | 146.15 ms | 258.75 ms |
| responsive | 24.11 | 23.69–24.52 | 95.13 ms | 108.91 ms | 141.58 ms |

FPS 按总有效帧数除以总耗时计算。表中延迟为 `input_to_output_ms`，分位数由每配置全部逐帧样本线性插值计算。每次运行的 p95 范围：balanced 143.72–148.41 ms，responsive 107.00–110.34 ms。

| 计时字段 | 起止点 | balanced p95 | responsive p95 |
| --- | --- | ---: | ---: |
| `input_to_detection_ms` | 开始准备输入 → 检测结果完成 | 112.31 ms | 84.82 ms |
| `input_to_delivery_ms` | 开始准备输入 → 主循环收到检测结果 | 118.65 ms | 85.54 ms |
| `input_to_encoder_ms` | 开始准备输入 → 提交画面到编码器返回 | 144.36 ms | 108.91 ms |
| `input_to_output_ms` | 开始准备输入 → 提交画面及可选截图保存结束 | 146.15 ms | 108.91 ms |
| `admission_to_encoder_ms` | 任务入队 → 编码器提交返回，包含准备线程排队 | 144.58 ms | 108.98 ms |

完整流程 FPS 包含读取/解码、运动估计、GPU 配准/预处理、推理、筛选/NMS/XYZ、渲染、编码、队列排空和编码器最终完成。模型载入、CUDA 内核编译及每上下文 3 帧预热不计入。全部有效输入均保留，不通过丢帧提高 FPS。

帧级延迟截止到画面提交及相应输出工作结束，不包含该帧压缩完成、网络传输或显示。输入来自可立即读取的文件；实时等待相机曝光或下一 RGB 帧的时间未计入。深度使用离线估计，在线深度网络也不在计时范围内。10 FPS 视频播放速度与处理吞吐分别定义。

p99 包含编码器启动及顺序交付产生的延迟尾部。配置选择和调度参数见 [延迟配置](TARTAN_LATENCY.md)。

## 结果一致性与精度范围

6 次运行逐帧匹配 [FP32 参考输出](../reports/published/reference_outputs.json)，包括类别、框、置信度、位置和有效性，见 [一致性记录](../reports/published/tartan_latency/output_equivalence.json)。每遍序列包含 21 个检测和 8 个有效位置。数值一致性不代表检测准确率；本序列没有检测 mAP 或召回率评价。

使用一套固定外参和时间修正；296 个有效帧中，279 帧接受 RGB 运动补偿，17 帧支持不足时回退。有限人工验证包含 3 帧、6 个水平边缘，平均误差约 4.79 px、标注不确定度约 ±4 px，不能视为完整视频的配准精度。参数和适用范围见 [标定配置](../configs/tartan_fixed_refinement.json)。

深度为估计值，XYZ 是框内有效点的表面代表位置；动态场景、遮挡及跨模态外观差异可能影响结果。当前证据不构成长期热稳定性或实时相机到屏幕延迟保证。

## 复核与重测

`python3 scripts/check_release.py` 使用标准库检查 6 份计时 CSV 的汇总分位数、FPS、来源哈希、视频/GIF 哈希和文档链接，不需要 GPU。

按 [部署指南](JETSON.md) 准备数据和模型后，运行：

```bash
python scripts/benchmark_tartan_latency.py --repeats 3
python scripts/verify_tartan_latency.py
```

在已激活运行 venv 且 `PYTHONPATH=src` 的仓库根目录执行。新的 engine 可能产生微小数值差异；先用普通运行入口生成输出并检查差异，不要覆盖公开参考。生成的报告和视频保存在本地输出目录。
