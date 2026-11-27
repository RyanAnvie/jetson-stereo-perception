# 吞吐与响应延迟配置

默认配置对应 `configs/tartan_joint.yaml`，与 `configs/tartan_joint_latency.yaml` 相同；响应优先配置为 `configs/tartan_joint_responsive.yaml`。运行命令见 [Jetson 部署](JETSON.md)，完整测量口径见 [性能报告](BENCHMARKS.md)。

| 参数 | previous | balanced | responsive |
| --- | ---: | ---: | ---: |
| 预取组数 | 4 | 2 | 1 |
| CPU 准备任务 | 3 | 2 | 1 |
| OpenCV 线程 | 4 | 10 | 12 |
| GPU 推理上下文 | 2 | 2 | 1 |
| 渲染队列上限 | 2 | 1 | 1 |
| 交付方式 | 同线程补充输入 | 独立调度 | 独立调度 |
| 完整 FPS | 33.31 | 34.71 | 24.11 |
| input-to-output p95 | 231.91 ms | 146.15 ms | 108.91 ms |

独立调度线程负责拉取输入，检测工作线程完成结果后直接交付。输入队列和渲染队列限制在较小容量，减少已准备帧在 GPU 前等待；减少单帧排队并不等于减少模型计算量。

准备完成后的等待 p95 从 76.30 ms 降为 balanced 19.93 ms / responsive 1.50 ms；渲染队列等待 p95 从 24.71 ms 降为 1.57 / 0.57 ms。双上下文提高重叠吞吐，单上下文减少 GPU 竞争，因此 responsive 更快返回单帧结果但整体 FPS 较低。

原 FP32 模型、固定标定、运动补偿、深度和全部有效输入均保持一致。均衡模式仍有编码器启动和顺序交付引起的 p99 长尾；这些样本保留在统计中。相机采集等待、在线深度、帧压缩完成与显示延迟尚未计入，不能把 108.91 ms 解释为完整相机到屏幕的延迟。

调参工具 `scripts/tune_tartan_latency.py` 提供默认、`--cpu-sweep`、`--advanced` 扫描，会在本机执行大量运行。先建立普通完整视频基准，再调整线程/队列；增加并发不保证降低延迟。
