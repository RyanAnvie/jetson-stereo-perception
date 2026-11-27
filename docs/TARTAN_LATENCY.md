# 吞吐与响应延迟配置

默认入口 `configs/tartan_joint.yaml` 与 `configs/tartan_joint_latency.yaml` 相同。`configs/tartan_joint_responsive.yaml` 用于优先降低单帧响应时间。运行方式见 [Jetson 部署](JETSON.md)，计时口径见 [性能说明](BENCHMARKS.md)。

| 参数 | balanced | responsive |
| --- | ---: | ---: |
| 预取组数 | 2 | 1 |
| CPU 准备任务 | 2 | 1 |
| OpenCV 线程 | 10 | 12 |
| GPU 推理上下文 | 2 | 1 |
| 渲染队列上限 | 1 | 1 |
| 输入调度 | 独立线程 | 独立线程 |
| 完整 FPS | 34.71 | 24.11 |
| input-to-output p95 | 146.15 ms | 108.91 ms |

独立调度线程负责拉取输入；检测工作线程完成结果后交付。输入准备和渲染队列保持有界，减少已准备帧等待。双推理上下文提高计算重叠，单上下文减少 GPU 竞争，因此 responsive 的单帧响应更快、整体吞吐较低。

两个配置使用相同 FP32 模型、固定标定、运动补偿、深度和全部有效输入。准备完成后的等待 p95 为 balanced 19.93 ms / responsive 1.50 ms，渲染队列等待 p95 为 1.57 / 0.57 ms。

编码器启动和按源顺序交付仍可能造成 p99 长尾。相机采集等待、在线深度计算、帧压缩完成与显示延迟不在当前帧级计时范围内。
