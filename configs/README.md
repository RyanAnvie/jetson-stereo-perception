# 配置入口

| 文件 | 用途 |
| --- | --- |
| `tartan_joint.yaml` | 默认 FP32 / CUDA / 双上下文 / 独立调度 / 小队列 |
| `tartan_joint_latency.yaml` | 与默认相同，用于正式三配置对照 |
| `tartan_joint_responsive.yaml` | 单上下文和最小预取，优先响应延迟 |
| `tartan_joint_max_fp32.yaml` | 上一轮吞吐配置，作为延迟对照 |
| `tartan_joint_max_fp16.yaml` | FP16 实验，位置结果有差异，不作默认 |
| `tartan_joint_fixed.yaml` | 固定外参 + 时序补偿的 CPU 注册参考路径 |
| `tartan_joint_fixed_baseline.yaml` | 固定标定对照实验 |
| `tartan_joint_fast.yaml` | 早期 CUDA 优化对照配置 |
| `tartan_joint_ecc_legacy.yaml` | 早期逐帧残差拟合实验，不作默认 |
| `tartan_fusion.yaml` | 仅几何融合的调试入口 |
| `default.yaml` | 原 TartanRGBT RGB-only 回放基线 |
| `kitti.yaml` | 独立 KITTI RGB 双目基线 |

`thermal_profile` 描述发布的热像图像坐标、上游来源和哈希。`tartan_fixed_refinement.json` 是当前固定外参/时间修正，`tartan_scene_refinement.json` 是较早场景校正，场景线/观测文件保留拟合输入。不要修改这些文件来绕过校验。

当前修正只验证了固定版本的单个序列。命令在仓库根目录运行，配置中的数据/模型路径均相对该目录。正式基准继续保留 FP32、无 ECC、相同阈值和全部有效输入。详见 [架构](../docs/ARCHITECTURE.md) 和 [延迟设置](../docs/TARTAN_LATENCY.md)。
