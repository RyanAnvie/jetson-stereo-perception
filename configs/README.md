# 配置入口

| 文件 | 用途 |
| --- | --- |
| `tartan_joint.yaml` | 默认 FP32 / CUDA / 双推理上下文 / 独立调度 |
| `tartan_joint_latency.yaml` | 与默认相同，供 balanced 性能测量使用 |
| `tartan_joint_responsive.yaml` | 单推理上下文、单组预取，优先响应速度 |
| `tartan_joint_fixed.yaml` | CPU 几何参考路径，保留固定外参与时序补偿 |
| `tartan_fusion.yaml` | 几何融合可视化入口 |
| `default.yaml` | TartanRGBT RGB 检测与多模态回放基线 |
| `kitti.yaml` | 独立 KITTI RGB 双目基线 |

`tartan_thermal_profile.json` 描述热像坐标域、上游标定来源和哈希。`tartan_fixed_refinement.json` 提供默认固定外参及有效时间修正。`tartan_scene_lines.json` 和 `tartan_alignment_observations.json` 为公开序列的标定约束，支持场景校正代码的复现。

当前标定适用于固定版本的指定序列；新序列需要重新验证。文件哈希用于检查来源和输入约定，不应绕过。所有数据和模型路径相对仓库根目录。详见 [架构](../docs/ARCHITECTURE.md) 与 [延迟配置](../docs/TARTAN_LATENCY.md)。
