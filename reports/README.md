# 公开测量证据

- [延迟汇总](published/tartan_latency/benchmark_summary.json)：previous / balanced / responsive，各三次、296 帧。
- 同目录九份 CSV：未经修改的逐帧计时；九份 JSON 保留计算参数和结果，去除机器环境转储及上游模型架构转储。
- [输出等价记录](published/tartan_latency/output_equivalence.json)：原始九份预测逐帧解析后与 [FP32 参考](jetson/tartan_fixed/fixed_corrected.outputs.json) 相等；相同的大文件只发布一份。
- [固定配准整段诊断](published/tartan_fixed/fixed_sequence_validation.json)、时间/深度诊断：包含有限验证和回退信息，不能当作完整像素真值。
- [GPU 几何对照](published/tartan_max/registration_equivalence.json)、[FP16 对照](published/tartan_max/fp16_prediction_comparison.json)、[持续运行摘要](published/tartan_max/sustained_summary.json)。
- [来源校验和](published/provenance.json)：原始与公开文件的 SHA-256；CSV 保持字节相同，JSON 仅整理公开元数据/路径。
- [发布整理验证](published/release_validation.json)：公开目录扫描、打包和目标设备验证，独立于历史性能测量。

[性能说明](../docs/BENCHMARKS.md) 定义所有计时边界。图像相关性不是检测 mAP；等价于参考不是等价于物理真值。

`jetson/tartan_fixed/fixed_corrected.outputs.json` 保持历史路径以支持已有比较脚本。`model_provenance.json` 为可选 YOLOX 下载器的来源/校验元数据。新的运行日志生成在 `reports/local/` 或 `reports/jetson/` 的其他子目录，不进入 Git。历史 Windows、早期 Jetson 实验、机器路径和原始中间文件保留在维护者本地备份，不是公开部署依赖。
