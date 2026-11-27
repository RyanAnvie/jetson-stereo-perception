# 性能与媒体证据

- [性能汇总](published/tartan_latency/benchmark_summary.json)：balanced / responsive 各 3 次完整运行，每次 296 有效组。
- 同目录 6 份 CSV 和 JSON：逐帧计时、处理配置和测量结果。
- [输出一致性](published/tartan_latency/output_equivalence.json)：6 次运行匹配 [公开 FP32 参考](published/reference_outputs.json)。
- [媒体验证](published/tartan_latency/media_validation.json)：完整视频的帧数、哈希和解码结果。
- [GIF 预览](published/readme_previews.json)：来源 MP4、尺寸、帧率、时长和循环设置。
- [来源校验和](published/provenance.json)：公开文件与计时来源的 SHA-256。

[性能说明](../docs/BENCHMARKS.md) 定义硬件环境、计时边界和适用范围。逐帧数值一致性不是检测准确率评价。

`model_provenance.json` 提供可选 YOLOX 模型的来源和下载校验信息。新的运行结果写入 `reports/local/` 或其他忽略目录；只将最终公开结果放入本目录。
