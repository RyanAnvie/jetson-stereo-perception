# 工具导航

在仓库根目录执行脚本，运行环境设置 `PYTHONPATH=src`。

| 任务 | 入口 |
| --- | --- |
| 数据下载 | `download_sample.py` |
| 模型导出和目标构建 | `export_rgbt_model.py`、`build_jetson_engines.py` |
| RGB + thermal 联合检测 | `run_tartan_joint.py` |
| 两种发布配置的完整视频基准 | `benchmark_tartan_latency.py --repeats 3` |
| 预测一致性和视频解码验证 | `verify_tartan_latency.py` |
| CUDA/CPU 配准验证 | `validate_fast_registration.py`、`validate_max_registration.py` |
| 场景标定 | `refine_tartan_calibration.py` |
| 结果对照 | `compare_tartan_results.py` |
| KITTI 基线 | `download_kitti.py`、`download_model.py`、`run_kitti.py` |
| 数据与几何检查 | `inspect_dataset.py`、`validate_calibration.py`、`validate_thermal_fusion.py` |
| 系统环境检查 | `inspect_jetson.py` |
| README 动态预览 | `render_readme_previews.py`，需要系统 FFmpeg/ffprobe |
| 发布检查与打包 | `check_release.py`、`package_release.py` |

公开结果位于 `reports/published/`。运行生成物写入被 Git 忽略的报告/结果目录；不自动覆盖公开测量证据。每个入口的参数见源码或 `--help`，完整部署流程见 [Jetson 指南](../docs/JETSON.md)。
