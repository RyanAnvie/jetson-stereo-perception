# 工具导航

所有脚本在仓库根目录执行；多数运行脚本需要 `PYTHONPATH=src`。

| 任务 | 入口 |
| --- | --- |
| 数据 / 模型 | `download_sample.py`、`export_rgbt_model.py`、`build_jetson_engines.py` |
| 主线运行 | `run_tartan_joint.py`，默认 `configs/tartan_joint.yaml` |
| 完整视频延迟基准 | `benchmark_tartan_latency.py --repeats 3` |
| CPU/GPU 几何对照 | `validate_fast_registration.py`、`validate_max_registration.py` |
| 场景固定标定研究 | `refine_tartan_fixed.py`、`validate_fixed_sequence.py`、`diagnose_fixed_sources.py` |
| 联合模型 / 数值对照 | `check_joint_export.py`、`validate_tartan_joint.py`、`compare_tartan_results.py` |
| 参数扫描 | `tune_tartan_latency.py`、`tune_tartan_max.py` |
| KITTI 基线 | `download_kitti.py`、`download_model.py`、`run_kitti.py` |
| CPU 基线 / 数据检查 | `run_demo.py`、`inspect_dataset.py`、`validate_calibration.py` |
| 发布检查与打包 | `check_release.py`、`package_release.py` |
| 从本地历史整理公开证据 | `prepare_public_evidence.py --source-root <original-reports-directory>` |

`benchmark_tartan_fast/max.py`、`probe_clock_geometry.py`、`audit_fixed_registration.py`、`verify_*_deployment.py` 等保留用于历史实验复现。它们可能要求先生成相应阶段的 runtime 报告；这些中间文件不全部随公开包分发。它们不是新用户部署的前置步骤。每个命令的完整参数见 `--help`，正式入口以 [Jetson 指南](../docs/JETSON.md) 为准。

公开证据位于 `reports/published/`，新的 runtime 报告默认进入被忽略的目录。`prepare_public_evidence.py` 用于维护发布证据，不能代替重新运行实验；不要对不匹配的原始文件静默重写公开结果。
