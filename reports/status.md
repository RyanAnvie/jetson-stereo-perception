# Implementation status — 2026-11-25

**Update:** The user's clarified goals are now implemented further through the KITTI YOLO + stereo baseline. See [current KITTI status](kitti_status.md). The phase table below preserves the earlier attachment-kit assessment and does not describe the new KITTI results.

The user's request was to implement the supplied project and create a GitHub-uploadable video. The PC implementation and video are delivered; the complete calibrated RGB–thermal–3D / Jetson target is **not fully achieved**. Attachment instructions were treated as project requirements/reference, not independently pasted new user requests. Independent PC work continued while the thermal-specific gate remained closed.

| Phase | Gate against the full supplied plan | Evidence / remaining work |
| --- | --- | --- |
| 0 environment / synthetic geometry | PASS | Installable package, isolated Python 3.12, smoke report, tests |
| 1 real data ingestion | PASS | Pinned manifest, 302 decoded groups, 296 valid groups, 6 FFC exclusions, audit JSON |
| 2 geometry / real thermal calibration | BLOCKED | RGB and inverse TF tests pass; image_processed projection domain lacks decisive source evidence |
| 3 calibrated multimodal mapping | BLOCKED | RGB-depth cloud works; synthetic z-buffer correspondence passes; real thermal mapping disabled |
| 4 detector / position / thermal association | BLOCKED | CPU model ran on 296 frames; XYZ output exists; predictions include visible false positives; thermal association blocked |
| 5 full fused replay performance | BLOCKED | Reduced RGB-depth pipeline benchmark and timestamp replay completed; no full thermal-fusion performance claim |
| 6 Jetson / TensorRT | BLOCKED | No device access; read-only portable probe and deployment instructions delivered |
| 7 local reproducibility / video package | PASS | README, provenance, reports, tests, preview and H.264 MP4; nothing pushed |

## Commands executed

All Python commands below used `.venv/Scripts/python.exe` on this machine.

```text
python -m pip install -e .
python -m pytest -q
python scripts/download_sample.py
python scripts/inspect_dataset.py
python scripts/validate_calibration.py
python scripts/run_demo.py --smoke
python scripts/benchmark.py
python scripts/run_demo.py --paced --report reports/benchmark_paced.json
python scripts/run_demo.py --video results/demo/demo_raw.avi --report reports/benchmark_render.json
python scripts/encode_video.py results/demo/demo_raw.avi results/demo/axg_multimodal_demo.mp4
python scripts/inspect_jetson.py
```

The initial dependency install explicitly installed NumPy, OpenCV, PyYAML, pytest, huggingface_hub and onnxruntime into `.venv`. Exact installed versions are in `requirements-lock-pc.txt`. See `tests.log`, `audit.log`, `calibration_run.log`, `benchmark_run.log`, `paced_run.log`, `render_run.log`, `video_encode.log`, and `jetson_probe.log` for execution evidence. Model acquisition used the same official release URL as `scripts/download_model.py`; model checksum verification is recorded separately.

## Provenance and limits

Dataset: `theairlabcmu/TartanRGBT`, revision `23475c0fa9190154e6f1fd78f6b82b29f9d508a0`, day2 / 20250824_094534. Model: YOLOX Nano release 0.1.1rc0, source/hash in `model_provenance.json`. Third-party licenses and citation retained.

No detection accuracy, depth MAE, registration pixel error, physical camera latency, Jetson FPS or optimization speedup was measured. Real test data were never replaced by synthetic data. The detector's empty and incorrect predictions were retained.

No Python decorators or classes were introduced. No training, GPU processing, multithreaded application pipeline, data normalization or unit conversion was added. Fixed-resolution model preprocessing and visualization crops are explicit. ONNX Runtime uses its normal internal CPU threading, configured as four inference threads.

## Final validation

- Main environment: **13 passed in 0.15 s**, recorded in `tests.log`.
- Separate clean environment (`data/validation-venv`, install `-e ".[test]"`): **13 passed in 1.09 s**, recorded in `fresh_tests.log`; installation log is `fresh_install.log`.
- Reviewed all 18 project Python files with AST parsing and checks for decorators, classes and trailing whitespace: no issues. See `release_validation.json`.
- MP4: **43.5 s, 435 frames, 1280×800, H.264, yuv420p, 4,515,608 bytes**. Entire file decoded with FFmpeg; title, midpoint and ending images visually reviewed. No cropped/overlapping text observed.
- Source timestamps: actual full-sequence rate **8.209 Hz**, interval p50 **133.269 ms**. Paced run completed all 296 valid frames with zero completion delays above 100 ms; pacing follows source timestamps rather than forcing a nominal cadence.
- `git check-ignore` confirmed raw data, model weights, video and `.env` exclusion. No nonignored file exceeded 5 MB. `git diff --check` exited cleanly; this is a newly initialized, uncommitted repository, so the separate source checks are the meaningful untracked-file validation.
- Manual interactive-window behavior, Linux/macOS execution, Jetson and TensorRT were not tested.
