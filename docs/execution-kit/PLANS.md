# Implementation Plan and Acceptance Gates

## Target
A reproducible RGB + thermal + metric estimated depth perception pipeline on the 10 Hz TartanRGBT release, with object detection, 3D target position estimation in a named camera coordinate frame, thermal association, visualization, timing measurements, and portable PC-to-Jetson deployment. A separate MID 3K evaluation adapter should use labels to compute verified person-detection performance.

**Important:** This plan supplies targets and checkpoints, not achieved results. Only benchmark output may support claims of accuracy, FPS, latency, or speedups.

## Official sources
* TartanRGBT official card: https://huggingface.co/datasets/theairlabcmu/TartanRGBT
* TartanRGBT released RGB intrinsics: https://huggingface.co/datasets/theairlabcmu/TartanRGBT/blob/main/calibration/released_zed_rectified_960x540.yaml
* TartanRGBT corrected transforms: https://huggingface.co/datasets/theairlabcmu/TartanRGBT/blob/main/calibration/tf_static.json
* TartanRGBT thermal model: https://huggingface.co/datasets/theairlabcmu/TartanRGBT/blob/main/calibration/thermal_left.yaml
* MID 3K: https://kennedyk1.github.io/MID-3K/
* MID 3K repository: https://github.com/kennedyk1/MID-3K
* Codex project instructions: https://learn.chatgpt.com/docs/agent-configuration/agents-md
* Codex iterative workflow: https://developers.openai.com/cookbook/examples/codex/iterating-development-workflows-with-codex
* JetPack docs: https://docs.nvidia.com/jetson/jetpack/
* TensorRT target-dependent build: https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/porting.html

## Data contracts and caveats

### Primary TartanRGBT release
* Use only the 10 Hz release. Inspect `target_timestamps.txt`, `thermal_left_rect_8/` timestamps/errors, `thermal_left_ffc/data.txt`, and the official calibration directory.
* Use `zed_left_rect` (960 x 540), `stereo_depth` (float32 NPY meters, estimated, not surveyed ground truth), thermal-left rectified frames (typically 640 x 512), and their sidecars.
* Intrinsics for released RGB come from `released_zed_rectified_960x540.yaml`, not raw full-resolution `zed.yaml`.
* Verify whether released thermal frames are truly in rectified pinhole coordinates and which `K`/projection matrix correspond. Do not apply `thermal_left.yaml`'s `equidistant` distortion to already rectified pixels by assumption. If official documentation and empirical check cannot resolve this, block 3D thermal mapping, show only separate RGB and thermal views, document the blocker.
* TartanRGBT may contain no benchmark-compatible person annotations. Do not measure detection mAP from it unless independently labeled and audited.
* FFC-flagged frames are unsuitable for thermal image quality evaluation. Record count and exclude them from association metrics.
* RGB and thermal frames are temporally matched but not strictly hardware-synchronized. Report actual mismatch distributions.

### Optional MID 3K evaluation
* Image scenes are selected, not guaranteed continuous 10 Hz video.
* Use official RGB and thermal YOLO `xywhn` labels to evaluate person detections, with train/test by acquisition day as published when relevant.
* The depth image is LiDAR-derived and projected onto the RGB plane; the PNG value-to-meter encoding is not sufficiently documented in the visible README. Verify provenance before metric geometry.
* If metric depth encoding is not verified, optionally download the raw PCD source with explicit approval and use LiDAR projections with verified `T_color_from_lidar` and `T_thermal_from_lidar`, or omit MID 3K-based 3D accuracy.
* COLOR–THERMAL direct transforms use millimeters; LiDAR-camera transforms use meters. Intrinsics are specific to each calibration block.

## Proposed repository structure (create during Phase 0)
```
AGENTS.md
PLANS.md
CODEX_PROMPTS.md
README.md
pyproject.toml
.gitignore
configs/
  default.yaml
src/multimodal_perception/
  __init__.py
  datasets/tartan_rgbt.py
  datasets/mid3k.py
  camera/models.py
  camera/transforms.py
  geometry/backproject.py
  fusion/thermal_projection.py
  detection/backend.py
  tracking/tracker.py
  pipeline/replay.py
  viz/dashboard.py
  evaluation/benchmark.py
  deployment/jetson_probe.py
scripts/
  inspect_dataset.py
  download_sample.py
  validate_calibration.py
  run_demo.py
  benchmark.py
  inspect_jetson.py
tests/
  test_calibration.py
  test_projection.py
  test_transforms.py
  test_dataset.py
  test_timestamps.py
  test_depth.py
  test_pipeline.py
reports/
  status.md
  decision_log.md
  environment_pc.json
  dataset_audit.json
  geometry_report.json
  jetson_environment.json
  benchmark_pc.json
  benchmark_jetson.json
  comparison.md
data/                   # ignored by git
models/weights/         # ignored by git
results/                # ignored by git; keep small README-safe examples separately after license review
```

## Phase 0: Repo, environment, reproducibility contract
**Tasks:** Inspect machine, Python, OS, GPU, disk, network; create isolated Python project; implement basic CLI and synthetic fixtures. Install only minimum compatible PC dependencies: numpy, opencv-python, scipy, pyyaml, pytest, matplotlib, huggingface_hub; install PyTorch and detectors in later phases. On Jetson use system-compatible libraries rather than blindly pip-replacing CUDA packages.
**Acceptance:** `python -m pytest -q` succeeds on synthetic tests; a smoke CLI reports current OS/Python; README explains clean environment setup; no dataset or model claims; git diff passes checks.
**Failure response:** package solver conflicts -> minimal virtual environment and constraint report; no internet -> implement synthetic tests and leave download step blocked; no GPU -> CPU baseline is valid.

## Phase 1: Data audit and minimal subset ingestion
**Tasks:** Enumerate TartanRGBT files using Hugging Face API; select one real trajectory by file manifest and size; download only four modalities and calibration/timestamp files; implement integrity checks; parse frame-index/time mappings and FFC flags. Avoid whole-dataset `data_extraction.py` because it downloads everything.
**Acceptance:** Save exact revision, sequence, file sizes, frame counts, image shapes, depth dtype/units/range, NaN/invalid fraction, timestamp offset p50/p95, FFC count. Successfully load and display or save at least 20 genuinely matched valid frame groups. No silent pairing by filename alone.
**Failure response:** download denied -> report endpoint and access issue; wrong shape -> check dataset revision and correct intrinsics; no valid matches -> check sidecar clock units and alignment; FFC -> reject invalid frames rather than faking content.

## Phase 2: Synthetic-tested geometry and calibration
**Tasks:** Implement typed pinhole rectified RGB model, verified thermal projection model, coordinate transforms, `P_rgb=[(u-cx)*Z/fx,(v-cy)*Z/fy,Z]`, RGB->thermal transform, target pixel projection. Verify all axes and transform directions. Use only 960 x 540 RGB intrinsics matched to released data.
**Acceptance:** Synthetic tests pass for identity/inverse transform, depth backprojection and reprojection, non-identity extrinsics, mismatched unit rejection, negative Z, NaN, out-of-FOV and resizing. Rotation matrices check orthogonality/determinant. Save annotated real RGB/thermal diagnostic overlays with both raw frames and point correspondence visualization. Declare calibrated projection valid only after checking the image domain.
**Failure response:** misalignment -> validate frame IDs, intrinsics scaling, RGB optical coordinate convention, correct child/parent transform inversion, thermal rectification model, time offset, depth validity, occlusion; do not fit an unexplained homography to conceal errors.

## Phase 3: Multimodal 3D fusion without detector
**Tasks:** Sample valid RGB points, convert depth to 3D, transform and project to thermal, enforce visible field of view and occlusion policy (z-buffer or closest depth), annotate using the thermal pixels. Save a 2D matched-pixel diagnostic, 3D scatter/map, depth valid mask, and statistics.
**Acceptance:** 100 consecutive available frames process without crash (if sequence long enough), invalid depth never becomes valid 3D output, all projected points obey image bounds, failure counters are nonzero only if expected/diagnosed, 3D values have stated frame and meters. Report valid projected-point fraction, not an invented cross-modal accuracy.
**Failure response:** drift or ghost edges -> examine calibration/time skew/occlusion; projected positions negative -> check transform frames and direction; empty projections -> inspect FOV overlap.

## Phase 4: Detection and optional object association
**Tasks:** Add swappable CPU PyTorch or ONNX Runtime detector backend for offline PC. Use pretrained general-person model with explicit provenance and licensing; no retraining until baseline is measurable. Within each verified RGB bounding box or segmentation mask, filter valid depth points and compute robust representative 3D location with a confidence/valid-point count; map valid points to thermal. Optional tracker if actual consecutive frames permit.
**Acceptance:** Each output includes class, model score, 2D bbox, camera-frame 3D point or `insufficient_depth`, thermal association or `outside_fov`, and timestamps. Draw representative images. Model inference runs with zero crashes on at least 100 valid data samples. No mAP on TartanRGBT without ground-truth labels.
**MID 3K extension:** Report mAP50/mAP50-95, recall, precision only on verified labels and a fixed evaluation split; document preprocessing and class mapping.
**Failure response:** poor detection -> verify RGB normalization, color channel, resize/letterbox and class IDs before retraining; empty depth -> mask filtering and bbox/background contamination; false thermal matches -> go back to geometry and sync.

## Phase 5: Playback and PC benchmark
**Tasks:** Implement dataset replay using supplied timestamps and synchronized read groups, a headless mode, and optional visualization. Measure preprocessing, inference, geometry/fusion, rendering, and total processing latency; separate decode/I/O and warmup; record p50/p95, throughput, dropped/skipped frames and memory usage. Compare CPU geometry baseline against optimized implementations.
**Acceptance:** JSON/CSV measurements include hardware/software versions, number of frames (>= 100 if available), warmup count, image size, detector model checksum, precision, replay pacing, and timestamps. Real-time design TARGET: sustain 10 frames per second for a 10 Hz input and p95 processing latency <= 100 ms. This is a target, NOT a guaranteed result and not a camera acquisition latency measurement.
**Failure response:** low throughput -> profile stage timing first; use batch=1 and avoid GUI when benchmarking; reduce visualization rate or image resolution only with documented tradeoffs; do not claim an improvement without measured repeat trials.

## Phase 6: Jetson AGX Orin 64GB portability, TensorRT and profiling
**Tasks:** On explicitly authorized Jetson, inspect `cat /etc/nv_tegra_release`, `python3 --version`, installed CUDA/TensorRT/OpenCV/torch, RAM/disk and available executables. DO NOT flash device or replace system libraries automatically. Export supported ONNX on PC, copy to device via an approved route, build TensorRT engine on target (not PC), compare FP16 to correct CPU/ONNX or PyTorch baseline, save first/last outputs and deviations. Profile `tegrastats` and stage timings. Keep input replay sequence exactly the same as PC benchmark.
**Acceptance:** A Jetson command runs a fully local 100-frame headless replay or an explicitly explained smaller smoke set; recorded version metadata, output validity, benchmark samples, peak memory and CPU/GPU/Power metrics where available. Compare equivalent model and preprocessing. Report relative speedup only from measured medians/p95 and state clock/power policy.
**Failure response:** torch or TensorRT import fail -> identify exact JetPack/Jetson Linux/architecture and compatible wheels; unrecognized ONNX op -> isolate operator/opset, modify export not result; engine file invalid -> build natively on target; memory error -> tune batch, buffer allocations and resolution; slower than PC -> expected on different GPU, optimize against Jetson's own baseline.

## Phase 7: Audit, docs, GitHub and LinkedIn deliverables
**Tasks:** Create concise README with problem statement, supported systems, pipeline diagram, public dataset provenance/license, setup with small subset, one reproducible command, accurate baseline/optimized benchmarks, failure modes and a few authorized screenshots/demo clips. Add clean gitignore and no raw datasets/credentials/engines. Document original contributions versus third-party models/datasets. Produce one page results table with only real metrics.
**Acceptance:** A fresh environment can follow documented commands on a small sample or declared offline synthetic fixture; all tests pass; source licenses cited; no unverifiable precision claims; Git status shows no large/private data; demo images accurately labeled as real outputs or mockups.

## Benchmark measurement and definitions
* `pipeline_fps = completed_valid_output_frames / elapsed_wall_seconds` with processing condition stated.
* `model_latency_ms = detector-infer start -> detector-infer end` with warmup omitted.
* `processing_latency_ms = frame-loaded -> fused output-ready`, not sensor capture latency.
* `playback_end_to_end_latency_ms = scheduled input replay time -> completed output`, with process scheduling design stated; not physical-camera E2E.
* `p95_ms` computed from per-frame samples over the recorded timed sequence, not the 95th of sampled benchmark averages.
* `valid_depth_fraction` uses finite, positive and (if documented) sensor-range-capped Z values.
* 3D position is in a named camera coordinate frame, not global geodetic location.
* `registration_error_px` requires verified correspondences or labels; an attractive fused overlay is not ground-truth evidence.
* `depth_MAE` requires a trusted independent metric reference and identical evaluated pixels; never compare depth against itself or treat FoundationStereo estimated depth as ground truth for independent accuracy.

## Phase gates and stop conditions
Status is one of `NOT_STARTED`, `IN_PROGRESS`, `PASS`, `BLOCKED`, `FAIL`.

Do not advance a phase if any mandatory check fails. After each attempt, capture expected vs observed, exact command, traceback, input asset name, recent code diff and smallest reproducible example. Change only one root-cause hypothesis per attempt. Retest original failing case plus unit tests. Stop and request user action if access, license, dataset download approval, SSH privilege, or hardware setup is missing.

## Debugging decision tree
1. **Environment or permission error?** Verify interpreter, architecture, import paths, ports and allowed network. Never patch library internals without a reason.
2. **Dataset or pairing error?** Check repo revision, zip layout, image dimensions/dtype, timestamps and FFC sidecars; sample a single matching group.
3. **Geometry wrong?** Use a synthetic point and identity transform, then non-identity, then invert check. Confirm 960 x 540 intrinsics, z-axis direction, distortion/rectification, transformation direction and translation meters.
4. **Thermal association wrong?** Plot source and projected points with depth mask, inspect RGB/thermal time offset, field-of-view difference and occlusions. Distinguish valid empty overlap from a bug.
5. **Detection weak?** Inspect preprocessing + class mapping, evaluate labeled MID 3K subset and run a known image sanity test. Only then consider fine-tuning.
6. **Performance slow?** Use per-stage profiling, separate I/O, decode and GUI. Run repeat trials. Avoid claiming online FPS from asynchronous replay with skipped input frames.
7. **Jetson-specific issue?** Verify OS/L4T, library compatibility, ONNX opset and target-built engine. Revert to working CPU baseline before changing acceleration path.

## Blocker report template
```
Phase:
Expected result:
Observed result:
Exact command + exit code:
Input file / dataset version:
Environment/version:
Relevant error log:
Minimal reproduction:
Hypothesis 1 + evidence + attempted fix:
Hypothesis 2 + evidence + attempted fix:
Tests after fix:
Remaining blocker:
Safest next action requiring approval (if any):
```
