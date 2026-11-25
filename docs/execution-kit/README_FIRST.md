# Real Time Multimodal 3D Perception: Codex Execution Kit

This kit is an **execution plan**, not a finished implementation. It is intended for a local Codex workspace on a PC or Mac first, then for a separate Jetson AGX Orin 64GB deployment.

## Start here

1. Create a NEW, empty project folder for this project. Do not put this kit on top of an existing research repository unless you have reviewed the files and backed it up.
2. Copy `AGENTS.md`, `PLANS.md`, and `CODEX_PROMPTS.md` into the root of the new folder.
3. Open that local folder as a Codex project. Ask Codex to read `AGENTS.md` and `PLANS.md`, then paste **Phase 0** from `CODEX_PROMPTS.md`.
4. Run one phase at a time. Do not proceed without the phase acceptance evidence in `reports/` and a successful test run.
5. Do not download the entire public dataset. Start with one sequence and only RGB, thermal, depth, timestamp, FFC and calibration modalities.
6. Preserve any failure with logs and a minimal reproduction before requesting a fix.

## Architecture

Local PC or Mac: dataset access, clean Python environment, calibration tests, geometry, multimodal visualization, model baseline and regression tests.

Jetson: environment inspection, supported inference runtime, TensorRT engine build on-device, 10 Hz replay and measured optimization.

## Primary and secondary datasets

Primary: TartanRGBT 10 Hz release: https://huggingface.co/datasets/theairlabcmu/TartanRGBT

Required subset: `calibration/` plus a single trajectory's `zed_left_rect.zip`, `thermal_left_rect_8.zip`, `stereo_depth.zip`, `target_timestamps.txt`; optional `zed_right_rect.zip`, `thermal_left_rect_16.zip`. The exact file paths and sizes must be discovered from the remote file manifest before download. `stereo_depth` is computed from FoundationStereo; do not label it as independent 3D ground truth.

Secondary: MID 3K: https://github.com/kennedyk1/MID-3K and https://kennedyk1.github.io/MID-3K/

Use MID 3K's provided YOLO person annotations for detection evaluation only after confirming image-to-label matching. Its LiDAR-derived depth PNG physical encoding is not documented clearly enough to assume metric depth; verify first or use raw LiDAR cloud and calibrated projection. Do not pretend selected MID 3K scenes are a continuous time-series.

## Critical calibration facts to check against source files

* TartanRGBT 10 Hz RGB rectified images are 960 x 540. Use `calibration/released_zed_rectified_960x540.yaml`, not the 1920 x 1080 factory intrinsics.
* `stereo_depth` contains float32 `.npy` depth in meters, estimated using FoundationStereo.
* `calibration/tf_static.json` was corrected in the current release. Verify transform direction and frame conventions with synthetic tests and a real overlay before using it.
* Thermal images are rectified in the downloaded release, whereas `thermal_left.yaml` declares an `equidistant` source distortion model. Identify the correct projection model and coordinate domain for the released thermal images. Never apply distortion twice.
* Use official timestamp-error and FFC sidecars to identify missing, mis-synced, or invalid thermal frames.
* MID 3K calibration blocks are pair-specific and contain different units: COLOR–THERMAL direct baseline is in mm, LiDAR-camera transforms use meters. Never mix intrinsics and extrinsics from unrelated blocks.

## Start prompt

Paste the **Phase 0** prompt in `CODEX_PROMPTS.md` into Codex. After it passes, continue to Phase 1. Use the corrective prompt whenever a gate fails.

## Project scope and claims

* "10 Hz playback" is not identical to measuring true camera-to-display latency.
* "3D position from dataset depth" is not an independent ground truth accuracy claim.
* TartanRGBT has no general-purpose YOLO labels in its published dataset card; do not report an mAP score from it without new trusted annotations.
* Use the source dataset's citation and licensing terms. Do not push large raw data or model weights to GitHub.
* This kit does not give Codex network or SSH permissions; those must be granted in the execution environment. Remote device changes require explicit user approval.

Sources: TartanRGBT dataset card and calibration files; MID 3K official dataset README and calibration website; Codex AGENTS.md and iterative workflow guidance; NVIDIA JetPack and TensorRT documentation. Links are provided in `PLANS.md`.
