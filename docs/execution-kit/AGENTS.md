# Agent instructions for Real Time Multimodal 3D Perception

## Objective
Build a reproducible, benchmarked Python pipeline for RGB + thermal + estimated metric depth fusion using a small public TartanRGBT sequence on a PC first, then deploy the same core pipeline to Jetson AGX Orin 64GB. Use MID 3K only for verified, labeled person-detection evaluation and optional geometry validation.

## Non-negotiable boundaries
1. Read the relevant phase of `PLANS.md` before editing. Execute only one requested phase per task.
2. Never claim a test passed, a dataset was downloaded, hardware was accessed, or a metric was measured unless command output or a saved artifact proves it.
3. Stop on unknown sensor units, undocumented depth PNG encoding, uncertain transform direction, unexpected image dimensions, missing time synchronization metadata, or unverified distortion model. Write a blocker report; do not substitute guesses.
4. Do not download entire datasets. List exact files and sizes first, choose one trajectory, require explicit approval for downloads exceeding 5 GB and for any credential-gated access.
5. Do not use elevation or temperature values unless the dataset documents the units and conversion. Raw thermal values are not Celsius temperatures.
6. Never silently use image resizing, camera parameter reuse, synthetic labels, identity extrinsics, zero distortion, fabricated data, random depth, or ground-truth labels to bypass a failed gate.
7. Do not push to GitHub, change the remote machine, flash/reinstall Jetson, install system packages with sudo, or modify the user's external files without explicit approval.
8. Keep model weights, public dataset payloads, cache, video renders, and secrets outside git; add `data/`, `models/weights/`, `results/large/`, `*.engine`, `.env` to `.gitignore`.
9. Write code that runs on PC or Mac without CUDA for basic geometry/tests. Jetson/TensorRT components must be optional and must fail with actionable diagnostics if unavailable.
10. Unit tests must exercise camera backprojection/reprojection round-trip, rigid transforms, inverse transforms, image resize intrinsics, invalid depth handling, timestamp matching, FFC filtering, and detection bbox clipping.
11. Separate algorithm processing time from end-to-end dataset playback latency and acquisition latency. Always report measurement method, sample count, device and precision.
12. Each code change must include a focused test or explicit reason why no test applies. Run `python -m pytest -q` and available lint commands; log results.
13. After any phase, update `reports/status.md`, `reports/decision_log.md`, and the relevant benchmark or data audit artifact. Record unresolved issues rather than suppressing them.
14. If blocked after two evidence-driven corrective attempts on the same failure, stop and produce a compact blocker report rather than repeatedly changing unrelated code.

## Engineering principles
* Prefer small, testable modules: dataset, calibration, geometry, detector, tracking, fusion, replay, visualization, benchmark, jetson.
* Use a typed sample record with explicit RGB/depth/thermal shapes, coordinate-frame IDs, timestamps, status flags and units.
* Use depth values only when finite, positive and in an explicitly known metric scale.
* Use versioned configuration and lock down dataset revision, sequence name, frame indices and model hash in reports.
* Keep CPU baseline output as an oracle during GPU optimization; quantify deviations rather than asserting equivalence.
* Never confuse a 3D point in camera coordinates with a georeferenced global position.
* Use OpenCV pinhole vs fisheye projection according to verified image domain. Never apply raw-camera distortion to rectified imagery without a documented need.

## Completion report format
At the end of each task report: changed files, exact commands executed, tests passed/failed, input data provenance, observed metrics (or `not measured`), remaining risks, and whether the acceptance gate passes.
