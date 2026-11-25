# Paste-ready Codex prompts

Use one prompt per Codex task. Keep the same local project open. Do not paste all phases at once. A phase may be repeated after a blocker is resolved.

## Phase 0 — Initial setup and synthetic tests
```
You are the lead implementation engineer for this repository. Read AGENTS.md and PLANS.md. Perform ONLY Phase 0. First inspect the current working directory and environment without modifying external paths. Propose the intended files. Then create a small installable Python package, a minimal CLI, pytest synthetic tests for 3D pinhole backprojection/reprojection and transform inverse, .gitignore, config skeleton and README setup instructions. PC/Mac CPU-only operation is required. Do not download datasets or models in this phase. Run available tests and record exact commands/results in reports/status.md. If blocked, produce the blocker template. Finish with changed files, test output, and a clear PASS or BLOCKED gate. Do not commit or push without my approval.
```

## Phase 1 — Dataset inspection and download
```
Read Phase 1 in PLANS.md. Inspect the TartanRGBT official repository manifest via supported read-only network access. Locate one real trajectory, list the exact file paths and remote sizes for RGB left rectified 960x540, thermal left rectified 8-bit, metric stereo_depth NPY, target timestamps, optional per-modality timestamps/FFC, and global calibration. Propose a minimum subset and storage size. Never run the dataset's all-data extraction script. Do not exceed 5 GB download without asking for approval. Implement scripts/download_sample.py using a supported Hugging Face API with selective downloads and resumability, and scripts/inspect_dataset.py. Decode and validate at least 20 time-matched frame groups. Save reports/dataset_audit.json with frame count, file shapes, depth dtype/value/unit, time delta statistics, FFC mask counts, exact revision, download manifest and missing files. Add tests and run them. Do not implement detection or geometry yet. Gate PASS only on documented evidence.
```

## Phase 2 — Geometry calibration correctness
```
Read Phase 2 in PLANS.md. Implement pinhole backprojection for 960x540 rectified RGB depth in METERS, rigid-body transforms and projection into the thermal frame using the correct released TartanRGBT calibration. Inspect the source calibration files; distinguish corrected tf_static.json parent/child semantics from matrix application direction. Verify whether thermal_left_rect_8 imagery uses a rectified pinhole projection or a remaining equidistant distortion model: do not assume and do not double-distort. Create synthetic unit tests for identity/nonidentity transforms, inverse, image resizing scaling, negative Z, NaNs and out-of-FOV. Generate at least one diagnostic overlay on true paired frames and save reports/geometry_report.json explaining image domain, transform direction and pixel error evidence. If this domain/model cannot be verified, mark thermal 3D mapping BLOCKED rather than inventing a projection. Run tests; report a gate.
```

## Phase 3 — 3D multimodal mapping
```
Read Phase 3 in PLANS.md. Using already-validated dataset samples and geometry, implement RGB depth backprojection to point cloud and depth-aware RGB-to-thermal correspondence. Use valid-depth masks, image bounds and an explicit occlusion policy; preserve data in consistent named coordinate frames and meters. Add diagnostics: RGB, thermal, estimated depth, projected point correspondence, valid masks and 3D target point data. Process at least 100 real paired frames if available; otherwise state the sample limit. Keep outputs and traces reproducible; save measurable coverage statistics, not invented registration accuracy. Add tests and a report, and do not proceed if calibration is blocked.
```

## Phase 4 — Detection, person positions, thermal association
```
Read Phase 4 in PLANS.md. Add a model backend abstraction and an optional pretrained RGB person detector that works on the PC. Record model source, version, SHA-256, license, preprocessing, class mapping and image dimensions. For each RGB detection, compute a robust 3D representative point only when enough valid depth exists; do not claim this is body-center ground truth. Project valid points to thermal image. If the sequence has no people, show the pipeline with detected supported objects and explicitly log the empty-person case; do not inject fake detections. Add tests. Separately implement an MID 3K evaluation adapter with its public YOLO person labels only if requested, and do not calculate TartanRGBT mAP without labeled ground truth. Gate PASS with reproducible visual + numeric artifacts.
```

## Phase 5 — 10 Hz dataset replay and benchmark
```
Read Phase 5 in PLANS.md. Build a replay CLI that uses TartanRGBT's target timestamps and synchronization metadata, can run headless, and avoids unbounded frame queues. Add stage timing for image read/decode, detector, geometry/fusion, visualization, whole pipeline and optional FPS/p95. Run at least 100 frames (if data provides them), report warmup policy, hardware, precision, throughput, frame drops and metrics in JSON and CSV. Distinguish processed FPS from dataset input rate, and simulated replay from real physical camera capture. Do not call it 'real-time' unless the 10 Hz throughput and <=100 ms p95 processing latency design targets are measured under defined conditions. Run tests and report gate.
```

## Phase 6 — Jetson deployment (requires explicitly authorized Jetson access)
```
Read Phase 6 in PLANS.md. First produce a read-only Jetson compatibility inspection plan. Only if this Codex environment has authorized SSH/local access, run read-only commands for L4T, aarch64, Python, installed TensorRT/CUDA/OpenCV/PyTorch, storage and GPU. Do not assume SSH access, sudo privileges or active network; if access is absent, generate a portable script plus clear manual run instructions and STOP at the access gate. After the device environment is known and user has approved writes, export the model to ONNX on PC and construct the TensorRT engine on the Jetson itself, with version-compatible tools and no system package replacement. Compare saved model predictions/geometry against the CPU baseline on identical frames; benchmark batch=1, same input resolution, warmup, FPS/p50/p95, memory and tegrastats. Do not invent FPS or claim a speedup without reproducible logs. If incompatibility occurs, preserve working CPU path and write a minimal blocker report. Report gate and test commands.
```

## Phase 7 — Publishable reproducibility package
```
Read Phase 7 in PLANS.md. Prepare a complete GitHub-ready README showing the architecture, attribution to public datasets/models, license checks, exact small-dataset fetch workflow, one-command synthetic validation, one-command real-subset demo, accurate benchmark methodology/results and limits. Generate diagrams only from actual code structure and visuals only from real processed samples. Verify .gitignore prevents raw datasets, secrets, weights and TensorRT engine files from appearing in a public commit. Run all tests and checks. Create a final status report with evidence for each claim and a short LinkedIn-ready engineering summary with measured fields only. Do not push or publish unless I explicitly approve.
```

## Corrective prompt — Use whenever results fail acceptance
```
The current phase has NOT passed acceptance. Do not add features or rewrite unrelated modules. Read the phase acceptance criteria and AGENTS.md. First reproduce the failing command/test, then write an evidence-based diagnosis containing: expected vs actual result, exact input file, dataset/model/software version, minimal reproducible case, suspected root cause(s) ordered by likelihood, and whether the issue is environment, data, time sync, calibration, model or performance. Test ONE hypothesis with the smallest code change. Add a regression test that fails before the change and passes after it when possible. Re-run the original failing command and the full available test suite. Show changed files and before/after metrics; never silently loosen thresholds, fabricate depth, replace missing frames or distort camera calibration. After two unsuccessful evidence-driven attempts, STOP and write a blocker report identifying what user approval or new evidence is required. Do not proceed to the next phase unless PASS is proven.
```

## Quality audit prompt — Use after each successful phase
```
Review the just-completed phase as a skeptical independent senior computer vision engineer. Check for incorrect calibration frame directions, meters/mm mismatches, 960x540 vs 1920x1080 intrinsics, fisheye/rectification double application, depth unit errors, ignored FFC/timestamp skew, missing occlusion policy, hidden synthetic fallbacks, missing labels, numerical bugs, invalid p95/FPS calculations, dataset license omissions, false claims of real camera latency, and giant files accidentally tracked by Git. Cite exact file paths and line numbers for issues. Fix confirmed problems within current phase scope, run tests, update reports, then conclude PASS or BLOCKED.
```
