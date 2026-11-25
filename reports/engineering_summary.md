# Engineering summary

Implemented a reproducible CPU pipeline for time-matched RGB, thermal and FoundationStereo estimated depth using a pinned public TartanRGBT sequence. It audits timestamps and FFC flags, runs YOLOX Nano, calculates RGB optical-frame box-depth positions, renders RGB-colored 3D, and records per-frame timings.

On one Windows CPU baseline run, 296 valid groups completed at 64.32 frames/s, with processing p95 27.86 ms. A separate timestamp-paced run reported 34.31 ms p95 replay-completion latency and no valid frame drops. Thirteen synthetic/integration tests pass in both working and fresh environments. The real-data H.264 demonstration is 43.5 seconds and 4.52 MB.

The dataset's actual sample intervals are irregular despite its nominal 10 Hz release. Detector output includes visible false positives. Thermal 3D registration is deliberately blocked until the released 8-bit image projection is evidenced. No detection mAP, independent depth accuracy, physical camera latency, GPU speedup or Jetson results are claimed. Data/model authorship remains with CMU AirLab/AnyThermal, FoundationStereo and Megvii/YOLOX; implementation and validation are this project's contributions.
