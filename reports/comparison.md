# Measured PC results

Each row is a separate single run on the same 296 valid frames, YOLOX Nano FP32, CPU, 4 inference threads. No optimized implementation or speedup was measured.

| Condition | Wall throughput (frames/s) | Processing p50 / p95 (ms) | Total frame p95 (ms) | Peak process memory (MiB) |
| --- | ---: | ---: | ---: | ---: |
| benchmark_pc | 64.32 | 6.07 / 27.86 | 34.82 | 163.4 |
| benchmark_paced | 8.23 | 6.33 / 26.72 | 33.81 | 163.7 |
| benchmark_render | 21.05 | 6.29 / 25.89 | 63.17 | 194.0 |

Paced playback: 0 output deadlines above 100 ms; playback completion latency p95 34.31 ms. Lower paced throughput follows source timestamps (about 8.21 Hz over the full sequence); no valid source frames were dropped.

The render run includes dashboard rendering and MJPEG intermediate encoding. Final H.264 transcoding and title/outro cards are outside its timed interval. Processing excludes I/O, rendering and encoding; throughput includes those selected operations. All runs exclude five warmup calls. These measurements concern the RGB/depth pipeline with separate thermal viewing, not calibrated thermal fusion.

| Unmeasured item | Status |
| --- | --- |
| Detector mAP, precision, recall | No trusted labels evaluated |
| Independent depth MAE | No independent depth reference |
| Thermal registration error | Calibration domain unverified |
| Jetson / TensorRT | No device access |
| Physical camera E2E latency | Dataset replay only |
