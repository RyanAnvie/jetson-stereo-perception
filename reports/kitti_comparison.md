# KITTI measured results

Windows / CPU / FP32 / YOLOX Nano / native 1242x375 stereo. Each row is one measured run; five warmup groups excluded. These are different execution conditions, not an optimized-vs-baseline speedup comparison.

| Run | Completed / dropped | Wall FPS | Processing p95 (ms) | Replay completion p95 (ms) | Peak working set (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| kitti_benchmark | 154 / 0 | 14.14 | 59.34 | not paced | not recorded |
| kitti_paced_10hz | 154 / 0 | 10.02 | 59.40 | 74.18 | 156.7 |
| kitti_overload_30hz | 83 / 71 | 16.06 | 59.97 | 95.31 | 156.6 |
| kitti_evaluation | 154 / 0 | 12.29 | 53.10 | not paced | 271.0 |

| Reference scope | Valid common pixels | MAE (m) | RMSE (m) | Median absolute error (m) | p95 absolute error (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| depth_all_common_pixels | 2,322,696 | 1.8570 | 12.2269 | 0.2311 | 6.5296 |
| depth_matched_object_surface_pixels | 97,334 | 0.9549 | 2.9378 | 0.3550 | 3.1544 |

Stereo coverage at all projected LiDAR pixels: 76.98%. Matched detection/tracklet instances: 216 of 259 eligible label instances. These counts are diagnostics, not an official detector recall score.

All errors include large residuals. LiDAR scans are not motion compensated; these are surface-depth discrepancies, not object-center accuracy. See docs/KITTI.md for the exact pixel selection and class mapping.

Prediction output digest for unpaced, 10 Hz and offline runs: `980c9935dc1a1363d1f59e9114f05cdf76a1fe08aaa361949a72e0bea45f3120`. All three agree.

Inference in this extension computes stereo depth every frame. It cannot be compared directly to the earlier TartanRGBT pipeline, which loaded already-estimated depth from disk.
