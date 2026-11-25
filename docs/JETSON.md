# Jetson portability gate

For the current KITTI workflow, start with the Chinese [cross-computer handoff and Jetson AGX deployment guide](../JETSON_AGX_HANDOFF.md). It covers source/data transfer, Orin/Xavier environment branches, CPU reproduction and the TensorRT integration still required. The status notes below remain valid.

No Jetson was provided or accessed. The local probe correctly reports `BLOCKED_NOT_ON_JETSON`. No SSH commands, flashing, system package changes or TensorRT engine build were attempted.

On an explicitly authorized Jetson AGX Orin, copy the source using the owner's approved route, then run the standard-library-only read-only probe:

```bash
python3 scripts/inspect_jetson.py
```

Review `reports/jetson_environment.json`, `/etc/nv_tegra_release`, Python/architecture, installed TensorRT/CUDA/OpenCV/PyTorch and available `trtexec`/`tegrastats`. Python package metadata may omit system-installed libraries; verify imports on the actual device before choosing an install route.

Keep the existing JetPack libraries. Do not use the Windows PC lock file on Jetson. Select a compatible inference runtime only after this inspection. The current backend is specifically ONNX Runtime CPU; TensorRT support is not claimed or implemented against an unknown JetPack version.

The downloaded YOLOX model already supplies ONNX. A TensorRT engine must be built **on the target**, with supported options from that device's `trtexec --help`. Compare model predictions on the same frame indices and exact BGR preprocessing before adding engine inference to the replay. Record FP32/FP16 deviations, batch=1 timings, memory, power and clock policy. Do not report speedups without measured equivalent runs.

Thermal mapping has its separate calibration gate regardless of target hardware. Required evidence: the actual `/thermal_left/image_processed` rectification implementation or matching CameraInfo projection, followed by real correspondence validation.

Official references: [JetPack](https://docs.nvidia.com/jetson/jetpack/), [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/).
