"""Report Jetson hardware, software versions and runtime dependencies."""

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/jetson_environment.json")
    args = parser.parse_args()
    report = {"os": platform.platform(), "architecture": platform.machine(), "python": platform.python_version(),
              "is_jetson": Path("/etc/nv_tegra_release").exists(), "disk_free_bytes": shutil.disk_usage(".").free,
              "packages": {}, "executables": {}, "device_changes": "none"}
    for name in ["torch", "tensorrt", "onnxruntime", "numpy", "opencv-python"]:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = "not found in Python package metadata; system packages may differ"
    for filename in ["/etc/nv_tegra_release", "/proc/meminfo", "/proc/device-tree/model", "/etc/os-release"]:
        path = Path(filename)
        if path.exists():
            report[filename] = path.read_text().strip("\x00\n")
    for name in ["trtexec", "tegrastats", "nvcc", "nvpmodel", "jetson_clocks"]:
        report["executables"][name] = shutil.which(name)
    for name, filename in [("trtexec", "/usr/src/tensorrt/bin/trtexec"), ("nvcc", "/usr/local/cuda/bin/nvcc")]:
        if not report["executables"][name] and Path(filename).is_file():
            report["executables"][name] = filename
    report["imports"] = {}
    for name in ["numpy", "cv2", "onnxruntime", "tensorrt", "cuda"]:
        try:
            module = importlib.import_module(name)
            report["imports"][name] = {"version": getattr(module, "__version__", None), "path": module.__file__}
        except ImportError as error:
            report["imports"][name] = {"error": str(error)}
    if report["executables"]["nvcc"]:
        result = subprocess.run([report["executables"]["nvcc"], "--version"], capture_output=True, text=True)
        report["nvcc_version"] = result.stdout
    report["hardware_compatibility"] = "jetson" if report["is_jetson"] else "non_jetson"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
