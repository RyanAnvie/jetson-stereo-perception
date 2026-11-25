"""Read-only device probe. Run locally on the explicitly authorized Jetson."""

import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess


if __name__ == "__main__":
    report = {"os": platform.platform(), "architecture": platform.machine(), "python": platform.python_version(),
              "is_jetson": Path("/etc/nv_tegra_release").exists(), "disk_free_bytes": shutil.disk_usage(".").free,
              "packages": {}, "executables": {}, "device_changes": "none"}
    for name in ["torch", "tensorrt", "onnxruntime", "numpy", "opencv-python"]:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = "not found in Python package metadata; system packages may differ"
    for filename in ["/etc/nv_tegra_release", "/proc/meminfo"]:
        path = Path(filename)
        if path.exists():
            report[filename] = path.read_text()
    for name in ["trtexec", "tegrastats", "nvcc", "nvpmodel", "jetson_clocks"]:
        report["executables"][name] = shutil.which(name)
    if report["executables"]["nvcc"]:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        report["nvcc_version"] = result.stdout
    report["gate"] = "ENVIRONMENT_INSPECTED" if report["is_jetson"] else "BLOCKED_NOT_ON_JETSON"
    Path("reports").mkdir(exist_ok=True)
    Path("reports/jetson_environment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
