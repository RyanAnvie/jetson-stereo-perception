"""Build YOLOX or LCAFNet engines on the target using its TensorRT 10 installation."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time


def build_engines(model, output_dir, report_dir, trtexec, precision):
    import tensorrt as trt

    if trt.__version__.split(".")[0] != "10":
        raise RuntimeError("This build recipe is validated for TensorRT 10 only")
    if platform.machine() != "aarch64" or not Path("/etc/nv_tegra_release").exists():
        raise RuntimeError("Build on the target Jetson, not the control computer")
    model = Path(model)
    output_dir, report_dir = Path(output_dir), Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    help_result = subprocess.run([trtexec, "--help"], capture_output=True, text=True, check=True)
    help_text = help_result.stdout + help_result.stderr
    for flag in ["--noTF32", "--fp16", "--exportLayerInfo", "--skipInference"]:
        if flag not in help_text:
            raise RuntimeError(f"Installed trtexec does not support {flag}")
    (report_dir / "trtexec_help.txt").write_text(help_text)
    for kind in (["fp32", "fp16"] if precision == "all" else [precision]):
        engine = output_dir / f"{model.stem}_{kind}.engine"
        if engine.exists():
            raise FileExistsError(f"Refusing to replace an existing engine: {engine}")
        command = [trtexec, f"--onnx={model}", f"--saveEngine={engine}", "--noTF32",
                   "--profilingVerbosity=detailed", "--skipInference",
                   f"--exportLayerInfo={report_dir / ('layers_' + kind + '.json')}"]
        if kind == "fp16":
            command.append("--fp16")
        print("Building", kind, flush=True)
        started = time.time()
        with (report_dir / f"build_{kind}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        metadata = {"model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                    "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
                    "precision": "fp32_no_tf32" if kind == "fp32" else "fp16_enabled_no_tf32",
                    "precision_note": "FP16 enabled permits FP32 fallback; consult exported layer information",
                    "tensorrt": trt.__version__, "command": command,
                    "build_started_unix_s": started, "build_elapsed_s": time.time() - started,
                    "device_model": Path("/proc/device-tree/model").read_text().strip("\x00\n"),
                    "l4t": Path("/etc/nv_tegra_release").read_text()}
        engine.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/weights/yolox_nano.onnx")
    parser.add_argument("--output-dir", default="models/weights/jetson")
    parser.add_argument("--report-dir", default="reports/jetson")
    parser.add_argument("--trtexec", default="/usr/src/tensorrt/bin/trtexec")
    parser.add_argument("--precision", choices=["all", "fp32", "fp16"], default="all")
    args = parser.parse_args()
    build_engines(args.model, args.output_dir, args.report_dir, args.trtexec, args.precision)
