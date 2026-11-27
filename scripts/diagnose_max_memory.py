"""Separate retained host allocator pages from live replay buffers between rounds.

Diagnostic only: collection and malloc_trim are outside all measured frame work
and are not added to the production pipeline.
"""
import argparse
import ctypes
import gc
import json
from pathlib import Path

import cv2
import yaml

from benchmark_tartan_max import current_rss_kib
from multimodal_perception.dataset import open_dataset
from multimodal_perception.joint_detection import load_joint_detector, close_joint_detector
from multimodal_perception.joint_pipeline import run_loaded_joint
from multimodal_perception.thermal_fusion import load_fusion_calibration
from multimodal_perception.tensorrt_inference import cuda_check


def diagnose(render=False):
    from cuda import cudart
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim.argtypes = [ctypes.c_size_t]
    libc.malloc_trim.restype = ctypes.c_int
    config = yaml.safe_load(Path("configs/tartan_joint_max_fp32.yaml").read_text())
    cv2.setNumThreads(config["opencv_threads"])
    dataset = open_dataset(config["dataset_root"], config["max_time_error_s"])
    calibration = load_fusion_calibration(config["dataset_root"], config["thermal_profile"], config["thermal_refinement"])
    baseline = json.loads(Path("reports/jetson/tartan_fixed/fixed_corrected.outputs.json").read_text())
    root = Path("reports/jetson/tartan_max/memory_diagnostic_render" if render else "reports/jetson/tartan_max/memory_diagnostic")
    root.mkdir(parents=True, exist_ok=True)
    detectors, rows = [], []
    try:
        for _ in range(config["inference_workers"]):
            detector = load_joint_detector(config["model"], config["engine"], config["backend"], config["onnx_threads"])
            detectors.append(detector)
            detector["trt"]["cuda_graph_requested"] = True
        for index in range(6):
            path = root/f"round_{index+1}.json"
            video = Path("results/jetson/tartan_max/memory_diagnostic.mp4") if render else None
            report = run_loaded_joint(dataset, calibration, detectors[0], config, 0, video, path, detectors[1:])
            actual = json.loads(path.with_suffix(".outputs.json").read_text())
            assert actual == baseline
            del actual
            before = current_rss_kib()
            collected = gc.collect()
            after_gc = current_rss_kib()
            trimmed = libc.malloc_trim(0)
            after_trim = current_rss_kib()
            free, total = cuda_check(cudart.cudaMemGetInfo())
            row = {"round": index+1, "rss_before_kib": before, "rss_after_gc_kib": after_gc,
                   "rss_after_trim_kib": after_trim, "unreachable_objects_collected": collected,
                   "malloc_trim_released_pages": bool(trimmed), "cuda_free_bytes": int(free),
                   "cuda_total_bytes": int(total), "output_exact": True, "fps": report["pipeline_fps"]}
            rows.append(row)
            print("MEMORY", json.dumps(row), flush=True)
    finally:
        for detector in detectors:
            close_joint_detector(detector)
    (root/"summary.json").write_text(json.dumps({"runs": rows,
        "includes_video": render,
        "scope": "Diagnostic replay; two persistent contexts. gc.collect and glibc malloc_trim run only between sequences, after frame timing. No production allocator policy changed."}, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    diagnose(parser.parse_args().render)
