"""Sustain full replay with persistent GPU contexts, bounded queues and telemetry."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
from threading import Event, Thread
import time

import cv2
import numpy as np
import yaml

from multimodal_perception.dataset import open_dataset
from multimodal_perception.joint_detection import load_joint_detector, close_joint_detector
from multimodal_perception.joint_pipeline import run_loaded_joint
from multimodal_perception.thermal_fusion import load_fusion_calibration


def current_rss_kib():
    return int(re.search(r"VmRSS:\s+(\d+)", Path("/proc/self/status").read_text()).group(1))


def benchmark(config_path, minimum_seconds, repeats):
    if minimum_seconds < 0 or repeats < 1:
        raise ValueError("Duration must be nonnegative and repeats positive")
    folder = Path("reports/jetson/tartan_max")
    videos = Path("results/jetson/tartan_max")
    folder.mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(config_path).read_text())
    cv2.setNumThreads(config["opencv_threads"])
    baseline = json.loads(Path("reports/jetson/tartan_fixed/fixed_corrected.outputs.json").read_text())
    dataset = open_dataset(config["dataset_root"], config["max_time_error_s"])
    calibration = load_fusion_calibration(config["dataset_root"], config["thermal_profile"], config["thermal_refinement"])
    rows, detectors, clocks = [], [], []
    timed_seconds = 0.
    stop = Event()

    def sample_clock():
        while not stop.is_set():
            clocks.append({"monotonic_s": time.monotonic(),
                           "gpu_hz": int(Path("/sys/class/devfreq/17000000.gpu/cur_freq").read_text()),
                           "rss_kib": current_rss_kib()})
            stop.wait(.5)

    started = time.perf_counter()
    with (folder/"tegrastats.log").open("w") as log:
        monitor = subprocess.Popen(["tegrastats", "--interval", "500"], stdout=log, stderr=subprocess.STDOUT)
        sampler = Thread(target=sample_clock, daemon=True)
        sampler.start()
        try:
            for _ in range(config.get("inference_workers", 1)):
                detector = load_joint_detector(config["model"], config.get("engine"), config["backend"], config["onnx_threads"])
                detectors.append(detector)
                detector["trt"]["cuda_graph_requested"] = config.get("cuda_graph", False)
            counts = {"headless": 0, "h264": 0}
            while min(counts.values()) < repeats or timed_seconds < minimum_seconds:
                mode = "headless" if counts["headless"] < repeats and counts["headless"] <= counts["h264"] else "h264"
                counts[mode] += 1
                name = f"sustained_{mode}_{counts[mode]:02d}"
                video = videos/"tartan_max_fp32.mp4" if mode == "h264" else None
                path = folder/f"{name}.json"
                report = run_loaded_joint(dataset, calibration, detectors[0], config, 0, video, path, detectors[1:])
                actual = json.loads(path.with_suffix(".outputs.json").read_text())
                changes = [a["index"] for a, b in zip(actual, baseline) if a != b]
                if len(actual) != len(baseline) or changes:
                    raise AssertionError(f"Output regression in {name}: {changes}")
                timed_seconds += report["elapsed_wall_s"]
                row = {"mode": mode, "repeat": counts[mode], "report": str(path),
                       "fps": report["pipeline_fps"], "frames": report["timed_frames"],
                       "elapsed_wall_s": report["elapsed_wall_s"],
                       "peak_rss_kib": report["process_peak_rss_kib"], "current_rss_kib": current_rss_kib(),
                       "baseline_output_changed_frames": changes,
                       "input_to_output_p50_p95_ms": [report["stages"]["input_to_output_ms"][k] for k in ["p50", "p95"]]}
                if video:
                    probe = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(video)], check=True, capture_output=True, text=True)
                    metadata = json.loads(probe.stdout)
                    stream = next(s for s in metadata["streams"] if s["codec_type"] == "video")
                    assert int(stream["nb_frames"]) == len(actual) and stream["codec_name"] == "h264"
                    row["video_frames"] = int(stream["nb_frames"])
                    (folder/f"{name}_video_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
                rows.append(row)
                (folder/"sustained_progress.json").write_text(json.dumps({"runs": rows, "timed_processing_seconds": timed_seconds}, indent=2)+"\n")
                print("SUSTAINED", json.dumps(row), flush=True)
        finally:
            for detector in detectors:
                close_joint_detector(detector)
            stop.set()
            sampler.join(timeout=5)
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
    raw = (folder/"tegrastats.log").read_text()
    temperatures = {}
    for name, value in re.findall(r"(\w+)@([\d.]+)C", raw):
        temperatures.setdefault(name, []).append(float(value))
    rails = {}
    for rail in ["VDD_IN", "VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"]:
        values = [float(x)/1000 for x in re.findall(rail+r" (\d+)mW", raw)]
        if values:
            rails[rail] = {"p50": float(np.median(values)), "max": max(values)}
    summary = {"runs": rows, "total_valid_groups": sum(r["frames"] for r in rows),
               "timed_processing_seconds": timed_seconds, "minimum_requested_seconds": minimum_seconds,
               "session_wall_s": time.perf_counter()-started, "persistent_contexts_reused": len(detectors),
               "fps": {mode: {"min": min(r["fps"] for r in rows if r["mode"] == mode),
                              "median": float(np.median([r["fps"] for r in rows if r["mode"] == mode])),
                              "max": max(r["fps"] for r in rows if r["mode"] == mode)} for mode in ["headless", "h264"]},
               "peak_temperatures_c": {name: max(values) for name, values in temperatures.items()},
               "individual_power_rails_w": rails,
               "observed_gpu_clocks_hz": sorted(set(row["gpu_hz"] for row in clocks)),
               "sampled_peak_process_rss_kib": max(row["rss_kib"] for row in clocks),
               "all_outputs_exactly_match_original_fp32": True,
               "scope": "Same two FP32 engine contexts reused for all rounds; warmup/setup excluded from timed_processing_seconds. All source valid frames decoded with bounded cache; no frame drops. H264 includes dashboard render, hardware encoding, queue drain and MP4 finalization. Offline supplied depth; no online depth generation or acquisition timing."}
    (folder/"clock_memory_samples.json").write_text(json.dumps(clocks, indent=2)+"\n")
    (folder/"sustained_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    shutil.copyfile(videos/"tartan_max_fp32.encoder.log", folder/"final_encoder.log")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tartan_joint_max_fp32.yaml")
    parser.add_argument("--minimum-seconds", type=float, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    benchmark(args.config, args.minimum_seconds, args.repeats)
