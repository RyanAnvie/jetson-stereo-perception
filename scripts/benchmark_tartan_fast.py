"""Repeat the complete replay, including hardware video drain, with telemetry."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from multimodal_perception.joint_pipeline import run_joint


def benchmark(config, repeats):
    if repeats < 1:
        raise ValueError("At least one repeat is required")
    folder = Path("reports/jetson/tartan_fast")
    videos = Path("results/jetson/tartan_fast")
    folder.mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(Path("reports/jetson/tartan_fixed/fixed_corrected.outputs.json").read_text())
    rows = []
    started = time.perf_counter()
    with (folder/"tegrastats.log").open("w") as log:
        monitor = subprocess.Popen(["tegrastats", "--interval", "500"], stdout=log, stderr=subprocess.STDOUT)
        try:
            for repeat in range(1, repeats+1):
                for mode in ["headless", "h264"]:
                    name = f"{mode}_{repeat}"
                    video = videos/f"{name}.mp4" if mode == "h264" else None
                    path = folder/f"{name}.json"
                    report = run_joint(config, 0, video, path)
                    actual = json.loads(path.with_suffix(".outputs.json").read_text())
                    changes = [a["index"] for a, b in zip(actual, baseline) if a != b]
                    if len(actual) != len(baseline):
                        raise AssertionError("Valid frame count changed")
                    row = {"mode": mode, "repeat": repeat, "report": str(path),
                           "fps": report["pipeline_fps"], "frames": report["timed_frames"],
                           "peak_rss_kib": report["process_peak_rss_kib"],
                           "baseline_output_changed_frames": changes,
                           "frame_interval_p50_p95_ms": [report["stages"]["total_frame_ms"][k] for k in ["p50", "p95"]],
                           "input_to_output_p50_p95_ms": [report["stages"]["input_to_output_ms"][k] for k in ["p50", "p95"]]}
                    rows.append(row)
                    print("BENCHMARK", json.dumps(row), flush=True)
                    assert not changes
                    if video:
                        probe = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(video)], check=True, capture_output=True, text=True)
                        metadata = json.loads(probe.stdout)
                        stream = next(s for s in metadata["streams"] if s["codec_type"] == "video")
                        assert int(stream["nb_frames"]) == len(actual) and stream["codec_name"] == "h264"
                        (folder/f"{name}_video_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
        finally:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
    text = (folder/"tegrastats.log").read_text()
    temperatures = {}
    for name, value in re.findall(r"(\w+)@([\d.]+)C", text):
        temperatures.setdefault(name, []).append(float(value))
    power = [float(x)/1000 for x in re.findall(r"VDD_IN (\d+)mW", text)]
    rails = {}
    for rail in ["VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0"]:
        values = [float(x)/1000 for x in re.findall(rail+r" (\d+)mW", text)]
        if values:
            rails[rail] = {"p50": float(np.median(values)), "max": max(values)}
    gpu_clocks = re.findall(r"GR3D_FREQ\s+\d+%@\[?(\d+)", text)
    summary = {"runs": rows, "total_valid_groups": sum(r["frames"] for r in rows),
               "session_wall_s": time.perf_counter()-started,
               "fps": {mode: {"min": min(r["fps"] for r in rows if r["mode"] == mode),
                              "median": float(np.median([r["fps"] for r in rows if r["mode"] == mode])),
                              "max": max(r["fps"] for r in rows if r["mode"] == mode)} for mode in ["headless", "h264"]},
               "peak_temperatures_c": {name: max(values) for name, values in temperatures.items()},
               "vdd_in_w": {"p50": float(np.median(power)), "max": max(power)} if power else None,
               "individual_power_rails_w": rails,
               "gpu_clock_after_run_hz": int(Path("/sys/class/devfreq/17000000.gpu/cur_freq").read_text()),
               "observed_gpu_clock_mhz": sorted(set(map(int, gpu_clocks))),
               "all_outputs_exactly_match_original_fp32": all(not r["baseline_output_changed_frames"] for r in rows),
               "scope": "Warm models (3 frames); all valid groups decoded from source, bounded cache only; no frame drops. H264 includes full dashboard, hardware encode, queue drain and MP4 finalization. Offline supplied depth, no live depth network or acquisition latency measurement."}
    (folder/"sustained_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    shutil.copyfile(videos/f"h264_{repeats}.mp4", videos/"tartan_fast_fp32.mp4")
    shutil.copyfile(videos/f"h264_{repeats}.encoder.log", folder/"final_encoder.log")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tartan_joint_fast.yaml")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    benchmark(args.config, args.repeats)
