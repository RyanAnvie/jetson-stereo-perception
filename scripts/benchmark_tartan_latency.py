"""Interleave full-video latency profiles; preserve all frames and exact outputs."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from multimodal_perception.joint_pipeline import run_joint


def benchmark(repeats):
    root = Path("reports/jetson/tartan_latency")
    root.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(Path("reports/published/reference_outputs.json").read_text())
    profiles = {"balanced": "configs/tartan_joint_latency.yaml",
                "responsive": "configs/tartan_joint_responsive.yaml"}
    if repeats < 1:
        raise ValueError("At least one repeat is required")
    rows, timings = [], {name: [] for name in profiles}
    for repeat in range(1, repeats+1):
        for name, config in profiles.items():
            path = root/f"{name}_{repeat}.json"
            video = Path("results/jetson/tartan_latency")/f"{name}.mp4"
            result = run_joint(config, 0, video, path)
            output = json.loads(path.with_suffix(".outputs.json").read_text())
            assert output == baseline, "Predictions differ from the published FP32 reference; inspect the generated outputs"
            assert result["held_video_frames"] == 0 and result["dropped_valid_frames"] == 0
            with path.with_suffix(".csv").open() as handle:
                samples = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
            for sample in samples:
                assert 0 <= sample["input_to_detection_ms"] <= sample["input_to_delivery_ms"] <= sample["input_to_encoder_ms"] <= sample["input_to_output_ms"]
                assert sample["admission_to_encoder_ms"] >= sample["input_to_encoder_ms"]
            timings[name].extend(samples)
            metadata = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)], check=True, capture_output=True, text=True).stdout)
            stream = next(item for item in metadata["streams"] if item["codec_type"] == "video")
            assert int(stream["nb_frames"]) == 296 and stream["codec_name"] == "h264"
            (root/f"{name}_{repeat}_video.json").write_text(json.dumps(metadata, indent=2)+"\n")
            rows.append({"profile": name, "repeat": repeat, "report": str(path), "frames": result["timed_frames"],
                         "fps": result["pipeline_fps"], "elapsed_wall_s": result["elapsed_wall_s"],
                         "output_exact": True, "latency": result["stages"]["input_to_output_ms"],
                         "peak_rss_kib": result["process_peak_rss_kib"]})
            (root/"benchmark_progress.json").write_text(json.dumps(rows, indent=2)+"\n")
            print("VERIFIED_LATENCY", json.dumps(rows[-1]), flush=True)
    metrics = ["input_to_output_ms", "input_to_encoder_ms", "admission_to_encoder_ms", "input_to_detection_ms", "input_to_delivery_ms",
               "preparation_active_ms", "preparation_queue_ms", "prepared_wait_ms", "inference_active_ms", "completed_wait_ms", "render_queue_wait_ms"]
    aggregates = {}
    for name in profiles:
        selected = [r for r in rows if r["profile"] == name]
        aggregates[name] = {"fps": sum(r["frames"] for r in selected)/sum(r["elapsed_wall_s"] for r in selected),
                            "fps_range": [min(r["fps"] for r in selected), max(r["fps"] for r in selected)],
                            "per_run_p95_range_ms": [min(r["latency"]["p95"] for r in selected), max(r["latency"]["p95"] for r in selected)],
                            "frames": len(timings[name]),
                            "latency": {key: {f"p{q}": float(np.percentile([r[key] for r in timings[name]], q)) for q in [50, 95, 99]} for key in metrics}}
    summary = {"runs": rows, "profiles": aggregates, "total_valid_frames": sum(r["frames"] for r in rows),
               "all_outputs_exact": True,
               "scope": "All frames retained in source order. Same FP32 engine, motion, depth, calibration and original preparation-start latency clock; full render + H264 drain in throughput. Physical capture/future-frame wait, online depth, encoder completion and display latency are not measured."}
    (root/"benchmark_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(aggregates, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    benchmark(parser.parse_args().repeats)
