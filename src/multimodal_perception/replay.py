import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import cv2
import numpy as np
import yaml

from multimodal_perception.camera import load_calibration
from multimodal_perception.dataset import open_dataset, load_frame
from multimodal_perception.detection import load_detector, detect
from multimodal_perception.geometry import depth_cloud, locate_detection
from multimodal_perception.dashboard import render_dashboard, title_card


def environment():
    versions = {}
    for name in ["numpy", "opencv-python", "PyYAML", "onnxruntime", "pytest", "huggingface_hub"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"os": platform.platform(), "python": platform.python_version(), "machine": platform.machine(),
              "cpu": platform.processor(), "logical_cpus": os.cpu_count(), "packages": versions,
              "free_disk_bytes": shutil.disk_usage(".").free, "compute_device": "CPU"}
    if shutil.which("nvidia-smi"):
        process = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
        result["available_gpu_not_used"] = process.stdout.strip()
    return result


def summarize_timings(rows, wall_seconds):
    if not rows or wall_seconds <= 0:
        raise ValueError("Benchmark needs measured frames and positive elapsed time")
    stages = {}
    for key in rows[0]:
        if key.endswith("_ms"):
            values = [row[key] for row in rows]
            stages[key] = {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}
    return {"timed_frames": len(rows), "elapsed_wall_s": wall_seconds,
            "pipeline_fps": len(rows)/wall_seconds, "stages": stages}


def process_sample(sample, camera, detector, config):
    start = time.perf_counter()
    detections, timing = detect(sample["rgb"], detector, config["score_threshold"], config["nms_iou"])
    geometry_start = time.perf_counter()
    points, pixels = depth_cloud(sample["depth"], camera, config["point_stride"])
    for detection in detections:
        detection["position"] = locate_detection(detection["bbox_xyxy"], sample["depth"], camera, config["min_depth_points"])
    finish = time.perf_counter()
    timing["geometry_ms"] = (finish-geometry_start)*1000
    timing["processing_ms"] = (finish-start)*1000
    return detections, points, pixels, timing


def run_replay(config_path="configs/default.yaml", limit=0, video=None, paced=False,
               report_path="reports/benchmark_pc.json", show=False):
    config = yaml.safe_load(Path(config_path).read_text())
    if config["thermal_mapping"] != "blocked_unverified_image_processed_projection":
        raise ValueError("Real thermal mapping requires new calibration evidence and validation")
    if limit < 0 or config["warmup_frames"] < 0 or config["display_depth_max_m"] <= 0:
        raise ValueError("Invalid frame count, warmup count, or display depth limit")
    dataset = open_dataset(config["dataset_root"], config["max_time_error_s"])
    camera, _ = load_calibration(Path(config["dataset_root"]) / "calibration")
    detector = load_detector(config["model"], config["onnx_threads"])
    indices = np.flatnonzero(dataset["valid"]).tolist()
    if limit:
        indices = indices[:limit]
    if not indices:
        raise ValueError("No valid time-matched frames")
    warmup = config["warmup_frames"]
    for i in indices[:warmup]:
        process_sample(load_frame(dataset, i), camera, detector, config)
    writer = None
    if video:
        video = Path(video)
        video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 800))
        if not writer.isOpened():
            raise RuntimeError("OpenCV video writer could not open the output")
        card = title_card("Seeing in three modalities", ["Real TartanRGBT footage. 10 Hz presentation timeline.",
                           "RGB detection + metric estimated depth + thermal replay.",
                           f"{len(indices)} selected valid groups; {int(dataset['ffc'].sum())} source FFC groups excluded.",
                           "Thermal 3D registration is blocked pending calibration evidence.",
                           "Unverified detector predictions include visible false positives.",
                           "No independent depth accuracy or camera-latency claim."])
        for _ in range(30):
            writer.write(card)
    rows = []
    outputs = []
    first_timestamp = float(dataset["target"][indices[0]])
    start = time.perf_counter()
    missed_deadlines = 0
    video_tick = 0
    last_canvas = None
    try:
        for position, index in enumerate(indices):
            scheduled_offset = float(dataset["target"][index])-first_timestamp
            if paced:
                delay = start + scheduled_offset - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            frame_start = time.perf_counter()
            sample = load_frame(dataset, index)
            loaded = time.perf_counter()
            detections, points, pixels, timing = process_sample(sample, camera, detector, config)
            output_ready = time.perf_counter()
            timing["io_decode_ms"] = (loaded-frame_start)*1000
            render_start = time.perf_counter()
            canvas = None
            if writer is not None or show:
                canvas = render_dashboard(sample, detections, points, pixels, timing, scheduled_offset, config["display_depth_max_m"])
            timing["render_ms"] = (time.perf_counter()-render_start)*1000
            encode_start = time.perf_counter()
            if writer is not None:
                # Quantize actual source timestamps to a 10 fps video timeline.
                # A held frame represents the gap; never fabricate a source frame.
                target_tick = round(scheduled_offset*10)
                while video_tick < target_tick and last_canvas is not None:
                    writer.write(last_canvas)
                    video_tick += 1
                writer.write(canvas)
                video_tick += 1
                last_canvas = canvas
                if position in [0, len(indices)//2, len(indices)-1]:
                    cv2.imwrite(str(video.parent / f"frame_{index:04d}.jpg"), canvas)
            if show:
                cv2.imshow("Multimodal perception", canvas)
                stop_requested = cv2.waitKey(1) == 27
            else:
                stop_requested = False
            timing["encode_ms"] = (time.perf_counter()-encode_start)*1000
            timing["total_frame_ms"] = (time.perf_counter()-frame_start)*1000
            if paced:
                timing["playback_latency_ms"] = (output_ready-start-scheduled_offset)*1000
                if timing["playback_latency_ms"] > 100:
                    missed_deadlines += 1
            rows.append({"index": index, "timestamp_s": sample["timestamp_s"], **timing})
            outputs.append({"index": index, "timestamp_s": sample["timestamp_s"],
                            "rgb_timestamp_s": sample["rgb_timestamp_s"], "thermal_timestamp_s": sample["thermal_timestamp_s"],
                            "detections": detections, "valid_cloud_points": len(points),
                            "frame": camera["frame"], "units": "m", "thermal_status": "calibration_unverified"})
            if position % 50 == 0:
                print(f"Processed {position+1}/{len(indices)} | frame {index} | {timing['processing_ms']:.1f} ms", flush=True)
            if stop_requested:
                break
        elapsed = time.perf_counter()-start
        if writer is not None:
            summary = summarize_timings(rows, elapsed)
            card = title_card("Measured. Reproducible. Bounded.", [f"{len(rows)} real frames processed. YOLOX Nano / CPU / FP32.",
                               f"Processing p50 {summary['stages']['processing_ms']['p50']:.1f} ms / p95 {summary['stages']['processing_ms']['p95']:.1f} ms in this render run.",
                               "Full headless benchmark and per-frame samples: reports/",
                               "Detector has visible false positives; no labeled accuracy score.",
                               "Thermal mapping: unverified. Jetson: not measured.",
                               "Data: CMU AirLab TartanRGBT / AnyThermal (ICRA 2026).",
                               "Model: Megvii YOLOX. Source attributions: THIRD_PARTY.md"])
            for _ in range(45):
                writer.write(card)
    finally:
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    report = summarize_timings(rows, elapsed)
    report.update(environment=environment(), warmup_frames=min(warmup, len(indices)),
                  warmup_policy="First valid groups run before timer, then included again in measured sequence",
                  dataset_revision=dataset["manifest"]["revision"], sequence=dataset["manifest"]["sequence"],
                  first_index=rows[0]["index"], last_index=rows[-1]["index"], input_rgb_shape=[540, 960, 3],
                  model=detector["metadata"], config=config, replay_paced=paced,
                  includes_render=bool(video or show), includes_video_encode=bool(video),
                  invalid_source_groups=int((~dataset["valid"]).sum()), dropped_valid_frames=len(indices)-len(rows),
                  missed_100ms_deadlines=missed_deadlines if paced else None,
                  total_detections=sum(len(item["detections"]) for item in outputs),
                  person_detections=sum(d["class_id"] == 0 for item in outputs for d in item["detections"]),
                  thermal_mapping="BLOCKED: image_processed projection unverified", accuracy="not measured",
                  processing_definition="loaded frame -> detections + RGB-depth 3D ready; excludes I/O, rendering, encoding",
                  fps_definition="completed valid groups / timed wall duration, including I/O and selected render mode",
                  acquisition_latency="not measured; dataset replay only")
    report["meets_10hz_processing_target"] = report["pipeline_fps"] >= 10 and report["stages"]["processing_ms"]["p95"] <= 100
    if platform.system() == "Windows":
        process = subprocess.run(["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {os.getpid()}).PeakWorkingSet64"], capture_output=True, text=True)
        report["peak_process_memory_bytes"] = int(process.stdout.strip()) if process.returncode == 0 else None
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["peak_process_memory_bytes"] = peak if platform.system() == "Darwin" else peak*1024
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with report_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer_csv = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    report_path.with_name(report_path.stem+"_outputs.json").write_text(json.dumps(outputs, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"frames": len(rows), "fps": report["pipeline_fps"], "processing_p95_ms": report["stages"]["processing_ms"]["p95"], "detections": report["total_detections"]}, indent=2))
    return report
