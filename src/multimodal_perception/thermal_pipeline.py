import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from multimodal_perception.dashboard import title_card, text
from multimodal_perception.dataset import open_dataset, load_frame
from multimodal_perception.detection import load_detector, close_detector, detect
from multimodal_perception.geometry import locate_detection
from multimodal_perception.replay import environment, summarize_timings
from multimodal_perception.thermal_dashboard import render_fusion_dashboard
from multimodal_perception.thermal_fusion import load_fusion_calibration, fuse_thermal


def process_fused_sample(sample, detector, calibration, config):
    started = time.perf_counter()
    detections, timing = detect(sample["rgb"], detector, config["score_threshold"], config["nms_iou"])
    geometry_started = time.perf_counter()
    for detection in detections:
        detection["position"] = locate_detection(detection["bbox_xyxy"], sample["depth"], calibration["rgb"], config["min_depth_points"])
    fusion_started = time.perf_counter()
    fusion = fuse_thermal(sample, detections, calibration, config["fusion_stride"],
                          config["min_thermal_points"], config["max_time_error_s"])
    finished = time.perf_counter()
    timing.update(geometry_ms=(fusion_started-geometry_started)*1000,
                  fusion_ms=(finished-fusion_started)*1000, processing_ms=(finished-started)*1000)
    return detections, fusion, timing


def run_fusion(config_path="configs/tartan_fusion.yaml", limit=0, video=None, paced=False,
               report_path="reports/jetson/tartan/fp32_fused.json", backend=None, engine=None):
    config = yaml.safe_load(Path(config_path).read_text())
    if backend is not None:
        config["backend"] = backend
        if backend == "onnxruntime":
            config["engine"] = None
    if engine is not None:
        config["engine"] = engine
    if limit < 0 or config["warmup_frames"] < 0 or config["display_depth_max_m"] <= 0:
        raise ValueError("Invalid frame count, warmup, or display depth range")
    if config.get("video_timeline", "source_time") not in ["source_time", "compact_valid"]:
        raise ValueError("Unknown video timeline policy")
    cv2.setNumThreads(config["opencv_threads"])
    dataset = open_dataset(config["dataset_root"], config["max_time_error_s"])
    calibration = load_fusion_calibration(config["dataset_root"], config["thermal_profile"])
    detector = load_detector(config["model"], config["onnx_threads"], config["backend"], config.get("engine"))
    try:
        return run_loaded_fusion(dataset, calibration, detector, config, limit, video, paced, report_path)
    finally:
        close_detector(detector)


def run_loaded_fusion(dataset, calibration, detector, config, limit, video, paced, report_path):
    indices = np.flatnonzero(dataset["valid"]).tolist()
    if limit:
        indices = indices[:limit]
    if not indices:
        raise ValueError("No valid time-matched groups")
    for index in indices[:config["warmup_frames"]]:
        process_fused_sample(load_frame(dataset, index), detector, calibration, config)
    runtime = f"YOLOX / {detector['metadata']['backend']} / {detector['metadata']['precision']}"
    writer = None
    rows, outputs, dropped = [], [], []
    if video:
        video = Path(video)
        video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 800))
        if not writer.isOpened():
            raise RuntimeError("Could not open video writer")
    try:
        if writer is not None:
            card = title_card("RGB + thermal, in 3D", [
                "Real TartanRGBT replay on NVIDIA Jetson AGX Orin.",
                "RGB detections -> estimated depth -> visible thermal pixels.",
                "Official fisheye rectification reconstructed; CAD nominal pose.",
                "Thermal colors show processed intensity, NOT temperature.",
                "Registration is approximate; detector false positives remain.",
                ("Valid samples at 10 fps; gaps cut; source timestamps shown."
                 if config.get("video_timeline") == "compact_valid"
                 else "10 Hz source timeline; benchmark reports measure actual speed.")])
            for _ in range(20):
                writer.write(card)
        first_timestamp = float(dataset["target"][indices[0]])
        start = time.perf_counter()
        video_tick, last_canvas, held_video_frames = 0, None, 0
        for position, index in enumerate(indices):
            offset = float(dataset["target"][index])-first_timestamp
            if paced:
                delay = start + offset - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.1:
                    dropped.append({"index": index, "reason": "more_than_100ms_late", "lateness_ms": -delay*1000})
                    continue
            frame_start = time.perf_counter()
            sample = load_frame(dataset, index)
            loaded = time.perf_counter()
            detections, fusion, timing = process_fused_sample(sample, detector, calibration, config)
            output_ready = time.perf_counter()
            timing["io_decode_ms"] = (loaded-frame_start)*1000
            if paced:
                timing["playback_latency_ms"] = (output_ready-start-offset)*1000
            render_start = time.perf_counter()
            canvas = None
            if writer is not None:
                canvas = render_fusion_dashboard(sample, detections, fusion, timing, offset, runtime, config["display_depth_max_m"])
            timing["render_ms"] = (time.perf_counter()-render_start)*1000
            encode_start = time.perf_counter()
            if writer is not None:
                target_tick = (video_tick if config.get("video_timeline") == "compact_valid" else round(offset*10))
                while video_tick < target_tick and last_canvas is not None:
                    held = last_canvas.copy()
                    cv2.rectangle(held, (0, 300), (1280, 365), (24, 24, 70), -1)
                    text(held, "SOURCE GAP / INVALID GROUP EXCLUDED - previous valid frame held", 55, 340, 0.70)
                    writer.write(held)
                    video_tick += 1
                    held_video_frames += 1
                writer.write(canvas)
                video_tick += 1
                last_canvas = canvas
                if detections or position in [0, len(indices)//2, len(indices)-1]:
                    cv2.imwrite(str(video.parent / f"frame_{index:04d}.jpg"), canvas)
            timing["encode_ms"] = (time.perf_counter()-encode_start)*1000
            timing["total_frame_ms"] = (time.perf_counter()-frame_start)*1000
            rows.append({"index": index, "timestamp_s": sample["timestamp_s"], **timing})
            outputs.append({"index": index, "timestamp_s": sample["timestamp_s"],
                            "rgb_timestamp_s": sample["rgb_timestamp_s"], "thermal_timestamp_s": sample["thermal_timestamp_s"],
                            "detections": detections, "sampled_depth_points": fusion["sampled_depth_points"],
                            "visible_thermal_samples": len(fusion["points"]), "thermal_status": fusion["status"],
                            "frame": calibration["rgb"]["frame"], "position_units": "m",
                            "thermal_units": "processed_intensity_u8"})
            if position % 50 == 0:
                print(f"Processed {len(rows)}/{len(indices)} | frame {index} | {timing['processing_ms']:.1f} ms", flush=True)
        elapsed = time.perf_counter()-start
        report = summarize_timings(rows, elapsed)
        if writer is not None:
            card = title_card("Measured on Jetson", [
                f"{len(rows)} valid groups. {runtime}",
                f"Render-run throughput {report['pipeline_fps']:.2f} groups/s; video plays at 10 Hz.",
                "Thermal footprints and colored 3D use the same projected points.",
                ("FFC and bad timestamps excluded; gaps cut from this video."
                 if config.get("video_timeline") == "compact_valid"
                 else "FFC and bad timestamps excluded; gaps hold the previous view."),
                "CAD pose + estimated depth: no pixel-perfect or metric accuracy claim.",
                "Data: CMU AirLab TartanRGBT. Detector: Megvii YOLOX."])
            for _ in range(20):
                writer.write(card)
    finally:
        if writer is not None:
            writer.release()
    env = environment()
    env["compute_device"] = "NVIDIA Jetson GPU + CPU" if detector["backend"] == "tensorrt" else "CPU"
    if detector["backend"] == "tensorrt":
        env.pop("available_gpu_not_used", None)
    report.update(environment=env, config=config, model=detector["metadata"], calibration=calibration["evidence"],
                  dataset_revision=dataset["manifest"]["revision"], sequence=dataset["manifest"]["sequence"],
                  warmup_frames=min(config["warmup_frames"], len(indices)), selected_valid_groups=len(indices),
                  source_groups=len(dataset["target"]), invalid_source_groups=int((~dataset["valid"]).sum()),
                  excluded=[{"index": i, "reasons": dataset["reasons"][i]} for i in np.flatnonzero(~dataset["valid"]).tolist()],
                  dropped_valid_frames=len(dropped), dropped=dropped, replay_paced=paced,
                  missed_100ms_deadlines=sum(r.get("playback_latency_ms", 0) > 100 for r in rows) if paced else None,
                  total_detections=sum(len(o["detections"]) for o in outputs),
                  thermal_associations=sum(d["thermal"]["status"] == "associated_nominal" for o in outputs for d in o["detections"]),
                  visible_thermal_samples_p50=int(np.median([o["visible_thermal_samples"] for o in outputs])),
                  includes_render=bool(video), includes_video_encode=bool(video),
                  video_timeline=config.get("video_timeline", "source_time") if video else None,
                  held_video_frames=held_video_frames if video else None,
                  source_gap_ticks_removed=(max(0, round((outputs[-1]["timestamp_s"]-first_timestamp)*10)+1-len(outputs))
                                            if video and config.get("video_timeline") == "compact_valid" else 0),
                  processing_definition="loaded RGB/thermal/depth -> RGB detections + estimated 3D + thermal associations; excludes I/O/render/encode",
                  fps_definition="completed valid groups / timed wall duration, including I/O and selected render mode; excludes title cards",
                  thermal_detector="none; RGB detections associated geometrically, not thermal-only detection",
                  accuracy="unmeasured detector accuracy; source-reconstructed projection with CAD nominal extrinsics",
                  acquisition_latency="unmeasured; dataset replay", depth_source="released FoundationStereo estimates; not computed live")
    report["meets_10hz_headless"] = not video and report["pipeline_fps"] >= 10 and report["stages"]["total_frame_ms"]["p95"] <= 100
    report["meets_10hz_paced"] = paced and not dropped and report["missed_100ms_deadlines"] == 0
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    report_path.with_suffix(".outputs.json").write_text(json.dumps(outputs, indent=2) + "\n")
    with report_path.with_suffix(".csv").open("w", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    print(json.dumps({k: report[k] for k in ["timed_frames", "pipeline_fps", "total_detections", "thermal_associations", "dropped_valid_frames", "missed_100ms_deadlines"]}, indent=2))
    return report
