import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import cv2
import numpy as np
import yaml

from multimodal_perception.detection import detect, load_detector, close_detector
from multimodal_perception.geometry import locate_detection
from multimodal_perception.kitti import open_kitti, load_kitti_frame, read_kitti_timestamps
from multimodal_perception.stereo import create_stereo_matcher, estimate_stereo_depth
from multimodal_perception.kitti_evaluation import evaluate_kitti_frame, summarize_evaluation
from multimodal_perception.kitti_dashboard import render_kitti
from multimodal_perception.dashboard import title_card
from multimodal_perception.replay import environment, summarize_timings


def process_kitti_images(left, right, calibration, detector, matcher, config):
    start = time.perf_counter()
    detections, timing = detect(left, detector, config["score_threshold"], config["nms_iou"])
    stereo_start = time.perf_counter()
    depth, disparity = estimate_stereo_depth(left, right, matcher, calibration)
    geometry_start = time.perf_counter()
    for detection in detections:
        position = locate_detection(detection["bbox_xyxy"], depth, calibration["camera"], config["min_depth_points"])
        position.pop("thermal_status")
        position["units"] = "m"
        position["definition"] = "componentwise median of finite positive stereo points in predicted bbox; visible surface, not object center"
        detection["position"] = position
    ready = time.perf_counter()
    timing["stereo_ms"] = (geometry_start-stereo_start)*1000
    timing["geometry_ms"] = (ready-geometry_start)*1000
    timing["processing_ms"] = (ready-start)*1000
    return detections, depth, disparity, timing


def replay_position(offsets, position, elapsed_s, drop_late):
    """At most one decoded frame; optionally select the latest due source frame."""
    if not drop_late:
        return position
    due = int(np.searchsorted(offsets, elapsed_s, side="right")-1)
    return min(max(position, due), len(offsets)-1)


def run_kitti(config_path, report_path, mode="benchmark", paced=False, rate_hz=None, drop_late=False, video=None, limit=0,
              backend=None, engine=None):
    if mode not in ["benchmark", "evaluate"] or limit < 0:
        raise ValueError("Invalid mode or frame limit")
    if rate_hz is not None and (not np.isfinite(rate_hz) or rate_hz <= 0):
        raise ValueError("Replay input rate must be finite and positive")
    if (rate_hz is not None or drop_late) and not paced:
        raise ValueError("An input rate or drop policy requires --paced")
    if mode == "evaluate" and paced:
        raise ValueError("Offline reference evaluation is separate from paced runtime measurement")
    if video and mode != "evaluate":
        raise ValueError("Video requires --mode evaluate to display reference diagnostics")
    config = yaml.safe_load(Path(config_path).read_text())
    config["backend"] = backend or config.get("backend", "onnxruntime")
    engine = engine or config.get("engine")
    if engine:
        config["engine"] = engine
    cv2.setNumThreads(config["opencv_threads"])
    dataset = open_kitti(config)
    detector = load_detector(config["model"], config["onnx_threads"], config["backend"], engine)
    try:
        return run_loaded_kitti(dataset, detector, config, report_path, mode, paced, rate_hz, drop_late, video, limit)
    finally:
        close_detector(detector)


def run_loaded_kitti(dataset, detector, config, report_path, mode, paced, rate_hz, drop_late, video, limit):
    calibration = dataset["calibration"]
    runtime_environment = environment()
    if config["backend"] == "tensorrt":
        runtime_environment["compute_device"] = "CUDA GPU detector; CPU stereo and geometry"
        runtime_environment.pop("available_gpu_not_used", None)
    device_name = runtime_environment.get("device_model") or platform.machine()
    config["runtime_label"] = f"{device_name} | {config['backend']} / {detector['metadata']['precision']}"
    matcher = create_stereo_matcher(config["stereo"])
    indices = np.flatnonzero(dataset["valid"])
    if limit:
        indices = indices[:limit]
    if not len(indices):
        raise ValueError("No synchronized stereo frame groups")
    source_ns = dataset["streams"]["image_02"]["timestamps_ns"]
    source_offsets = (source_ns[indices]-source_ns[indices[0]])/1e9
    offsets = np.arange(len(indices))/rate_hz if rate_hz is not None else source_offsets
    warmup = min(config["warmup_frames"], len(indices))
    for index in indices[:warmup]:
        sample = load_kitti_frame(dataset, int(index), include_reference=False)
        process_kitti_images(sample["left"], sample["right"], calibration, detector, matcher, config)

    output_dir = Path("results/kitti")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if video:
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 800))
        if not writer.isOpened():
            raise RuntimeError("Video output failed to open")
        card = title_card("From detections to 3D positions", ["YOLOX Nano + calibrated RGB stereo. Real KITTI data.",
                           "Positions are estimated visible-surface representatives.",
                           "LiDAR and labels are used only in offline evaluation.",
                           config["runtime_label"],
                           "Offline replay; no sensor acquisition or object-center accuracy claim.",
                           "Data: Geiger et al. / KITTI / CC BY-NC-SA 3.0."])
        for _ in range(30):
            writer.write(card)
    rows = []
    evaluations = []
    skipped = []
    detections_total = 0
    valid_positions = 0
    output_hash = hashlib.sha256()
    log_path = report_path.with_name(report_path.stem+"_outputs.jsonl")
    position = 0
    video_tick = 0
    previous_canvas = None
    started = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as output_file:
            while position < len(indices):
                if paced:
                    selected = replay_position(offsets, position, time.perf_counter()-started, drop_late)
                    skipped.extend(int(i) for i in indices[position:selected])
                    position = selected
                    wait_s = started+offsets[position]-time.perf_counter()
                    if wait_s > 0:
                        time.sleep(wait_s)
                index = int(indices[position])
                frame_start = time.perf_counter()
                sample = load_kitti_frame(dataset, index, include_reference=(mode == "evaluate"))
                loaded = time.perf_counter()
                detections, depth, disparity, timing = process_kitti_images(sample["left"], sample["right"], calibration, detector, matcher, config)
                ready = time.perf_counter()
                timing["io_decode_ms"] = (loaded-frame_start)*1000
                timing["output_ready_ms"] = (ready-frame_start)*1000
                if paced:
                    timing["replay_completion_latency_ms"] = (ready-started-offsets[position])*1000
                    timing["start_lateness_ms"] = (frame_start-started-offsets[position])*1000
                evaluation = None
                if mode == "evaluate":
                    eval_start = time.perf_counter()
                    evaluation = evaluate_kitti_frame(sample, depth, detections, calibration, config)
                    timing["offline_evaluation_ms"] = (time.perf_counter()-eval_start)*1000
                    render_start = time.perf_counter()
                    canvas = None
                    if writer or position in [0, len(indices)//2, len(indices)-1]:
                        canvas = render_kitti(sample, depth, detections, evaluation, timing, source_offsets[position], config)
                    if position in [0, len(indices)//2, len(indices)-1]:
                        cv2.imwrite(str(output_dir/f"frame_{index:04d}.jpg"), canvas)
                    if writer:
                        tick = round(source_offsets[position]*10)
                        while video_tick < tick and previous_canvas is not None:
                            writer.write(previous_canvas)
                            video_tick += 1
                        writer.write(canvas)
                        video_tick += 1
                        previous_canvas = canvas
                    timing["render_encode_ms"] = (time.perf_counter()-render_start)*1000
                    if position == 0:
                        np.savez_compressed(output_dir / "first_frame_reference.npz", depth_m=depth, disparity_px=disparity,
                                            lidar_xyzi=sample["lidar_xyzi"], reference_pixels=evaluation["lidar"]["pixels_uv"])
                    # Aggregate scalar records and error vectors offline, not decoded frames.
                    for key in ["labels", "matches", "lidar"]:
                        evaluation.pop(key)
                    evaluations.append(evaluation)
                row = {"index": index, "source_timestamp_ns": sample["timestamp_ns"],
                       "schedule_offset_s": float(offsets[position]), **timing}
                rows.append(row)
                record = {"index": index, "source_timestamp_ns": sample["timestamp_ns"],
                          "right_timestamp_ns": sample["right_timestamp_ns"], "detections": detections,
                          "depth_source": "RGB StereoSGBM", "reference_inputs_used": False}
                line = json.dumps(record, allow_nan=False)
                output_file.write(line+"\n")
                output_hash.update(line.encode())
                detections_total += len(detections)
                valid_positions += sum(d["position"]["status"] == "ok" for d in detections)
                if position % 25 == 0:
                    print(f"{position+1}/{len(indices)} | frame {index} | {len(detections)} detections | {timing['processing_ms']:.1f} ms", flush=True)
                position += 1
        elapsed = time.perf_counter()-started
        evaluation_report = summarize_evaluation(evaluations) if evaluations else None
        if writer:
            metrics = evaluation_report["depth_all_common_pixels"]
            card = title_card("Geometry backed by measurements", [f"{len(rows)} real frames / {detections_total} predictions / {valid_positions} depth positions.",
                               f"Stereo vs raw LiDAR: MAE {metrics['mae_m']:.2f} m over {metrics['count']:,} common pixels.",
                               f"Reference-pixel coverage: {evaluation_report['stereo_coverage_at_lidar']*100:.1f}%. Invalid depth is reported.",
                               "Raw LiDAR has scan-motion and synchronization limitations.",
                               "Surface-depth error is not object-center accuracy.",
                               "Data: KITTI / Geiger et al. / CC BY-NC-SA 3.0.",
                               "Code, parameters and raw per-frame results accompany this video."])
            for _ in range(50):
                writer.write(card)
    finally:
        if writer:
            writer.release()
    report = summarize_timings(rows, elapsed)
    report.update(mode=mode, environment=runtime_environment, config=config, model=detector["metadata"],
                  dataset_manifest_sha256=hashlib.sha256((dataset["root"]/"sample_manifest.json").read_bytes()).hexdigest(),
                  sequence=config["sequence"], warmup_frames=warmup, warmup_policy="first groups before timing, then rerun in measured sequence",
                  frame_queue_capacity=1, dropped_valid_frames=len(skipped), dropped_indices=skipped,
                  invalid_stereo_groups=int((~dataset["valid"]).sum()), replay_paced=paced,
                  schedule="fixed-rate replay" if rate_hz else "source timestamps", requested_rate_hz=rate_hz,
                  source_timestamps_preserved=True, drop_late=drop_late,
                  missed_100ms_deadlines=int(sum(row.get("replay_completion_latency_ms", 0)>100 for row in rows)) if paced else None,
                  detections=detections_total, valid_positions=valid_positions, outputs_sha256=output_hash.hexdigest(),
                  includes_reference_work=mode=="evaluate", includes_video=bool(video),
                  definitions={"processing": "RGB frame loaded -> YOLO + stereo + surface positions ready; excludes reference evaluation and rendering",
                               "pipeline_fps": "processed frames / measured wall time, including I/O, selected offline work and pacing",
                               "replay_latency": "scheduled replay input -> output ready; not physical sensor latency"},
                  independent_object_center_accuracy="not measured",
                  jetson="measured on target" if runtime_environment.get("l4t") else "not measured")
    if platform.system() == "Windows":
        process = subprocess.run(["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {os.getpid()}).PeakWorkingSet64"], capture_output=True, text=True)
        report["peak_process_memory_bytes"] = int(process.stdout.strip()) if process.returncode == 0 else None
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["peak_process_memory_bytes"] = peak if platform.system() == "Darwin" else peak*1024
    if mode == "benchmark":
        report["meets_10fps_unpaced_target"] = report["pipeline_fps"] >= 10 and report["stages"]["processing_ms"]["p95"] <= 100 if not paced else None
    if evaluation_report:
        report["evaluation"] = evaluation_report
        frame_rows = [item["frame"] for item in evaluations]
        object_rows = [row for item in evaluations for row in item["objects"]]
        report_path.with_name(report_path.stem+"_errors.json").write_text(json.dumps({"frames": frame_rows, "objects": object_rows}, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with report_path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    times = source_ns
    velo_times = dataset["streams"]["velodyne_points"]["timestamps_ns"]
    scan_start = read_kitti_timestamps(dataset["directory"]/"velodyne_points/timestamps_start.txt")
    scan_end = read_kitti_timestamps(dataset["directory"]/"velodyne_points/timestamps_end.txt")
    audit = {"source_frames": len(times), "checked_frames_this_run": len(rows), "sequence": config["sequence"],
             "valid_stereo_groups": int(dataset["valid"].sum()), "tracklets": dataset["tracklet_count"],
             "shape": [calibration["camera"]["height"], calibration["camera"]["width"], 3],
             "baseline_x_m": calibration["baseline_x_m"], "K_cam2": calibration["camera"]["K"].tolist(),
             "frame": calibration["camera"]["frame"], "units": "m",
             "T_cam2_from_velo": {"R": calibration["T_cam2_from_velo"]["R"].tolist(), "t": calibration["T_cam2_from_velo"]["t"].tolist()},
             "source_rate_hz": float((len(times)-1)/((times[-1]-times[0])/1e9)),
             "image_interval_ms_p50_p95": np.percentile(np.diff(times)/1e6, [50,95]).tolist(),
             "absolute_stereo_skew_ms_p50_p95": np.percentile(np.abs(dataset["stereo_delta_s"])*1000,[50,95]).tolist(),
             "absolute_lidar_camera_stamp_skew_ms_p50_p95": np.percentile(np.abs(velo_times-times)/1e6,[50,95]).tolist(),
             "scan_duration_ms_p50_p95": np.percentile((scan_end-scan_start)/1e6,[50,95]).tolist(),
             "timestamp_timezone": "not specified; original nanosecond timestamp strings interpreted only for relative differences",
             "projection": "P_rect_02 @ R_rect_00 @ T_cam0_from_velo; cam2 translation retains x/y/z",
             "stereo": "horizontal disparity from already-rectified images; exact horizontal P equations; no second distortion correction"}
    Path("reports/kitti_dataset_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps({"frames":len(rows),"fps":report["pipeline_fps"],"processing_p95_ms":report["stages"]["processing_ms"]["p95"],
                      "detections":detections_total,"positions":valid_positions,"dropped":len(skipped)},indent=2),flush=True)
    return report
