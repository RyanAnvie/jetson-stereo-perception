import csv
import json
from pathlib import Path
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import yaml

from multimodal_perception.dashboard import title_card
from multimodal_perception.dataset import open_dataset
from multimodal_perception.joint_detection import load_joint_detector, close_joint_detector, process_joint_sample
from multimodal_perception.joint_dashboard import render_joint_dashboard
from multimodal_perception.replay import environment, summarize_timings
from multimodal_perception.thermal_fusion import load_fusion_calibration
from multimodal_perception.rig_motion import load_exposure_sample
from multimodal_perception.joint_prefetch import prepared_samples
from multimodal_perception.joint_workers import processed_samples


def peak_rss_kib():
    try:
        import resource
        import sys
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value/1024 if sys.platform == "darwin" else value
    except ImportError:
        return None


def run_joint(config_path, limit, video, report_path, backend=None):
    config = yaml.safe_load(Path(config_path).read_text())
    if backend is not None:
        config["backend"] = backend
    if limit < 0 or config["warmup_frames"] < 0:
        raise ValueError("Frame counts must be nonnegative")
    if config["video_timeline"] != "compact_valid":
        raise ValueError("Joint replay presents valid frames once, without gap holds")
    cv2.setNumThreads(config["opencv_threads"])
    dataset = open_dataset(config["dataset_root"], config["max_time_error_s"])
    calibration = load_fusion_calibration(config["dataset_root"], config["thermal_profile"], config["thermal_refinement"])
    worker_count = int(config.get("inference_workers", 1))
    if not 1 <= worker_count <= 3 or (worker_count > 1 and config["backend"] != "tensorrt"):
        raise ValueError("Inference worker count must be 1 to 3; parallel inference requires TensorRT")
    detectors = []
    try:
        for _ in range(worker_count):
            detector = load_joint_detector(config["model"], config.get("engine"), config["backend"], config["onnx_threads"])
            detectors.append(detector)
            if config.get("cuda_graph", False):
                if detector["backend"] != "tensorrt":
                    raise ValueError("CUDA graph requires TensorRT")
                detector["trt"]["cuda_graph_requested"] = True
        return run_loaded_joint(dataset, calibration, detectors[0], config, limit, video, report_path, detectors[1:])
    finally:
        for detector in detectors:
            close_joint_detector(detector)


def run_loaded_joint(dataset, calibration, detector, config, limit, video, report_path, additional_detectors=()):
    indices = np.flatnonzero(dataset["valid"]).tolist()
    if limit:
        indices = indices[:limit]
    if not indices:
        raise ValueError("No valid groups")
    render_bound = int(config.get("render_queue_frames", 2))
    if not 1 <= render_bound <= 2:
        raise ValueError("Render queue must contain 1 or 2 frames")
    detectors = [detector, *additional_detectors]
    for current in detectors:
        for index in indices[:config["warmup_frames"]]:
            process_joint_sample(load_exposure_sample(dataset, index, calibration, config.get("temporal_compensation", False)), current, calibration, config)
    writer = None
    card_frames = int(config.get("video_title_frames", 15))
    if card_frames < 0:
        raise ValueError("Title frame count must be nonnegative")
    if video:
        video = Path(video)
        video.parent.mkdir(parents=True, exist_ok=True)
        if config.get("video_encoder", "mjpeg") == "jetson_h264":
            from multimodal_perception.jetson_video import JetsonH264Writer
            writer = JetsonH264Writer(video, bitrate=int(config.get("video_bitrate", 1800000)))
        elif config.get("video_encoder", "mjpeg") == "mjpeg":
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 800))
        else:
            raise ValueError("Unknown video encoder")
        if not writer.isOpened():
            writer.release()
            raise RuntimeError("Video writer failed")
    rows, outputs = [], []
    ahead = int(config.get("prefetch_frames", 0))
    preparation_workers = int(config.get("preparation_workers", 1))
    prepared = prepared_samples(dataset, indices, calibration, config.get("temporal_compensation", False), ahead, preparation_workers) if ahead else None
    processed = None
    if len(detectors) > 1 or config.get("inference_dispatch") == "independent":
        source = prepared if prepared is not None else (load_exposure_sample(dataset, index, calibration, config.get("temporal_compensation", False)) for index in indices)
        processed = processed_samples(source, detectors, calibration, config)
    render_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-output") if video and config.get("async_video", False) else None
    render_pending = deque()

    def render_encode(sample, detections, fusion, timing, offset, snapshot, submitted):
        begin = time.perf_counter()
        canvas = render_joint_dashboard(sample, detections, fusion, timing, offset)
        drawn = time.perf_counter()
        writer.write(canvas)
        encoded = time.perf_counter()
        if snapshot:
            cv2.imwrite(str(video.parent / f"joint_{sample['index']:04d}.jpg"), canvas)
        return {"render_ms": (drawn-begin)*1000, "encode_ms": (time.perf_counter()-drawn)*1000,
                "render_queue_wait_ms": (begin-submitted)*1000,
                "encoder_submit_ms": (encoded-drawn)*1000,
                "input_to_encoder_ms": (encoded-sample["pipeline_input_started"])*1000,
                "admission_to_encoder_ms": (encoded-sample.get("preparation_submitted", sample["pipeline_input_started"]))*1000,
                "input_to_output_ms": (time.perf_counter()-sample["pipeline_input_started"])*1000}

    def finish_render():
        future, row = render_pending.popleft()
        row.update(future.result())
    try:
        if writer is not None and card_frames:
            card = title_card("Two modalities, one detector", [
                "LCAFNet: separate RGB / thermal branches and learned feature fusion.",
                "M3FD pretrained model; no TartanRGBT accuracy certification.",
                "Fixed RGB-thermal rig extrinsics; clock / motion treatment is recorded.",
                "Each valid frame appears once; source gaps are cut, not held.",
                "10 fps presentation is separate from measured processing speed.",
                "Thermal is processed intensity; depth is a supplied estimate."])
            for _ in range(card_frames):
                writer.write(card)
        first_timestamp = float(dataset["target"][indices[0]])
        started = time.perf_counter()
        for position, index in enumerate(indices):
            tick = time.perf_counter()
            if processed is not None:
                sample, detections, fusion, timing = next(processed)
                loaded = time.perf_counter()
            else:
                sample = next(prepared) if prepared is not None else load_exposure_sample(dataset, index, calibration, config.get("temporal_compensation", False))
                loaded = time.perf_counter()
                sample["inference_started"] = loaded
                detections, fusion, timing = process_joint_sample(sample, detector, calibration, config)
                sample["inference_finished"] = time.perf_counter()
            delivered = time.perf_counter()
            if sample["index"] != index:
                raise RuntimeError("Inference pipeline delivered frames out of source order")
            sample["pipeline_input_started"] = sample.get("preparation_started", tick)
            timing["input_wait_ms"] = (loaded-tick)*1000
            timing["io_decode_ms"] = sample.get("preparation_timing", {}).get("io_decode_ms", (loaded-tick)*1000)
            preparation_start = sample.get("preparation_started", tick)
            preparation_end = sample.get("preparation_finished", loaded)
            timing["preparation_queue_ms"] = (preparation_start-sample.get("preparation_submitted", preparation_start))*1000
            timing["preparation_active_ms"] = (preparation_end-preparation_start)*1000
            timing["prepared_wait_ms"] = (sample["inference_started"]-preparation_end)*1000
            timing["inference_active_ms"] = (sample["inference_finished"]-sample["inference_started"])*1000
            timing["completed_wait_ms"] = (delivered-sample["inference_finished"])*1000
            timing["input_to_detection_ms"] = (sample["inference_finished"]-preparation_start)*1000
            timing["input_to_delivery_ms"] = (delivered-preparation_start)*1000
            offset = sample["timestamp_s"]-first_timestamp
            render_started = time.perf_counter()
            canvas = render_joint_dashboard(sample, detections, fusion, timing, offset) if writer is not None and render_executor is None else None
            timing["render_ms"] = (time.perf_counter()-render_started)*1000
            encode_started = time.perf_counter()
            encoded_at = encode_started
            if writer is not None and render_executor is None:
                writer.write(canvas)
                encoded_at = time.perf_counter()
                if position in [0, len(indices)//2, len(indices)-1] or (detections and index % 10 == 0):
                    cv2.imwrite(str(video.parent / f"joint_{index:04d}.jpg"), canvas)
            timing["encode_ms"] = (time.perf_counter()-encode_started)*1000
            timing["total_frame_ms"] = (time.perf_counter()-tick)*1000
            timing["input_to_output_ms"] = (time.perf_counter()-sample.get("preparation_started", tick))*1000
            timing["input_to_encoder_ms"] = (encoded_at-preparation_start)*1000 if writer is not None else 0.
            timing["admission_to_encoder_ms"] = (encoded_at-sample.get("preparation_submitted", preparation_start))*1000 if writer is not None else 0.
            timing["render_queue_wait_ms"] = 0.
            timing["encoder_submit_ms"] = (encoded_at-encode_started)*1000
            row = {"index": index, "timestamp_s": sample["timestamp_s"], **timing}
            rows.append(row)
            if render_executor is not None:
                snapshot = position in [0, len(indices)//2, len(indices)-1] or (detections and index % 10 == 0)
                future = render_executor.submit(render_encode, sample, detections, fusion, timing.copy(), offset, snapshot, time.perf_counter())
                render_pending.append((future, row))
                if len(render_pending) >= render_bound:
                    finish_render()
                row["total_frame_ms"] = (time.perf_counter()-tick)*1000
            outputs.append({"index": index, "timestamp_s": sample["timestamp_s"],
                            "rgb_timestamp_s": sample["rgb_timestamp_s"], "thermal_timestamp_s": sample["thermal_timestamp_s"],
                            "detections": detections, "visible_thermal_samples": len(fusion["points"]),
                            "rgb_observed_fraction": float(np.mean(fusion["rgb_observed_mask"] > 0)),
                            "rgb_near_hole_fill_fraction": float(np.mean((fusion["rgb_usable_mask"] > 0) & (fusion["rgb_observed_mask"] == 0))),
                            "residual_alignment": fusion["residual_alignment"],
                            "time_alignment": fusion["time_alignment"],
                            "recognition": "LCAFNet RGB + thermal learned feature fusion", "position_units": "m"})
            if position % 25 == 0:
                print(f"Joint {position+1}/{len(indices)} frame={index} detections={len(detections)} processing={timing['processing_ms']:.1f}ms", flush=True)
        while render_pending:
            finish_render()
        elapsed = time.perf_counter()-started
        report = summarize_timings(rows, elapsed)
        if writer is not None and card_frames:
            card = title_card("Joint recognition, measured", [
                f"{len(rows)} valid frames; no held gap frames.",
                "Full pipeline speed and final encoder drain are recorded in the report.",
                "RGB and thermal both enter learned feature fusion before detection.",
                "Modality ablations and calibration residuals are saved in reports.",
                "Approximate registration; dataset shift and false positives remain.",
                "Data: CMU AirLab TartanRGBT. Joint detector: WenCongWu/LCAFNet."])
            for _ in range(card_frames):
                writer.write(card)
    finally:
        if processed is not None:
            processed.close()
        if prepared is not None:
            prepared.close()
        if render_executor is not None:
            render_executor.shutdown(wait=True, cancel_futures=True)
        if writer is not None:
            finalize_started = time.perf_counter()
            writer.release()
            encoder_finalize_ms = (time.perf_counter()-finalize_started)*1000
        else:
            encoder_finalize_ms = 0.
    report = summarize_timings(rows, elapsed+encoder_finalize_ms/1000)
    env = environment()
    env["compute_device"] = "NVIDIA Jetson GPU + CPU" if detector["backend"] == "tensorrt" else "CPU"
    env.pop("available_gpu_not_used", None)
    report.update(environment=env, config=config, model=detector["metadata"], calibration=calibration["evidence"],
                  dataset_revision=dataset["manifest"]["revision"], sequence=dataset["manifest"]["sequence"],
                  warmup_frames=min(len(indices), config["warmup_frames"]), source_groups=len(dataset["target"]),
                  selected_valid_groups=len(indices), dropped_valid_frames=0,
                  invalid_source_groups=int((~dataset["valid"]).sum()),
                  excluded=[{"index": i, "reasons": dataset["reasons"][i]} for i in np.flatnonzero(~dataset["valid"]).tolist()],
                  total_detections=sum(len(o["detections"]) for o in outputs),
                  valid_positions=sum(d["position"]["status"] == "ok" for o in outputs for d in o["detections"]),
                  accepted_residual_alignment_groups=sum(o["residual_alignment"]["accepted"] for o in outputs),
                  motion_compensated_groups=sum(o["time_alignment"]["compensated"] for o in outputs),
                  uncompensated_motion_groups=[o["index"] for o in outputs if o["time_alignment"]["enabled"] and not o["time_alignment"]["compensated"]],
                  includes_render=bool(video), includes_video_encode=bool(video),
                  video_timeline="compact_valid" if video else None, held_video_frames=0,
                  video_frames=(len(rows)+card_frames*2) if video else None,
                  video_title_frames=card_frames*2 if video else 0,
                  video_encoder=config.get("video_encoder", "mjpeg") if video else None,
                  encoder_finalize_ms=encoder_finalize_ms,
                  encoder_command=getattr(writer, "command", None),
                  latency_definition="Decode start through detection output, or through rendered-frame submission to the encoder; includes bounded queue residence, excludes physical acquisition. Hardware encoder final drain is accounted in wall FPS.",
                  detailed_latency_definition="input_to_output_ms starts at input preparation and includes optional snapshot writes. input_to_encoder_ms stops at writer.write return before snapshots. admission_to_encoder_ms also includes waiting for a CPU preparation worker. These intervals exclude camera exposure and encoder completion.",
                  source_gap_ticks_removed=max(0, round((outputs[-1]["timestamp_s"]-first_timestamp)*10)+1-len(rows)) if video else 0,
                  processing_definition="loaded sample -> optional RGB rig motion + dense registration + both-input model + decoded detections + estimated 3D",
                  fps_definition="completed valid groups / wall duration including I/O, selected rendering and video encoding, queue drain and encoder finalization; excludes warmup and title cards",
                  acquisition_latency="not measured; offline replay", detection_accuracy="not measured on labeled TartanRGBT",
                  depth_source="released FoundationStereo estimates; not computed online")
    report.update(registration_backend=config.get("registration_backend", "numpy"),
                  gpu_registration=detector["cuda_registration"].metadata if "cuda_registration" in detector else None,
                  gpu_registration_memory_scope="per inference worker; mapped_shared_bytes is a subset of persistent_pinned_bytes",
                  gpu_registration_all_workers=[d["cuda_registration"].metadata for d in detectors if "cuda_registration" in d],
                  preparation_overlap=bool(ahead), preparation_queue_bound=ahead,
                  preparation_workers=preparation_workers,
                  inference_workers=len(detectors), inference_queue_bound=len(detectors) if processed is not None else 0,
                  inference_dispatch=config.get("inference_dispatch", "inline_refill") if processed is not None else "single",
                  render_encode_overlap=render_executor is not None, render_queue_bound=render_bound if render_executor is not None else 0,
                  decoded_frame_cache_bound=ahead+4 if ahead else 0,
                  timing_note="Preparation, independent inference streams and optional rendering overlap; do not add stage medians or invert per-worker latency for pipeline FPS. CUDA preprocessing time is CPU enqueue time; its device work is included in inference wall time. Wall FPS includes first preparation and final output drain. input_to_output_ms includes queue latency.",
                  process_peak_rss_kib=peak_rss_kib())
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    path.with_suffix(".outputs.json").write_text(json.dumps(outputs, indent=2) + "\n")
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    print(json.dumps({k: report[k] for k in ["timed_frames", "pipeline_fps", "total_detections", "valid_positions", "held_video_frames"]}, indent=2))
    return report
