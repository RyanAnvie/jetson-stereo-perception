"""Compare target backends on byte-identical saved tensors and full-sequence outputs."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from multimodal_perception.detection import close_detector, infer_tensor, load_detector, preprocess
from multimodal_perception.kitti import load_kitti_frame, open_kitti
from multimodal_perception.kitti_evaluation import match_detections
from multimodal_perception.replay import environment


def statistics(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if not len(values):
        return {"count": 0}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite comparison values")
    return {"count": int(values.size), "mean": float(values.mean()), "max": float(values.max()),
            "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99))}


def export_reference(config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(config["opencv_threads"])
    dataset = open_kitti(config)
    detector = load_detector(config["model"], config["onnx_threads"])
    manifest = {"config": config, "model": detector["metadata"], "environment": environment(), "frames": []}
    try:
        for index in np.flatnonzero(dataset["valid"]):
            sample = load_kitti_frame(dataset, int(index), include_reference=False)
            tensor, ratio = preprocess(sample["left"])
            output, _ = infer_tensor(tensor, detector)
            filename = f"frame_{index:04d}.npz"
            np.savez_compressed(directory / filename, tensor=tensor, output=output, ratio=ratio,
                                image_shape=np.array(sample["left"].shape))
            manifest["frames"].append({"index": int(index), "file": filename,
                                       "tensor_sha256": hashlib.sha256(tensor.tobytes()).hexdigest()})
    finally:
        close_detector(detector)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {len(manifest['frames'])} exact CPU inputs and raw outputs", flush=True)


def read_outputs(report):
    path = Path(report)
    metadata = json.loads(path.read_text())
    rows = [json.loads(line) for line in path.with_name(path.stem + "_outputs.jsonl").read_text().splitlines()]
    records = {row["index"]: row for row in rows}
    if len(records) != len(rows) or len(rows) != metadata["timed_frames"]:
        raise ValueError("Duplicate or missing pipeline outputs")
    return metadata, records


def compare_predictions(reference_report, candidate_report):
    ref_meta, reference = read_outputs(reference_report)
    candidate_meta, candidate = read_outputs(candidate_report)
    if reference.keys() != candidate.keys():
        raise ValueError("Full-sequence comparison requires identical frame indices")
    excluded_config = {"backend", "engine", "runtime_label"}
    ref_config = {k: v for k, v in ref_meta["config"].items() if k not in excluded_config}
    candidate_config = {k: v for k, v in candidate_meta["config"].items() if k not in excluded_config}
    if ref_config != candidate_config or ref_meta["model"]["sha256"] != candidate_meta["model"]["sha256"]:
        raise ValueError("Pipeline model or configuration differs")
    rows, ious, score_errors, xyz_errors, valid_fraction_errors = [], [], [], [], []
    lost, added, status_changes = [], [], []
    for index, ref in reference.items():
        other = candidate[index]
        if ref["source_timestamp_ns"] != other["source_timestamp_ns"] or ref["right_timestamp_ns"] != other["right_timestamp_ns"]:
            raise ValueError("Source timestamps differ")
        left, right = ref["detections"], other["detections"]
        mapping = {d["class"]: d["class"] for d in left}
        matches = match_detections(right, left, mapping, .5)
        used_left, used_right = set(), set()
        for match in matches:
            li, ri = match["label_index"], match["detection_index"]
            used_left.add(li)
            used_right.add(ri)
            a, b = left[li], right[ri]
            score_delta = b["score"] - a["score"]
            row = {"frame": index, "reference_index": li, "candidate_index": ri,
                   "class": a["class"], "iou": match["iou"], "score_delta": score_delta}
            ious.append(match["iou"])
            score_errors.append(abs(score_delta))
            pa, pb = a["position"], b["position"]
            if pa["frame"] != "kitti_cam2_rect" or pb["frame"] != pa["frame"] or pa["units"] != "m" or pb["units"] != "m":
                raise ValueError("Coordinate frame or units changed")
            row["valid_fraction_delta"] = pb["valid_depth_fraction"] - pa["valid_depth_fraction"]
            valid_fraction_errors.append(abs(row["valid_fraction_delta"]))
            if pa["status"] != pb["status"]:
                status_changes.append({"frame": index, "reference": a, "candidate": b})
            if pa["point_m"] is not None and pb["point_m"] is not None:
                delta = np.array(pb["point_m"]) - np.array(pa["point_m"])
                row["xyz_delta_m"] = delta.tolist()
                xyz_errors.append(delta)
            rows.append(row)
        lost.extend({"frame": index, "detection": detection} for j, detection in enumerate(left) if j not in used_left)
        added.extend({"frame": index, "detection": detection} for j, detection in enumerate(right) if j not in used_right)
    xyz = np.asarray(xyz_errors).reshape(-1, 3)
    return {"frames": len(reference), "matching": "same class, greedy descending IoU >= 0.5; diagnostic only",
            "matched": len(rows), "lost_count": len(lost), "added_count": len(added),
            "reference_detections": ref_meta["detections"], "candidate_detections": candidate_meta["detections"],
            "reference_valid_positions": ref_meta["valid_positions"], "candidate_valid_positions": candidate_meta["valid_positions"],
            "reference_invalid_depth": ref_meta["detections"] - ref_meta["valid_positions"],
            "candidate_invalid_depth": candidate_meta["detections"] - candidate_meta["valid_positions"],
            "iou": statistics(ious), "score_abs_delta": statistics(score_errors),
            "xyz_abs_delta_m": {axis: statistics(np.abs(xyz[:, i])) for i, axis in enumerate("XYZ")},
            "xyz_distance_m": statistics(np.linalg.norm(xyz, axis=1)),
            "valid_depth_fraction_abs_delta": statistics(valid_fraction_errors),
            "depth_status_changes": status_changes, "lost": lost, "added": added, "matches": rows,
            "near_score_threshold_cases": [case for case in lost + added
                                           if abs(case["detection"]["score"] - ref_config["score_threshold"]) <= .02],
            "near_threshold_definition": "within 0.02 of configured score cutoff; descriptive, not an acceptance tolerance"}


def compare(config, directory, engines, reference_report, candidate_reports, output_path):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if config != manifest["config"]:
        raise ValueError("Reference tensor export configuration differs")
    dataset = open_kitti(config)
    cv2.setNumThreads(config["opencv_threads"])
    report = {"acceptance": "No numerical acceptance tolerance specified; measurements only",
              "reference": manifest, "candidate_environment": environment(), "comparisons": []}
    reference_meta, reference_outputs = read_outputs(reference_report)
    if set(reference_outputs) != {frame["index"] for frame in manifest["frames"]}:
        raise ValueError("Reference pipeline and tensor export frame sets differ")
    if reference_meta["model"]["sha256"] != manifest["model"]["sha256"]:
        raise ValueError("Reference model changed")
    for engine, pipeline_report in zip(engines, candidate_reports):
        detector = load_detector(config["model"], config["onnx_threads"], "tensorrt", engine)
        absolute, relative, frame_rows = [], [], []
        near_zero = 0
        try:
            candidate_meta, _ = read_outputs(pipeline_report)
            if candidate_meta["model"]["engine_sha256"] != detector["metadata"]["engine_sha256"]:
                raise ValueError("Pipeline engine differs from raw-output comparison engine")
            for frame in manifest["frames"]:
                with np.load(directory / frame["file"], allow_pickle=False) as saved:
                    tensor, expected = saved["tensor"], saved["output"]
                sample = load_kitti_frame(dataset, frame["index"], include_reference=False)
                current_tensor, _ = preprocess(sample["left"])
                if hashlib.sha256(tensor.tobytes()).hexdigest() != frame["tensor_sha256"] or not np.array_equal(tensor, current_tensor):
                    raise ValueError("Preprocessed tensor changed across environments")
                actual, _ = infer_tensor(tensor, detector)
                if actual.shape != expected.shape or not np.isfinite(actual).all():
                    raise ValueError("Invalid candidate network output")
                error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                ref_abs = np.abs(expected.astype(np.float64))
                absolute.append(error.ravel())
                relative.append((error / np.maximum(ref_abs, 1e-6)).ravel())
                near_zero += int(np.count_nonzero(ref_abs < 1e-6))
                frame_rows.append({"index": frame["index"], "absolute": statistics(error)})
            entry = {"engine": str(engine), "model": detector["metadata"], "identical_preprocessed_frames": len(frame_rows),
                     "raw_output_absolute_error": statistics(np.concatenate(absolute)),
                     "raw_output_relative_error": statistics(np.concatenate(relative)),
                     "relative_error_definition": "abs(candidate-reference) / max(abs(reference), 1e-6)",
                     "reference_values_below_denominator_floor": near_zero,
                     "raw_output_frames": frame_rows,
                     "predictions": compare_predictions(reference_report, pipeline_report)}
            report["comparisons"].append(entry)
            print(json.dumps({"engine": str(engine), "raw_error": entry["raw_output_absolute_error"],
                              "matched": entry["predictions"]["matched"], "lost": entry["predictions"]["lost_count"],
                              "added": entry["predictions"]["added_count"]}), flush=True)
        finally:
            close_detector(detector)
    Path(output_path).write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/kitti.yaml")
    parser.add_argument("--reference-dir", default="results/jetson/cpu_reference")
    parser.add_argument("--export-reference", action="store_true")
    parser.add_argument("--engine", action="append", default=[])
    parser.add_argument("--cpu-report", default="reports/jetson/cpu_benchmark.json")
    parser.add_argument("--candidate-report", action="append", default=[])
    parser.add_argument("--report", default="reports/jetson/backend_comparison.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.export_reference:
        export_reference(config, args.reference_dir)
    else:
        if not args.engine or len(args.engine) != len(args.candidate_report):
            parser.error("Provide one --candidate-report per --engine")
        compare(config, args.reference_dir, args.engine, args.cpu_report, args.candidate_report, args.report)
